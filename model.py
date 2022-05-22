from operations import *
from torch.autograd import Variable
# from utils import streng_func
import random


class Stage(nn.Module):
    def __init__(self, geno_stage, channel_in, channel_out, stride=1):
        super(Stage, self).__init__()
        self._geno_stage = geno_stage
        self.stride = stride
        self._ops = nn.ModuleList()
        #self.prepocess = SeparableConv2d2(channel_in, channel_out, kernel_size=3, stride=stride)
        if stride == 1:
            self.prepocess = None
        else:
            self.prepocess = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(channel_out),
                nn.MaxPool2d(3, 2, 1),
                #SeparableConv2d(channel_in, channel_out, kernel_size=3, stride=stride, padding=1, bias=False)
                #FactorizedReduce(channel_in, channel_out),
            )
            self.prepocess2 = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(channel_in, channel_out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(channel_out),
                #nn.MaxPool2d(3, 2, 1),
                SeparableConv2d(channel_out, channel_out, kernel_size=3, stride=stride, padding=1, bias=False)
            )
            self.pre = ReLUConvBN(channel_out * 2, channel_out, 1, 1, 0)
        for n in geno_stage.values():
            for l in n:
                self._ops.append(OPS[l[0]](channel_out, stride=1))

        self.nodes_to_used = []  # used to save nodes that have to be saved
        for n in self._geno_stage.values():
            for e in n:
                self.nodes_to_used.append(e[1])
        # create nodes_to_used end

    def forward(self, x, drop_rate=None):
        if self.prepocess is None:
            inp = x
        else:
            #inp = self.prepocess(x) + self.prepocess2(x)
            inp = self.pre(torch.cat([self.prepocess(x), self.prepocess2(x)], dim=1))
        layers = dict()
        layers[0] = inp
        th_ops = 0  # index of the operator in self._ops
        node = None
        nodes_to_used = self.nodes_to_used.copy()
        for n in self._geno_stage.keys():
            th_n = int(n[4:])  # used for as key for layers to save features
            input_inds = [link[1] for j, link in enumerate(self._geno_stage[n])]
            node = sum([self._ops[th_ops + j](layers[l]) for j, l in enumerate(input_inds)])
            th_ops += len(input_inds)
            for _ in input_inds:
                nodes_to_used.pop(0)
            if th_n in nodes_to_used:
                layers[th_n] = node
            layers = self.filter(layers, nodes_to_used)
        return node

    def filter(self, layers, nodes_to_used):
        layer_ = dict()
        for n in layers.keys():
            if n in nodes_to_used:
                layer_[n] = layers[n]
        return layer_


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(C, C * 2, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # image size = 2 x 2
            nn.Conv2d(C * 2, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        # print('before feature:', x.shape)
        x = self.features(x)
        # print('feature x:', x.shape)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, genotype, auxiliary):
        super(NetworkCIFAR, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._genotype = genotype
        self._auxiliary = auxiliary

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
        )
        # # the channle number of each stage can be stable
        self.stage1 = Stage(genotype[0], self._C, self._C)
        self.stage2 = Stage(genotype[1], self._C, self._C * 2, stride=2)
        self.stage3 = Stage(genotype[2], self._C * 2, self._C * 4, stride=2)

        if self._auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(self._C * 2, num_classes)
        self.final = nn.Sequential(
            SeparableConv2d(self._C * 4, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, input):
        logits_aux = None
        s = self.stem(input)
        print('stem:', s.shape)
        s = self.stage1(s, 0.03)
        print('stage1:', s.shape)
        s = self.stage2(s, 0.06)
        print('stage2:', s.shape)
        if self._auxiliary and self.training:
            logits_aux = self.auxiliary_head(s)
            print('logits aux:', logits_aux.shape)
        s = self.stage3(s, 0.09)
        print('stage3:', s.shape)
        out = self.final(s)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(C, C * 2, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C*2, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            # nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        # print('before feature:', x.shape)
        x = self.features(x)
        # print('feature:', x.shape)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkImageNet(nn.Module):
    def __init__(self, C, num_classes, genotype, auxiliary):
        super(NetworkImageNet, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._genotype = genotype
        self._auxiliary = auxiliary

        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            SeparableConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            SeparableConv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            SeparableConv2d(32, C, kernel_size=3, stride=1, padding=1, bias=False),
            SeparableConv2d(C, C, kernel_size=3, stride=2, padding=1, bias=False),
        )
        # # the channle number of each stage can be stable
        self.stage1 = Stage(genotype[0], self._C, self._C)
        self.stage2 = Stage(genotype[1], self._C, self._C * 3, stride=2)
        self.stage3 = Stage(genotype[2], self._C * 3, self._C * 4, stride=2)

        if self._auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(self._C * 3, num_classes)
        self.final = nn.Sequential(
            SeparableConv2d(self._C * 4, self._C * 16, 3, 1, 1),
            nn.BatchNorm2d(self._C * 16),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(self._C * 16, num_classes)

    def forward(self, input):
        logits_aux = None
        s = self.stem(input)
        # print('stem:', s.shape)
        s = self.stage1(s)
        # print('stage1:', s.shape)
        s = self.stage2(s)
        # print('stage2:', s.shape)
        if self._auxiliary and self.training:
            logits_aux = self.auxiliary_head(s)
            #print('logits aux:', logits_aux.shape)
        s = self.stage3(s)
        # print('stage3:', s.shape)
        out = self.final(s)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


from genotypes import supernet_genotype, show_genotype, TAAS_sample, geno_cifar, geno_image
import utils
if __name__ == '__main__':
    # show_genotype(geno)
    model = NetworkImageNet(64, 1000, geno_image, auxiliary=True)
    print("param size = %fMB", utils.count_parameters_in_MB(model))
    x = torch.randn(2, 3, 224, 224)
    y, _ = model(x)
    y = y.sum()
    y.backward()
