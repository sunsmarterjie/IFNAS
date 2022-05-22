from operations import *
from torch.autograd import Variable
from genotypes import PRIMITIVES

lenOP = len(PRIMITIVES)


class Stage(nn.Module):
    def __init__(self, geno_stage, channel_in, channel_out, stride=1, length=4):
        super(Stage, self).__init__()
        self._geno_stage = geno_stage
        self.stride = stride
        self._ops = nn.ModuleList()
        self._length = length
        for n in geno_stage.values():
            for l in n:
                self._ops.append(OPS[l[0]](channel_out, stride=1))

        if stride == 2:
            self.prepocess = nn.Sequential(
                SeparableConv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1, bias=False),
                # SeparableConv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1, bias=False),
                nn.MaxPool2d(3, 2, 1),
            )
            self.skip = nn.Sequential(
                # nn.ReLU(),
                nn.Conv2d(channel_in, channel_out, 1, 2, bias=False),
                nn.BatchNorm2d(channel_out)
            )
        else:
            self.prepocess = nn.Sequential(
                SeparableConv2d(channel_in, channel_out, kernel_size=3, stride=1, padding=1, bias=False),
                # SeparableConv2d(channel_out, channel_out, kernel_size=3, stride=1, padding=1, bias=False),
                # nn.MaxPool2d(3, 2, 1),
            )
            self.skip = None

        self.nodes_to_used = []  # used to save nodes that have to be saved
        for n in self._geno_stage.values():
            for e in n:
                self.nodes_to_used.append(e[1])
        # create nodes_to_used end

    def forward(self, x, weights_edge, weights_op, turns):
        if self.skip is None:
            inp = self.prepocess(x) + x
        else:
            inp = self.prepocess(x) + self.skip(x)
        #################### input end ##################
        layers = dict()
        layers[0] = inp
        th_ops = 0  # index of the operator in self._ops
        node = None
        nodes_to_used = self.nodes_to_used.copy()
        for k, n in enumerate(self._geno_stage.keys()):
            th_n = int(n[4:])  # used for as key for layers to save features
            input_inds = [link[1] for j, link in enumerate(self._geno_stage[n])][::2]
            if k % (self._length + 1) == turns:
                # print('k:', k)
                weight_edge = weights_edge[th_n - 1]
                weight_op = weights_op[th_n - 1]
            else:
                weight_edge = torch.tensor([0.] * min(k, self._length - 1) + [1.],
                                      requires_grad=False).cuda()
                weight_op = torch.ones((min(k + 1, self._length), lenOP), requires_grad=False).cuda()
            # print(weights.shape)
            input_nodes = [
                self._ops[th_ops + j * 2](layers[l]) * weight_op[j][0] + self._ops[th_ops + j * 2 + 1](layers[l]) *
                weight_op[j][1] for j, l in enumerate(input_inds)]
            node = sum([input_node * weight_edge[j] * 2 / len(input_nodes) for j, input_node in enumerate(input_nodes)])
            th_ops += len(input_inds)
            for _ in input_inds:
                nodes_to_used.pop(0)
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


class AuxiliaryHeadImageNet(nn.Module):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(C, C * 2, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=2, padding=0, count_include_pad=False),
            nn.Conv2d(C * 2, 128, 1, bias=False),
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


class Network(nn.Module):
    def __init__(self, C, num_classes, genotype, auxiliary=True, func='sigmoid', length=4, turns=0):
        super(Network, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._genotype = genotype
        self._func = func
        self._length = length
        self._turns = turns
        self._auxiliary = auxiliary

        self.stem1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            # SeparableConv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.stem2 = nn.Sequential(
            SeparableConv2d(32, 48, kernel_size=3, stride=1, padding=1, bias=False),
            SeparableConv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(3, 2, 1)
        )
        self.skip2 = nn.Sequential(
            nn.Conv2d(32, 48, 1, 2, bias=False),
            nn.BatchNorm2d(48)
        )
        self.stem3 = nn.Sequential(
            SeparableConv2d(48, C, kernel_size=3, stride=1, padding=1, bias=False),
            SeparableConv2d(C, C, kernel_size=3, stride=1, padding=1, bias=False),
            nn.MaxPool2d(3, 2, 1)
        )
        self.skip3 = nn.Sequential(
            nn.Conv2d(48, C, 1, 2, bias=False),
            nn.BatchNorm2d(C)
        )
        # # the channle number of each stage can be stable
        self.stage1 = Stage(genotype[0], self._C, self._C, length=self._length)
        self.stage2 = Stage(genotype[1], self._C, self._C * 2, stride=2, length=self._length)
        self.stage3 = Stage(genotype[2], self._C * 2, self._C * 4, stride=2, length=self._length)

        if self._auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(self._C * 2, num_classes)
        self.final = nn.Sequential(
            SeparableConv2d(self._C * 4, self._C * 16, 3, 1, 1),
            nn.BatchNorm2d(self._C * 16),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(self._C * 16, num_classes)

        self._initialize_alphas()
        # self._initialize_prunes()

    def forward(self, input, turns):
        self.generate_weight()
        logits_aux = None
        s = self.stem1(input)
        s = self.stem2(s) + self.skip2(s)
        s = self.stem3(s) + self.skip3(s)
        ###################################################
        s = self.stage1(s, self.weight1_edge, self.weight1_op, turns)
        s = self.stage2(s, self.weight2_edge, self.weight2_op, turns)
        if self._auxiliary and self.training:
            logits_aux = self.auxiliary_head(s)
        s = self.stage3(s, self.weight3_edge, self.weight3_op, turns)
        out = self.final(s)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits#, logits_aux

    def _initialize_alphas(self):
        # max node id each stage
        num_nodes = [int(list(sta.keys())[-1][4:]) for _, sta in enumerate(self._genotype)]
        self.alphas_stage1, self.betas_stage1 = [], []
        self.alphas_stage2, self.betas_stage2 = [], []
        self.alphas_stage3, self.betas_stage3 = [], []
        for i, j in enumerate(range(num_nodes[0])):
            self.betas_stage1.append(nn.Parameter(1e-5 * torch.randn(1, self._length).cuda()))
            self.alphas_stage1.append(nn.Parameter(1e-5 * torch.randn(min(i + 1, self._length), lenOP).cuda()))
        for i, j in enumerate(range(num_nodes[1])):
            self.betas_stage2.append(nn.Parameter(1e-5 * torch.randn(1, self._length).cuda()))
            self.alphas_stage2.append(nn.Parameter(1e-5 * torch.randn(min(i + 1, self._length), lenOP).cuda()))
        for i, j in enumerate(range(num_nodes[2])):
            self.betas_stage3.append(nn.Parameter(1e-5 * torch.randn(1, self._length).cuda()))
            self.alphas_stage3.append(nn.Parameter(1e-5 * torch.randn(min(i + 1, self._length), lenOP).cuda()))
        self._arch_parameters = [
            self.alphas_stage1,
            self.alphas_stage2,
            self.alphas_stage3,
            self.betas_stage1,
            self.betas_stage2,
            self.betas_stage3,
        ]

    def arch_parameters(self):
        return self._arch_parameters

    def parameter_omega(net, lr):
        parameters = [
            {'params': net.classifier.weight, 'lr': lr},
            {'params': net.classifier.bias, 'lr': lr},
        ]
        parameters.extend([{'params': net.stem1.parameters(), 'lr': lr}])
        parameters.extend([{'params': net.stem2.parameters(), 'lr': lr}])
        parameters.extend([{'params': net.stem3.parameters(), 'lr': lr}])
        parameters.extend([{'params': net.skip2.parameters(), 'lr': lr}])
        parameters.extend([{'params': net.skip3.parameters(), 'lr': lr}])
        parameters.extend([{'params': net.auxiliary_head.parameters(), 'lr': lr}])
        parameters.extend([{'params': net.final.parameters(), 'lr': lr}])
        parameters.extend([{'params': net.stage1.parameters(), 'lr': lr}])
        parameters.extend([{'params': net.stage2.parameters(), 'lr': lr}])
        parameters.extend([{'params': net.stage3.parameters(), 'lr': lr}])
        return parameters

    def parameter_alpha(net, lr_arch):
        parameters = []
#         for para in net.alphas_stage1:
#             parameters += {'params': para, 'lr': lr_arch},
#         for para in net.alphas_stage2:
#             parameters += {'params': para, 'lr': lr_arch},
#         for para in net.alphas_stage3:
#             parameters += {'params': para, 'lr': lr_arch},
        ##################################################
        for para in net.betas_stage1:
            parameters += {'params': para, 'lr': lr_arch},
        for para in net.betas_stage2:
            parameters += {'params': para, 'lr': lr_arch},
        for para in net.betas_stage3:
            parameters += {'params': para, 'lr': lr_arch},

        return parameters

    def prune(self, threshold=0.1):
        for k, s in enumerate(self._genotype[0]):
            mask = torch.sigmoid(self.alphas_stage1[int(s[4:]) - 1]) < threshold
            self.prune_stage1[int(s[4:]) - 1][mask] = 0.0
        for k, s in enumerate(self._genotype[1]):
            mask = torch.sigmoid(self.alphas_stage2[int(s[4:]) - 1]) < threshold
            self.prune_stage2[int(s[4:]) - 1][mask] = 0.0
        for k, s in enumerate(self._genotype[2]):
            mask = torch.sigmoid(self.alphas_stage3[int(s[4:]) - 1]) < threshold
            self.prune_stage3[int(s[4:]) - 1][mask] = 0.0

    def generate_weight(self):
        self.weight1_edge, self.weight1_op = [], []
        self.weight2_edge, self.weight2_op = [], []
        self.weight3_edge, self.weight3_op = [], []
        for k, be in enumerate(self.betas_stage1):
            try:
                #  when one node is delete KeyError occurs here
                inp_length = len(self._genotype[0]['node%d' % (k + 1)])
                # print(inp_length)
            except KeyError:
                inp_length = self._length
            if self._func == 'sigmoid':
                self.weight1_edge.append(torch.sigmoid(be[0, :min(self._length, (k + 1), inp_length)]))
                self.weight1_op.append(torch.sigmoid(self.alphas_stage1[k]))
            # elif self._func == 'softmax':
            #     self.weight1.append(torch.softmax(al[0, :min(self._length, k + 1, inp_length)], dim=0))
        for k, be in enumerate(self.betas_stage2):
            try:
                #  when one node is delete KeyError occurs here
                inp_length = len(self._genotype[1]['node%d' % (k + 1)])
            except KeyError:
                inp_length = self._length
            if self._func == 'sigmoid':
                self.weight2_edge.append(torch.sigmoid(be[0, :min(self._length, (k + 1), inp_length)]))
                self.weight2_op.append(torch.sigmoid(self.alphas_stage2[k]))
            elif self._func == 'softmax':
                self.weight2.append(torch.softmax(be[0, :min(self._length, k + 1, inp_length)], dim=0))
        for k, be in enumerate(self.betas_stage3):
            try:
                #  when one node is delete KeyError occurs here
                inp_length = len(self._genotype[2]['node%d' % (k + 1)])
            except KeyError:
                inp_length = self._length
            if self._func == 'sigmoid':
                self.weight3_edge.append(torch.sigmoid(be[0, :min(self._length, (k + 1), inp_length)]))
                self.weight3_op.append(torch.sigmoid(self.alphas_stage3[k]))
            elif self._func == 'softmax':
                self.weight3.append(torch.softmax(be[0, :min(self._length, k + 1, inp_length)], dim=0))
        self.weight_edge = [self.weight1_edge, self.weight2_edge, self.weight3_edge]
        self.weight_op = [self.weight1_op, self.weight2_op, self.weight3_op]


from genotypes import supernet_genotype, show_genotype, TAAS_sample, geno_image

if __name__ == '__main__':
    geno = supernet_genotype(18, 20, 18, length=4)
    # show_genotype(geno)
    # geno = TAAS_sample(geno, turns=3)
    # show_genotype(geno)
    # assert False

    model = Network(16, 100, geno, func='sigmoid', turns=3)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)[0]
    print(y.shape)
    y = y.sum()
    y.backward()
