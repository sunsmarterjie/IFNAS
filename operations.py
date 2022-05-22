import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial

OPS = {
    'skip': lambda C, stride: Identity() if stride == 1 else FactorizedReduce(C, C),
    'none': lambda C, stride: Zero(),
    'conv3': lambda C, stride: SeparableConv2d2(C, C, kernel_size=3),
    'sconv3': lambda C, stride: SeparableConv2d(C, C, kernel_size=3),
    'avg_pool_3x3': lambda C, stride: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride: nn.MaxPool2d(3, stride=stride, padding=1),
    # 'conv5': lambda C, stride: SeparableConv2d2(C, C, 5, stride, 2),
    # 'conv3_skip': lambda C, stride: SeparableConv_Skip(C, C),
    # 'conv3_skip': lambda C, stride: InvertedResidual(C, C),
    # 'conv3_skip': lambda C, expand_ratio, image_size, stride: MBConvBlock(C, C, expand_ratio=expand_ratio,
    #                                                                       image_size=image_size),
}


class Zero(nn.Module):
    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, x):
        return x - x


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, inplace=False, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=inplace),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(C_out)
        )
        #for layer in self.op.modules():
        #    if isinstance(layer, nn.Conv2d):
        #        nn.init.xavier_normal_(layer.weight)

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, inplace=False, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=inplace),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Conv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding=1, inplace=False, affine=True):
        super(Conv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=inplace),
            nn.Conv2d(C_in, C_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(C_out),  # memory overflow if use this BN
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
      

      
# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, inplace=False, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=inplace)
        #self.relu = Swish()
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(C_out)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

      

class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, expand_ratio=1, image_size=None, stride=1, padding=1,
                 bias=False, inplace=False):
        super(SeparableConv2d, self).__init__()
        self.ops = nn.Sequential(
            nn.ReLU(inplace=False),
            #Swish(),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, drop_rate=0):
        return self.ops(x)



class SeparableConv2d2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False, inplace=False, affine=True):
        super(SeparableConv2d2, self).__init__()
        self.ops = nn.Sequential(
            nn.ReLU(inplace=False),
            #Swish(),
            nn.Conv2d(in_channels, in_channels, kernel_size, 1, padding, groups=in_channels, bias=bias),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=False),
            #Swish(),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.ops(x)

class SeparableConv2d3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, bias=False, inplace=False, affine=True):
        super(SeparableConv2d3, self).__init__()
        self.ops = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size, 1, padding, groups=in_channels, bias=bias),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(in_channels, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size, 1, padding, groups=in_channels, bias=bias),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(in_channels, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias),
            nn.BatchNorm2d(out_channels, affine=affine),
        )

    def forward(self, x):
        return self.ops(x)


class BottleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=1, dilation=1, bias=False):
        super(BottleConv2d, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, 1, 1, 0, 1, 1, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_channels // 2)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size, stride, padding, dilation, groups=32,
                               bias=bias)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, 1, 1, 0, 1, 1, bias=bias)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        skip = x
        x = self.relu1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x + skip


class Xception_Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps=2, strides=1, start_with_relu=True, grow_first=True):
        super(Xception_Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        rep = []

        filters = in_filters
        if grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(nn.ReLU(inplace=True))
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


class SeparableConv_Skip(nn.Module):
    def __init__(self, in_filters, out_filters, strides=1, start_with_relu=True, grow_first=True):
        super(SeparableConv_Skip, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        rep = []
        # rep.append(nn.ReLU(inplace=True))
        rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
        # rep.append(nn.BatchNorm2d(out_filters))
        filters = out_filters
        # rep.append(nn.ReLU(inplace=True))
        rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
        # rep.append(nn.BatchNorm2d(filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x


# An ordinary implementation of Swish function
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride=1, expand_ratio=4, kernel=3):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            # nn.ReLU6(inplace=True),
            MemoryEfficientSwish(),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, kernel, stride, kernel // 2, groups=inp * expand_ratio,
                      bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            # nn.ReLU6(inplace=True),
            MemoryEfficientSwish(),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


#############################################################################################
###############################  operators used in efficientnet #############################
#############################################################################################

def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.

    Args:
        image_size (int or tuple): Size of the image.

    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow, for a dynamic image size.
       The padding is operated in forward function by calculating dynamically.
    """

    # Tips for 'SAME' mode padding.
    #     Given the following:
    #         i: width or height
    #         s: stride
    #         k: kernel size
    #         d: dilation
    #         p: padding
    #     Output after Conv2d:
    #         o = floor((i+p-((k-1)*d+1))/s+1)
    # If o equals i, i = floor((i+p-((k-1)*d+1))/s+1),
    # => p = (i-1)*s+((k-1)*d+1)-i

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


def drop_connect(inputs, p, training):
    """Drop connect.

    Args:
        input (tensor: BCWH): Input of this structure.
        p (float: 0.0~1.0): Probability of drop connection.
        training (bool): The running mode.

    Returns:
        output: Output after drop connection.
    """
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    if not training:
        return inputs

    batch_size = inputs.shape[0]
    keep_prob = 1 - p

    # generate binary_tensor mask according to probability (p for 0, 1-p for 1)
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)

    output = inputs / keep_prob * binary_tensor
    return output


def get_width_and_height_from_size(x):
    """Obtain height and width from x.

    Args:
        x (int, tuple or list): Data size.

    Returns:
        size: A tuple or list (H,W).
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()


def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.

    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.

    Returns:
        output_image_size: A list [H,W].
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


class MBConvBlock(nn.Module):
    def __init__(self, input_filters, output_filters, expand_ratio=4, image_size=None, stride=1):
        super().__init__()
        # self._block_args = block_args
        self._bn_mom = 0.01  # 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = 0.001  # global_params.batch_norm_epsilon
        self.has_se = True  # (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = True  # block_args.id_skip  # whether to use skip connection and drop connect
        self.se_ratio = 0.25
        self.input_filters = input_filters
        self.output_filters = output_filters
        self.stride = stride

        # Expansion phase (Inverted Bottleneck)
        inp = self.input_filters  # self._block_args.input_filters  # number of input channels
        self.expand_ratio = expand_ratio
        oup = self.input_filters * self.expand_ratio  # number of output channels
        if expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = 3  # self._block_args.kernel_size
        s = stride  # self._block_args.stride
        # print('    ', oup, oup, k, s)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(input_filters * self.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        # Expansion and Depthwise Convolution
        x = inputs
        if self.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self.input_filters, self.output_filters
        if self.id_skip and self.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x
