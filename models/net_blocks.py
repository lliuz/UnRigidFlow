import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomConv2D(nn.Conv2d):
    # add `pad_same` option to native conv2d
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, pad_same=False):
        self.pad_same = pad_same
        super(CustomConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride=stride,
            padding=padding, dilation=dilation, groups=groups, bias=bias)

    def _conv2d_same_padding(self, input, weight, bias, stride, dilation, groups):

        def _compute_padding(i, k, s, d):
            len_o = (i + s - 1) // s
            return max(0, (len_o - 1) * s + (k - 1) * d + 1 - i)

        pad_row = _compute_padding(input.size(2), weight.size(2), stride[0], dilation[0])
        pad_col = _compute_padding(input.size(3), weight.size(3), stride[1], dilation[1])

        if pad_row % 2:
            input = F.pad(input, (0, 0, 0, 1))  # padding the last row
        if pad_col % 2:
            input = F.pad(input, (0, 1, 0, 0))  # padding the last column

        return F.conv2d(input, weight, bias, stride,
                        padding=(pad_row // 2, pad_col // 2),
                        dilation=dilation, groups=groups)

    def forward(self, input):
        if self.pad_same:
            return self._conv2d_same_padding(input, self.weight, self.bias, self.stride,
                                             self.dilation, self.groups)
        else:
            return super(CustomConv2D, self).forward(input)


def conv(batch_norm, in_planes, out_planes,
         kernel_size=3, stride=1, dilation=1, pad_same=True):
    if batch_norm:
        return nn.Sequential(
            CustomConv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=((kernel_size - 1) * dilation) // 2,
                         dilation=dilation, bias=True, pad_same=pad_same),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.1, inplace=True)
        )
    else:
        return nn.Sequential(
            CustomConv2D(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=((kernel_size - 1) * dilation) // 2,
                         dilation=dilation, bias=True, pad_same=pad_same),
            nn.LeakyReLU(0.1, inplace=True)
        )


class FeaturePyramidExtractor(nn.Module):

    def __init__(self, lv_chs, bn=False):
        super(FeaturePyramidExtractor, self).__init__()

        self.convs = []
        for l, (ch_in, ch_out) in enumerate(zip(lv_chs[:-1], lv_chs[1:])):
            layer = nn.Sequential(
                conv(bn, ch_in, ch_out, stride=2),
                conv(bn, ch_out, ch_out)
            )
            self.add_module('Feature(Lv{})'.format(l + 1), layer)
            self.convs.append(layer)

    def forward(self, x):
        feature_pyramid = []
        for conv in self.convs:
            x = conv(x)
            feature_pyramid.append(x)

        return feature_pyramid[::-1]


class OpticalFlowEstimator(nn.Module):
    def __init__(self, ch_in, ch_out=2, bn=False):
        super(OpticalFlowEstimator, self).__init__()
        self.conv1 = conv(bn, ch_in, 128)
        self.conv2 = conv(bn, 128, 128)
        self.conv3 = conv(bn, 128 + 128, 96)
        self.conv4 = conv(bn, 128 + 96, 64)
        self.conv5 = conv(bn, 96 + 64, 32)

        self.final_out = 32
        self.predict_flow = CustomConv2D(64 + 32, ch_out, kernel_size=3, stride=1,
                                         padding=1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(torch.cat([x1, x2], dim=1))
        x4 = self.conv4(torch.cat([x2, x3], dim=1))
        x5 = self.conv5(torch.cat([x3, x4], dim=1))

        flow = self.predict_flow(torch.cat([x4, x5], dim=1))

        return x5, flow


class ContextNetwork(nn.Module):

    def __init__(self, ch_in, ch_out=2, bn=False):
        super(ContextNetwork, self).__init__()
        self.convs = nn.Sequential(
            conv(bn, ch_in, 128, 3, 1, 1),
            conv(bn, 128, 128, 3, 1, 2),
            conv(bn, 128, 128, 3, 1, 4),
            conv(bn, 128, 96, 3, 1, 8),
            conv(bn, 96, 64, 3, 1, 16),
            conv(bn, 64, 32, 3, 1, 1),
        )
        self.predict_flow = CustomConv2D(32, ch_out, kernel_size=3, stride=1,
                                         padding=1)

    def forward(self, x):
        x = self.convs(x)
        return self.predict_flow(x)