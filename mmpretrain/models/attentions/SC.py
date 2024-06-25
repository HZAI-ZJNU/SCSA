import torch
import torch.nn as nn
from mmengine.model import BaseModule

__all__ = ['SC']


class SRU(BaseModule):
    """
    Spatial Reconstruction Unit

    Reduce redundancy in spatial dimension

    main parts:
    1. Separate
        -Separate informative feature maps from less informative ones corresponding to the spatial content,
        so that, we can reconstruct low redundancy feature.

    2. Reconstruction
        -Interacting between different channels(informative channels and less informative channels)
        strengthen the information flow between these channels, so it may improve accuracy,
        reduce redundancy feature and improve feature representation of CNNs.

    """

    def __init__(
            self,
            channels: int,
            group_num: int = 4,
            gate_threshold: float = 0.5,
    ):
        super(SRU, self).__init__()
        self.gn = nn.GroupNorm(group_num, channels)
        self.gate_threshold = gate_threshold
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        # split informative feature maps from less informative ones corresponding to the spatial content
        gn_x = self.gn(x)
        # use gn.weight to measure the variance of spatial pixels for each batch and channel
        w = (self.gn.weight / torch.sum(self.gn.weight)).view(1, -1, 1, 1)
        w = self.sigmoid(w * gn_x)
        infor_mask = w >= self.gate_threshold
        less_infor_maks = w < self.gate_threshold
        x1 = infor_mask * gn_x
        x2 = less_infor_maks * gn_x

        # reconstruct feature with informative feature and less informative feature
        x11, x12 = torch.split(x1, x1.size(1) // 2, dim=1)
        x21, x22 = torch.split(x2, x2.size(1) // 2, dim=1)
        out = torch.cat([x11 + x22, x12 + x21], dim=1)
        return out


class CRU(BaseModule):
    """
    Spatial Reconstruction Unit
    CRU extracts rich representative features through lightweight convolutional operations
    while proceeds redundant features with cheap operation and feature reuse schemes.

    main parts:
    1.Split
        -split and squeeze, divide the spatial features into Xup(upper transformation stage)
         and Xlow(lower transformation stage)
        Xup is serving as a 'Rich Feature Extractor'.
        Xlow is serving as a 'Detail information supplement'

        Xup use GWC(Group-wise Convolution) and PWC(Point-wise Convolution) to replace the expensive standard k x k
        convolutions to extract high-level representative information as well as reduce the computational cost.

        GWC can reduce the amount of parameters and calculations but cut off the information flow between
        channel groups, so another path use PWC to help information flow across feature channels, then sum up
        the output of GWC and PWC to form Y2, which used to extract rich representative information

        Xlow reuses preceding feature map and utilizes 1x1 PWC to serve as a supplementary to Rich Feature Extractor,
        then concat them to form Y2.

    2.Fuse
        -like SKNet, use GAP(Global Average Pooling) and Soft-Attention in channel dimension to restructure
        new feature.
    """

    def __init__(
            self,
            channels: int,
            alpha: float = 0.5,
            squeeze_ratio: int = 2,
            groups: int = 2,
            stride: int = 1,
    ):
        super(CRU, self).__init__()
        self.upper_channel = int(channels * alpha)
        self.low_channel = channels - self.upper_channel
        s_up_c, s_low_c = self.upper_channel // squeeze_ratio, self.low_channel // squeeze_ratio
        self.squeeze_up = nn.Conv2d(self.upper_channel, s_up_c, 1, stride=stride, bias=False)
        self.squeeze_low = nn.Conv2d(self.low_channel, s_low_c, 1, stride=stride, bias=False)

        # up -> GWC + PWC
        self.gwc = nn.Conv2d(s_up_c, channels, 3, stride=1, padding=1, groups=groups)
        self.pwc1 = nn.Conv2d(s_up_c, channels, 1, bias=False)

        # low -> concat(preceding features, PWC)
        self.pwc2 = nn.Conv2d(s_low_c, channels - s_low_c, 1, bias=False)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor):
        up, low = torch.split(x, [self.upper_channel, self.low_channel], dim=1)
        up, low = self.squeeze_up(up), self.squeeze_low(low)

        # up -> GWC + PWC
        y1 = self.gwc(up) + self.pwc1(up)
        # low -> concat(preceding features, PWC)
        y2 = torch.cat((low, self.pwc2(low)), dim=1)

        out = torch.cat((y1, y2), dim=1)
        # enhance the feature maps that include large amount of information
        out_s = self.softmax(self.gap(out))
        out = out * out_s
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        # reduce redundant information
        return out1 + out2


class SC(BaseModule):

    def __init__(
            self,
            channels: int,
            group_num: int = 4,
            gate_threshold: int = 0.5,
            alpha: float = 0.5,
            squeeze_ratio: int = 2,
            groups: int = 2,
            stride: int = 1,
    ):
        super(SC, self).__init__()
        self.sru = SRU(channels, group_num, gate_threshold)
        self.cru = CRU(channels, alpha, squeeze_ratio, groups, stride)

    def forward(self, x: torch.Tensor):
        x = self.sru(x)
        x = self.cru(x)
        return x
