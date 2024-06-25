import typing as t
import torch
import torch.nn as nn
from mmengine.model import BaseModule

__all__ = ['ELA']


class ELA(BaseModule):
    """
    Efficient Local Attention.
    Spatial Attention.

    BN -> GN
    Spatial Conv2D -> Spatial Conv1D respectively
    """

    def __init__(
            self,
            chans: int,
            kernel_size: int = 7,
    ):
        super(ELA, self).__init__()

        self.chans = chans

        self.conv = nn.Conv1d(chans, chans, kernel_size, padding=kernel_size // 2, groups=chans, bias=False)
        self.gn = nn.GroupNorm(num_groups=16, num_channels=chans)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size()
        # (b, c, h)
        x_h = x.mean(3, keepdim=True).view(b, c, h)
        # (b, c, w)
        x_w = x.mean(2, keepdim=True).view(b, c, w)
        # compute attention
        a_h = self.sigmoid(self.gn(self.conv(x_h))).view(b, c, h, 1)
        a_w = self.sigmoid(self.gn(self.conv(x_w))).view(b, c, 1, w)

        return x * a_w * a_h
