import torch
import torch.nn as nn
from mmengine.model import BaseModule

__all__ = ['SA']


class SA(BaseModule):
    """
    Shuffle Attention
    """

    def __init__(self, in_chans: int, group_num: int = 64):
        super(SA, self).__init__()
        self.in_chans = in_chans
        self.group_num = group_num

        # channel weight and bias
        self.c_w = nn.Parameter(torch.zeros((1, in_chans // (2 * group_num), 1, 1)), requires_grad=True)
        self.c_b = nn.Parameter(torch.ones((1, in_chans // (2 * group_num), 1, 1)), requires_grad=True)

        # spatial weight and bias
        self.s_w = nn.Parameter(torch.zeros((1, in_chans // (2 * group_num), 1, 1)), requires_grad=True)
        self.s_b = nn.Parameter(torch.ones((1, in_chans // (2 * group_num), 1, 1)), requires_grad=True)

        self.gn = nn.GroupNorm(in_chans // (2 * group_num), in_chans // (2 * group_num))
        self.gate = nn.Sigmoid()

    @staticmethod
    def channel_shuffle(x: torch.Tensor, groups: int):
        b, c, h, w = x.shape

        x = x.reshape(b, groups, -1, h, w)
        x = x.permute(0, 2, 1, 3, 4)

        # flatten
        x = x.reshape(b, -1, h, w)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        # (B, C, H, W) -> (B * G, C // G, H, W)
        x = x.reshape(b * self.group_num, -1, h, w)
        x_0, x_1 = x.chunk(2, dim=1)
        # (B * G, C // 2G, H, W) -> (B * G, C // 2G, 1, 1)
        xc = x_0.mean(dim=(2, 3), keepdim=True)
        xc = self.c_w * xc + self.c_b
        xc = x_0 * self.gate(xc)

        # (B * G, C // 2G, H, W) -> (B * G, C // 2G, 1, 1)
        xs = self.gn(x_1)
        xs = self.s_w * xs + self.s_b
        xs = x_1 * self.gate(xs)

        out = torch.cat((xc, xs), dim=1)
        out = out.reshape(b, -1, h, w)

        out = self.channel_shuffle(out, 2)
        return out

