# Dual Attention network
import typing as t
import torch
import torch.nn as nn
from mmengine.model import BaseModule

__all__ = ['DualAttention']


class PAM(BaseModule):
    """
    position attention module with self-attention mechanism
    """

    def __init__(self, in_chans: int):
        super(PAM, self).__init__()
        self.in_chans = in_chans
        self.q = nn.Conv2d(in_chans, in_chans // 8, kernel_size=1)
        self.k = nn.Conv2d(in_chans, in_chans // 8, kernel_size=1)
        self.v = nn.Conv2d(in_chans, in_chans, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size()
        # (B, HW, C)
        q = self.q(x).view(b, -1, h * w).permute(0, 2, 1)
        # (B, C, HW)
        k = self.k(x).view(b, -1, h * w)
        # (B, C, HW)
        v = self.v(x).view(b, -1, h * w)
        # (B, HW, HW)
        attn = self.softmax(torch.bmm(q, k))
        # (B, C, HW)
        out = torch.bmm(v, attn.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        out = self.gamma * out + x
        return out


class CAM(BaseModule):
    """
    channel attention module with self-attention mechanism
    """

    def __init__(self, in_chans: int):
        super(CAM, self).__init__()
        self.in_chans = in_chans
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, w, h = x.size()
        # (B, C, HW)
        q = x.view(b, c, -1)
        # (B, HW, C)
        k = x.view(b, c, -1).permute(0, 2, 1)
        # (B, C, HW)
        v = x.view(b, c, -1)
        # (B, C, C)
        energy = torch.bmm(q, k)
        energy_new = torch.max(energy, dim=-1, keepdim=True)[0].expand_as(energy) - energy
        attn = self.softmax(energy_new)
        # (B, C, HW)
        out = torch.bmm(attn, v)
        out = out.view(b, c, h, w)
        out = self.gamma * out + x
        return out


class DualAttention(BaseModule):

    def __init__(self, in_chans: int):
        super(DualAttention, self).__init__()
        self.in_chans = in_chans
        self.pam = PAM(in_chans)
        self.cam = CAM(in_chans)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pam = self.pam(x)
        cam = self.cam(x)
        return pam + cam


if __name__ == '__main__':
    model = DualAttention(in_chans=64)
    x = torch.rand((3, 64, 40, 40))
    print(model(x).size())
