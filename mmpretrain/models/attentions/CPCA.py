from cmath import log
import typing as t
import torch
import torch.nn as nn
from mmengine.model import BaseModule

__all__ = ['CPCA']


class ChannelAttention(BaseModule):
    """
    Channel attention module based on CPCA
    use hidden_chans to reduce parameters instead of conventional convolution
    """

    def __init__(self, in_chans: int, hidden_chans: int):
        super().__init__()
        self.fc1 = nn.Conv2d(in_chans, hidden_chans, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(hidden_chans, in_chans, kernel_size=1, stride=1, bias=True)
        self.act = nn.ReLU(inplace=True)
        self.in_chans = in_chans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim is (B, C, H, W)
        """
        # (B, C, 1, 1)
        x1 = x.mean(dim=(2, 3), keepdim=True)
        x1 = self.fc2(self.act(self.fc1(x1)))
        x1 = torch.sigmoid(x1)

        # (B, C, 1, 1)
        x2 = x.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x2 = self.fc2(self.act(self.fc1(x2)))
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.in_chans, 1, 1)
        return x


class CPCA(BaseModule):
    """
    Channel Attention and Spatial Attention based on CPCA
    """

    def __init__(self, in_chans: int, reduction: int = 16):
        super(CPCA, self).__init__()
        self.in_chans = in_chans

        hidden_chans = in_chans // reduction
        # Channel Attention
        self.ca = ChannelAttention(in_chans, hidden_chans)

        # Spatial Attention
        self.dwc5_5 = nn.Conv2d(in_chans, in_chans, kernel_size=5, padding=2, groups=in_chans)
        self.dwc1_7 = nn.Conv2d(in_chans, in_chans, kernel_size=(1, 7), padding=(0, 3), groups=in_chans)
        self.dwc7_1 = nn.Conv2d(in_chans, in_chans, kernel_size=(7, 1), padding=(3, 0), groups=in_chans)
        self.dwc1_11 = nn.Conv2d(in_chans, in_chans, kernel_size=(1, 11), padding=(0, 5), groups=in_chans)
        self.dwc11_1 = nn.Conv2d(in_chans, in_chans, kernel_size=(11, 1), padding=(5, 0), groups=in_chans)
        self.dwc1_21 = nn.Conv2d(in_chans, in_chans, kernel_size=(1, 21), padding=(0, 10), groups=in_chans)
        self.dwc21_1 = nn.Conv2d(in_chans, in_chans, kernel_size=(21, 1), padding=(10, 0), groups=in_chans)

        # used to model feature connections between different receptive fields
        self.conv = nn.Conv2d(in_chans, in_chans, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.act(x)
        channel_attn = self.ca(x)
        x = channel_attn * x

        x_init = self.dwc5_5(x)
        x1 = self.dwc1_7(x_init)
        x1 = self.dwc7_1(x1)

        x2 = self.dwc1_11(x_init)
        x2 = self.dwc11_1(x2)

        x3 = self.dwc1_21(x_init)
        x3 = self.dwc21_1(x3)

        spatial_atn = x1 + x2 + x3 + x_init
        spatial_atn = self.conv(spatial_atn)
        y = x * spatial_atn
        y = self.conv(y)
        return y
