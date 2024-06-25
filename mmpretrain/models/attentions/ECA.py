from cmath import log
import typing as t
import torch
import torch.nn as nn
from mmengine.model import BaseModule

__all__ = ['ECA']


class ECA(BaseModule):
    """
    Efficient Channel Attention:
    1.Efficient and lightweight channel attention mechanism with low model complexity
    2.The Core innovation points are as follows:
        - dimensionality reduction may have a side effect on channel interaction, so discard it.
        - use GWConv, which can be regarded as a depth-wise separable convolution, and generateds channel weights
        by performing a fast 1D convolution of size K, where k is adaptively determined  via a non-linearity mapping
        of channel dimension C.

    note: transpose channel dimension and spatial dimension to use fast 1D convolution with kernel size K. K is based
    on the channel dimension.
    """

    def __init__(
            self,
            in_chans: int,
            kernel_size: t.Optional[int] = None,
            gamma: int = 2,
            b: int = 1,
            auto: bool = False
    ):
        super(ECA, self).__init__()
        if auto:
            t = int(abs((log(in_chans, 2) + b) / gamma))
            kernel_size = kernel_size or t if t % 2 else t + 1
        if kernel_size is None and in_chans < 96:
            kernel_size = 1
        elif kernel_size is None:
            kernel_size = 3
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x dim is (B, C, H, W)
        """
        # (B, C, 1, 1)
        y = x.mean((2, 3), keepdim=True)
        # (B, 1, C)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        # (B, C, 1, 1)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y
