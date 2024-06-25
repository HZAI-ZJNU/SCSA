import torch.nn as nn
import torch

from mmengine.model import BaseModule

__all__ = ['CBAM']


class CBAM(BaseModule):
    """
    The combination of channel attention and spatial attention,
    using average pooling and maximum pooling to aggregate channels and spaces along different dimensions
    """

    def __init__(
            self,
            in_chans: int,
            reduction: int = 16,
            kernel_size: int = 7,
            min_channels: int = 8,
    ):
        super(CBAM, self).__init__()
        # channel-wise attention
        hidden_chans = max(in_chans // reduction, min_channels)
        self.mlp_chans = nn.Sequential(
            nn.Conv2d(in_chans, hidden_chans, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_chans, in_chans, kernel_size=1, bias=False),
        )
        # space-wise attention
        self.mlp_spaces = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=3, bias=False)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, C, 1, 1)
        avg_x_s = x.mean((2, 3), keepdim=True)
        max_x_s = x.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
        x = x * self.gate(self.mlp_chans(avg_x_s) + self.mlp_chans(max_x_s))

        # (B, 1, H, W)
        avg_x_c = x.mean(dim=1, keepdim=True)
        max_x_c = x.max(dim=1, keepdim=True)[0]
        x = x * self.gate(self.mlp_spaces(torch.cat((avg_x_c, max_x_c), dim=1)))
        return x
