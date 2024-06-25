from cmath import log
import typing as t
import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule

__all__ = ['CA']


class CA(BaseModule):
    """
    Coordinate Attention Block, which embeds positional information into channel attention.
    1.It considers spatial dimension attention and channel dimension attention, it helps model locate, identify and
    enhance more interesting objects.

    2.CA utilizes two 2-D GAP operation to respectively aggregate the input features along the vertical and horizontal
    directions into two separate direction aware feature maps. Then, encode these two feature maps separately into
    an attention tensor.

    3. Among these two feature maps(Cx1xW and CxHx1), one uses GAP to model the long-distance dependencies of
    the feature maps on a spatial dimension, while retaining position information int the other spatial dimension.
    In that case, the two feature maps, spatial information, and long-range dependencies complement each other.

    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            reduction: int = 32,
            norm_cfg: t.Dict = dict(type='BN'),
            act_cfg:  t.Dict = dict(type='HSwish'),
    ):
        super(CA, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans

        hidden_chans = max(8, in_chans // reduction)
        self.conv = ConvModule(in_chans, hidden_chans, kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg)

        self.attn_h = nn.Conv2d(hidden_chans, out_chans, 1)
        self.attn_w = nn.Conv2d(hidden_chans, out_chans, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor):
        b, c, h, w = x.size()
        # (b, c, h, 1)
        x_h = x.mean(3, keepdim=True)
        # (b, c, 1, w) -> (b, c, w, 1)
        x_w = x.mean(2, keepdim=True).permute(0, 1, 3, 2)
        # (b, c, h + w, 1)
        y = torch.cat((x_h, x_w), dim=2)
        y = self.conv(y)

        # split
        # x_h: (b, c, h, 1),  x_w: (b, c, w, 1)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        # (b, c, 1, w)
        x_w = x_w.permute(0, 1, 3, 2)

        # compute attention
        a_h = self.sigmoid(self.attn_h(x_h))
        a_w = self.sigmoid(self.attn_w(x_w))

        return x * a_w * a_h
