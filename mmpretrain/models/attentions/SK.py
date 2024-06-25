import torch
import torch.nn as nn
import typing as t
from mmcv.cnn import ConvModule
from mmengine import MODELS
from mmengine.model import BaseModule

from mmpretrain.models.attentions.utils import auto_pad

__all__ = ['SK']


@MODELS.register_module()
class SK(BaseModule):
    """
    SK Module combines the Inception and SE ideas, considering different channels and kernel block.
    It can split into three parts:

        1.Split: For any feature map, using different size kernel convolutions(3x3, 5x5)
        to extract new feature map. use dilation convolution (3x3, dilation=2) can
        increase regularization to avoid over-fitting caused by large convolutional kernels
        , add multi-scale information and increase receptive field.

        2.Fuse: Fusing different output(last layer feature map) of different branch and
        compute attention on channel-wise

        3.Select: 'chunk, scale, fuse'.  Focusing on different convolutional kernels for different target sizes.
    """

    def __init__(
            self,
            in_chans: int,
            out_chans: int,
            num: int = 2,
            kernel_size: int = 3,
            stride: int = 1,
            groups: int = 1,
            reduction: int = 16,
            norm_cfg: t.Dict = dict(type='BN'),
            act_cfg: t.Dict = dict(type='ReLU'),
    ):
        """
        Args:
            num: the number of different kernel, by the way, it means the number of different branch, using
                Inception ideas.
            reduction: Multiple of dimensionality reduction, used to reduce params quantities and improve nonlinear
                ability.
        """
        super(SK, self).__init__()
        self.num = num
        self.out_chans = out_chans
        self.kernel_size = kernel_size
        self.conv = nn.ModuleList()
        for i in range(num):
            self.conv.append(ConvModule(in_chans, out_chans, kernel_size, stride=stride, groups=groups, dilation=1 + i,
                                        padding=auto_pad(k=kernel_size, d=1 + i), norm_cfg=norm_cfg, act_cfg=act_cfg))

        # fc can be implemented by 1x1 conv
        self.fc = nn.Sequential(
            # use relu act to improve nonlinear expression ability
            ConvModule(in_chans, out_chans // reduction, kernel_size=1, norm_cfg=norm_cfg, act_cfg=act_cfg),
            nn.Conv2d(out_chans // reduction, out_chans * self.num, kernel_size=1, bias=False)
        )
        # compute channels weight
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # use different convolutional kernel to conv
        temp_feature = [conv(x) for conv in self.conv]
        x = torch.stack(temp_feature, dim=1)
        # fuse different output and squeeze
        attn = x.sum(1).mean((2, 3), keepdim=True)
        # excitation
        attn = self.fc(attn)
        batch, c, h, w = attn.size()
        attn = attn.view(batch, self.num, self.out_chans, h, w)
        attn = self.softmax(attn)
        # select
        x = x * attn
        x = torch.sum(x, dim=1)
        return x
