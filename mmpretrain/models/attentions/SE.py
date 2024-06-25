import torch
import torch.nn as nn
from mmengine import MODELS
from mmengine.model import BaseModule

__all__ = ['SE', 'SEConv']


@MODELS.register_module()
class SE(BaseModule):
    """
    SE Block: A Channel Based Attention Mechanism.

        Traditional convolution in computation, it blends the feature relationships of the channel
    with the spatial relationships learned from the convolutional kernel, because a conv sum the
    operation results of each channel, so, using SE Block to pay attention to more important channels,
    suppress useless channels regard to current task.

    SE Block Contains three parts:
    1.Squeeze: Global Information Embedding.
        Aggregate (H, W, C) dim to (1, 1, C) dim, use GAP to generate aggregation channel,
    encode the entire spatial feature on a channel to a global feature.

    2.Excitation: Adaptive Recalibration.
        It aims to fully capture channel-wise dependencies and improve the representation of image,
    by using two liner layer, one activation inside and sigmoid or softmax to normalize,
    to produce channel-wise weights.
        Maybe like using liner layer to extract feature map to classify, but this applies at channel
    level and pay attention to channels with a large number of information.

    3.Scale: feature recalibration.
        Multiply the learned weight with the original features to obtain new features.
        SE Block can be added to Residual Block.

    """

    def __init__(self, channel: int, reduction: int = 16):
        super(SE, self).__init__()
        # part 1:(H, W, C) -> (1, 1, C)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # part 2, compute weight of each channel
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),  # nn.Softmax is OK here
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


@MODELS.register_module()
class SEConv(BaseModule):

    def __init__(self, channel: int, reduction: int = 16):
        super(SEConv, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.BatchNorm2d(channel // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.size()
        y = x.mean((2, 3), keepdim=True)
        y = self.fc(y)
        return x * y
