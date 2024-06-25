import os
import random
import typing as t

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from mmengine import Config

from mmpretrain import get_model

feature_map: t.Optional[torch.Tensor] = None

IMAGENET_1K = r'C:\dataset\ImageNet1k-2012\val'


def get_pic_from_each_category() -> t.List:
    cls_path = [os.path.join(IMAGENET_1K, name) for name in os.listdir(IMAGENET_1K)]
    res = []
    for path in cls_path:
        temp = random.choice(os.listdir(path))
        res.append(os.path.join(path, temp))
    return res


# 定义钩子函数, 用于获取特征图
def hook_fn(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
    global feature_map
    feature_map = output


class VanillaConv1D(nn.Module):

    def __init__(self, in_chans: int, kernel_size: int):
        super(VanillaConv1D, self).__init__()
        self.in_chans = in_chans
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv1d(in_chans, in_chans, kernel_size, padding=kernel_size // 2, bias=True)
        self.conv2 = nn.Conv1d(in_chans, in_chans, kernel_size, padding=kernel_size // 2, bias=True)

        self.norm_h = nn.GroupNorm(1, in_chans)
        self.norm_w = nn.GroupNorm(1, in_chans)

        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        # (B, C, H)
        x_h = x.mean(dim=3)
        # (B, C, W)
        x_w = x.mean(dim=2)

        x_h = self.gate(self.norm_h(self.conv1(x_h)))
        x_w = self.gate(self.norm_w(self.conv2(x_w)))

        x_h_attn = x_h.view(b, c, h, 1)
        x_w_attn = x_w.view(b, c, 1, w)
        return x * x_h_attn * x_w_attn


class SharedConv1D(nn.Module):

    def __init__(self, in_chans: int, kernel_size: int):
        super(SharedConv1D, self).__init__()
        self.in_chans = in_chans
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_chans, in_chans, kernel_size, stride=1, padding=kernel_size // 2, bias=True)

        self.norm_h = nn.GroupNorm(1, in_chans)
        self.norm_w = nn.GroupNorm(1, in_chans)

        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.size()
        # (B, C, H)
        x_h = x.mean(dim=3)
        # (B, C, W)
        x_w = x.mean(dim=2)

        x_h = self.gate(self.norm_h(self.conv(x_h)))
        x_w = self.gate(self.norm_w(self.conv(x_w)))

        x_h_attn = x_h.view(b, c, h, 1)
        x_w_attn = x_w.view(b, c, 1, w)
        return x * x_h_attn * x_w_attn


def transformer():
    compose = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return compose


def get_field_heatmap(
        model: nn.Module,
        image_paths: t.List,
        k: int,
        layer: t.Optional[nn.Module] = None) -> t.Any:
    heatmap = torch.zeros([224, 224])
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    for path in random.choices(image_paths, k=k):
        img = Image.open(path)
        if len(img.mode) == 1:
            continue
        img = transformer()(img)
        img = img.unsqueeze(0).to('cuda')
        hook = layer.register_forward_hook(hook_fn)
        img.requires_grad = True
        # img.retain_grad()
        model(img)
        hook.remove()

        # use softmax to compute feature
        weights = feature_map.mean(dim=[2, 3])
        weights = torch.softmax(weights, dim=0)
        temp_fea = (feature_map * weights[:, :, None, None]).sum(dim=1).squeeze()
        temp_fea[temp_fea.shape[0] // 2 - 1][temp_fea.shape[1] // 2 - 1].backward()
        grad = torch.abs(img.grad)
        # (H, W)
        grad = grad.mean(dim=1, keepdim=False).squeeze()
        heatmap = heatmap + grad.cpu().numpy()
        img.grad = None

    # use normalization
    mean = heatmap.mean()
    std = heatmap.std()
    heatmap = (heatmap - mean) / std
    heatmap = np.clip((heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()), 0, 1)  # 再次归一化到0-1

    cam = cv2.applyColorMap(np.uint8(heatmap * 255), cv2.COLORMAP_PINK)
    cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cam)


def get_heatmap(config: str, checkpoint: str, name: str, k: int) -> t.Any:
    image_paths: t.List = get_pic_from_each_category()
    cfg = Config.fromfile(config)
    model = get_model(cfg, checkpoint)
    if 'baseline' in name:
        layer = model.backbone.layer4[2].bn2
    else:
        layer = model.backbone.layer4[2].attn
    img = get_field_heatmap(model, image_paths, k, layer)
    img.save(rf'D:\projects\mmpretrain\tools\visualization\{name}.png')
    return img


if __name__ == '__main__':
    iters = 300
    layer = 4
    config2 = r'D:\projects\mmpretrain\work_dirs\resnet\spmca\resnet50-spmca7\resnet50_8xb32_in1k_spmca7.py'
    checkpoint2 = r'D:\projects\mmpretrain\work_dirs\resnet\spmca\resnet50-spmca7\20240408_124835\best_accuracy_top1_epoch_94.pth'

    config1 = r'D:\projects\mmpretrain\work_dirs\resnet\baseline\resnet50_8xb32_in1k.py'
    checkpoint1 = r'D:\projects\mmpretrain\work_dirs\resnet\baseline\resnet50\best_accuracy_top1_epoch_96.pth'

    img1 = get_heatmap(config1, checkpoint1, f'baseline-layer{layer}-{iters}', iters)
    img2 = get_heatmap(config2, checkpoint2, f'scsa-layer{layer}-{iters}', iters)


    # model1 = nn.Sequential(*[VanillaConv1D(128, 9) for i in range(10)])
    # model2 = nn.Sequential(*[SharedConv1D(128, 9) for i in range(10)])

    fig = plt.figure(figsize=(12, 12))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(img1)
    # plt.axis('off')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.imshow(img2)
    # plt.axis('off')
    plt.show()
