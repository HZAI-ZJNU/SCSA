"""Calculate the SSIM of two featurs"""
import inspect
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import argparse
import typing as t
from mmpretrain.apis import get_model
from mmengine import Config
from torchvision import transforms
from PIL import Image
from pytorch_msssim import ssim

feature_map = None


def hook_func(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
    global feature_map
    feature_map = output


class Args(object):

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Analysis of ssim')
        self.parser.add_argument('config', type=str, help='model config path')
        self.parser.add_argument('checkpoint', type=str, help='model checkpoint path')
        self.parser.add_argument('dataset', type=str, help='dataset path')
        self.parser.add_argument('--use-random', action='store_true', help='use random method to retrieve images')
        self.parser.add_argument('--num', type=int, default=-1,
                                 help='When the "use-random" parameter is set to true, '
                                      'this parameter must be passed to randomly obtain a specified number of samples.')
        self.parser.add_argument('--method', type=str, default='avg', help='pooling method')
        self.parser.add_argument('--input-size', type=int, default=224, help='resize images')
        self.parser.add_argument('--layer', type=str, default='model.backbone.layer[1]', help='layer name')
        self.parser.add_argument('--up-mode', type=str, default='bilinear', help='upsample method')
        self.args = self.parser.parse_args()


class SSIM(object):

    def __init__(
            self,
            opts: argparse.Namespace,
    ):
        self.args = opts
        self.input_sizes = (self.args.input_size, self.args.input_size)
        self.transform = transforms.Compose([
            transforms.Resize(self.input_sizes),
            transforms.ToTensor(),
        ])
        self.model, self.hook = self.get_model()

    def get_model(self) -> t.Tuple:
        config = Config.fromfile(self.args.config)
        model = get_model(config, self.args.checkpoint)
        print(model)
        layer = eval(self.args.layer)
        hook = layer.register_forward_hook(hook_func)
        model.eval()
        return model, hook

    def release_hook(self):
        self.hook.remove()

    def compute_ssim(self, path: str):
        method_name = self.args.method
        assert method_name in dir(SSIM) and inspect.isfunction(getattr(SSIM, method_name)), f'{method_name} not exist'
        img = Image.open(path).convert('RGB')
        o_img = img.copy()
        o_img = transforms.ToTensor()(o_img)
        img = self.transform(img).unsqueeze(0)
        self.model(img)
        d_img = feature_map
        d_img = nn.functional.interpolate(d_img, size=self.input_sizes, mode=self.args.up_mode, align_corners=False)
        res = ssim(o_img, d_img, data_range=max(o_img.max(), d_img.max()) - min(o_img.min(), d_img.min()))
        return res.detach().cpu().numpy()

    def main(self):
        paths = os.listdir(self.args.dataset)
        res = [self.compute_ssim(path) for path in paths]
        res = np.mean(res)
        print(f'Dataset: {self.args.dataset}  mean-SSIM: {res}')


if __name__ == '__main__':
    args = Args()
    ssim_ = SSIM(args.args)
    ssim_.main()
