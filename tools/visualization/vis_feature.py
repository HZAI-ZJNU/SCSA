import cv2
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from mmengine import Config
from mmpretrain.apis import get_model

feature_map = None


# 定义钩子函数, 用于获取特征图
def hook_fn(module, input, output):
    global feature_map
    feature_map = output


if __name__ == '__main__':
    config = r'D:\projects\mmpretrain\work_dirs\resnet\spmca\resnet50-spmca7\resnet50_8xb32_in1k_spmca7.py'
    checkpoint = r'D:\projects\mmpretrain\work_dirs\resnet\spmca\resnet50-spmca7\20240408_124835\best_accuracy_top1_epoch_94.pth'
    cfg = Config.fromfile(config)
    model = get_model(cfg, checkpoint)

    aim_layer = model.backbone.layer3[2].bn2
    print(aim_layer)

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # image = Image.open(r'C:\dataset\ImageNet1k-2012\train\n01484850\n01484850_146.JPEG')
    # image = transform(image).unsqueeze(0)
    image = cv2.imread(r'C:\dataset\ImageNet1k-2012\train\n01484850\n01484850_146.JPEG')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = transform(image)
    image = image.unsqueeze(0)

    hook = aim_layer.register_forward_hook(hook_fn)
    model = model.eval()
    model(image)
    hook.remove()

    num_features = feature_map.shape[1]
    print(feature_map.size())
    plt.figure(figsize=(12, 12))
    feature_map = torch.mean(feature_map, dim=1, keepdim=True)
    plt.imshow(feature_map[0, 0].detach().cpu().numpy(), cmap='viridis')
    plt.axis('off')
    # for i in range(8):
    #     plt.subplot(2, 4, i + 1)
    #     feature_map = torch.mean(feature_map, dim=1, keepdim=True)
    #     plt.imshow(feature_map[0, 0].detach().cpu().numpy(), cmap='viridis')
    #     plt.axis('off')
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.show()
