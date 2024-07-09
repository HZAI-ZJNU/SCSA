# SCSA
This repo is the official of implementation of "[SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention](https://arxiv.org/pdf/2407.05128v1)".

## Introduction

In this paper, starting from the synergy of multi-semantic information, we propose a plug-and-play Spatial and Channel Synergistic Attention module(SCSA).

We conduct extensive experiments on seven benchmark datasets, including
classification on ImageNet-1K, object detection on MSCOCO
2017, segmentation on ADE20K, and four other complex scene
detection datasets to validate the effectiveness of our method.

## Running

### Install

We implement SCSA using `MMPretrain V1.2.0`, `MMDetection V3.3.0`, `MMSegmentation V1.2.2` and `MMCV V2.1.0`.  
We train and test our models under `python=3.10`, `pytorch=2.1.1`, `cuda=11.8`.

```shell
# Create a virtual environment and activate it.
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install -e .
```
### Data preparation

The ImageNet dataset should be prepared as follows:

```
$ tree data
imagenet
├── train
│   ├── class1
│   │   ├── img1.jpeg
│   │   ├── img2.jpeg
│   │   └── ...
│   ├── class2
│   │   ├── img3.jpeg
│   │   └── ...
│   └── ...
└── val
    ├── class1
    │   ├── img4.jpeg
    │   ├── img5.jpeg
    │   └── ...
    ├── class2
    │   ├── img6.jpeg
    │   └── ...
    └── ...
```

### Training
Our work employs a single GPU for training on classification tasks.    
Here is an example: train SCSA + ResNet-50 with an single GPU:
```shell
python tools\train.py work_dirs\resnet50_1xb128_in1k_scsa.py --work-dir path_to_exp --amp 
```

### Testing
Test SCSA + ResNet-50 with an single GPU:
```shell
python tools\test.py work_dirs\resnet50_1xb128_in1k_scsa.py path_to_checkpoint --work-dir path_to_exp
```

## Results
We will open source the relevant model weights later.

## Acknowledgement
The code in this repository is developed based on the [MMPretrain](https://github.com/open-mmlab/mmpretrain). Furthermore, the detection and segmentation tasks involved in this work are implemented based on the [MMDetection](https://github.com/open-mmlab/mmdetection) and [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).
## Cite SCSA
If you find this repository useful, please use the following BibTeX entry for citation.
```latex
@InProceedings{si2024SCSA,
  title={SCSA: Exploring the Synergistic Effects Between Spatial and Channel Attention},
  author={Si, Yunzhong and Xu, Huiying and Zhu, Xinzhong and Zhang, Wenhao and Dong, Yao and Chen, Yuxing and Li, Hongbo},
  journal={arXiv preprint arXiv:2407.05128v1},
  year={2024}
}
```

## Concat
If you have any questions, please feel free to contact the authors.

Yunzhong Si: 
[siyunzhong@zjnu.edu.cn](mailto:iyunzhong@zjnu.edu.cn)
