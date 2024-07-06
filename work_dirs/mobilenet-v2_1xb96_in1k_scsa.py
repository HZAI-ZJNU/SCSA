auto_scale_lr = dict(base_batch_size=96)
data_preprocessor = dict(
    mean=[
        123.675,
        116.28,
        103.53,
    ],
    num_classes=1000,
    std=[
        58.395,
        57.12,
        57.375,
    ],
    to_rgb=True)
dataset_type = 'ImageNet'
default_hooks = dict(
    checkpoint=dict(interval=1, save_best='auto', type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(enable=False, type='VisualizationHook'))
default_scope = 'mmpretrain'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'none'
load_from = None
log_level = 'INFO'
model = dict(
    backbone=dict(
        scsa_cfg=dict(
            attn_drop_ratio=0.0,
            down_sample_mode='avg_pool',
            gate_layer='sigmoid',
            group_kernel_sizes=[
                3,
                5,
                7,
                9,
            ],
            head_num=1,
            window_size=7),
        type='MobileNetV2SCSA',
        widen_factor=1.0),
    head=dict(
        in_channels=1280,
        loss=dict(loss_weight=1.0, type='CrossEntropyLoss'),
        num_classes=1000,
        topk=(
            1,
            5,
        ),
        type='LinearClsHead'),
    neck=dict(type='GlobalAveragePooling'),
    type='ImageClassifier')
optim_wrapper = dict(
    loss_scale='dynamic',
    optimizer=dict(lr=0.045, momentum=0.9, type='SGD', weight_decay=4e-05),
    type='AmpOptimWrapper')
param_scheduler = dict(by_epoch=True, gamma=0.98, step_size=1, type='StepLR')
randomness = dict(deterministic=False, seed=0)
resume = False
test_cfg = dict()
test_dataloader = dict(
    batch_size=96,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='C:\\dataset\\ImageNet1k-2012',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='val',
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', edge='short', scale=256, type='ResizeEdge'),
    dict(crop_size=224, type='CenterCrop'),
    dict(type='PackInputs'),
]
train_cfg = dict(by_epoch=True, max_epochs=400, val_interval=1)
train_dataloader = dict(
    batch_size=96,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='C:\\dataset\\ImageNet1k-2012',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', scale=224, type='RandomResizedCrop'),
            dict(direction='horizontal', prob=0.5, type='RandomFlip'),
            dict(type='PackInputs'),
        ],
        split='train',
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(backend='pillow', scale=224, type='RandomResizedCrop'),
    dict(direction='horizontal', prob=0.5, type='RandomFlip'),
    dict(type='PackInputs'),
]
val_cfg = dict()
val_dataloader = dict(
    batch_size=96,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        data_root='C:\\dataset\\ImageNet1k-2012',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(backend='pillow', edge='short', scale=256, type='ResizeEdge'),
            dict(crop_size=224, type='CenterCrop'),
            dict(type='PackInputs'),
        ],
        split='val',
        type='ImageNet'),
    num_workers=5,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    topk=(
        1,
        5,
    ), type='Accuracy')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='UniversalVisualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'work_dirs/mobilenetv2/scsa/mv2-scsa2'
