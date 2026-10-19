_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    './_base_/dataset.py', './_base_/schedule_40k.py',
    '../_base_/default_runtime.py'
]

model = dict(
    pretrained=None,
    data_preprocessor=dict(size=(512, 512)),
    backbone=dict(
        depth=18,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='open-mmlab://resnet18_v1c')),
    decode_head=dict(
        in_channels=512,
        c1_in_channels=64,
        c1_channels=12,
        channels=128,
        num_classes=2,
        out_channels=2,
        loss_decode=dict(avg_non_ignore=True)),
    auxiliary_head=dict(
        in_channels=256,
        channels=64,
        num_classes=2,
        out_channels=2,
        loss_decode=dict(avg_non_ignore=True)),
    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005))

param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]
