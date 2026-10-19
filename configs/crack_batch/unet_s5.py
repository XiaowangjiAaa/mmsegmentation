_base_ = [
    '../_base_/models/fcn_unet_s5-d16.py', './_base_/dataset.py',
    './_base_/schedule_40k.py', '../_base_/default_runtime.py'
]

model = dict(
    data_preprocessor=dict(size=(512, 512)),
    decode_head=dict(
        num_classes=2,
        out_channels=2,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=0.5,
                avg_non_ignore=True),
            dict(
                type='DiceLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                ignore_index=255)
        ]),
    auxiliary_head=dict(
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
