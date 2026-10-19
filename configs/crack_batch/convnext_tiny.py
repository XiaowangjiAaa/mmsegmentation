_base_ = [
    '../_base_/models/upernet_convnext.py',
    './_base_/dataset.py', './_base_/schedule_40k.py',
    '../_base_/default_runtime.py'
]

checkpoint_file = ('https://download.openmmlab.com/mmclassification/v0/'
                   'convnext/downstream/'
                   'convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-'
                   '795e9634.pth')

model = dict(
    data_preprocessor=dict(size=(512, 512)),
    backbone=dict(
        arch='tiny',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=checkpoint_file,
            prefix='backbone.'),
        drop_path_rate=0.1),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=2,
        out_channels=2,
        loss_decode=dict(avg_non_ignore=True)),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=2,
        out_channels=2,
        loss_decode=dict(avg_non_ignore=True)),
    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        decay_rate=0.9, decay_type='stage_wise', num_layers=6),
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')
