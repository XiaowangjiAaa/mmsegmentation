_base_ = [
    '../_base_/models/twins_pcpvt-s_upernet.py', './_base_/dataset.py',
    './_base_/schedule_40k.py', '../_base_/default_runtime.py'
]

checkpoint_file = ('https://download.openmmlab.com/mmsegmentation/v0.5/'
                   'pretrain/twins/pcpvt_small_20220308-e638c41c.pth')

model = dict(
    data_preprocessor=dict(size=(512, 512)),
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
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
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
