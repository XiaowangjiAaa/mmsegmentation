_base_ = [
    '../_base_/models/efficientvit-b0.py', './_base_/dataset.py',
    './_base_/schedule_40k.py', '../_base_/default_runtime.py'
]

# Converted from the official EfficientViT-B0 Cityscapes checkpoint by
# scripts/prepare_efficientvit_b0.py.  The 19-class classifier is skipped;
# backbone and SegHead fusion weights are loaded, conv_seg is random.
load_from = 'work_dirs/pretrained/efficientvit_b0_crack.pth'

model = dict(
    data_preprocessor=dict(size=(512, 512)),
    decode_head=dict(
        num_classes=2,
        out_channels=2,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                avg_non_ignore=True,
                loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=1.0)
        ]),
    test_cfg=dict(mode='whole'))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0004, betas=(0.9, 0.999), weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2))
