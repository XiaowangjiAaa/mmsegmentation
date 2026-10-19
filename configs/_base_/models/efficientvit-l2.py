# model settings
norm_cfg = dict(type='BN', eps=1e-7)
act_cfg = dict(type='GELU', approximate='tanh')
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='EfficientViT',
        arch='l2',
        in_channels=3,
        out_indices=(0, 1, 2, 3),
        norm_cfg=norm_cfg,
        act_cfg=act_cfg,
        init_cfg=None),
    decode_head=dict(
        type='EfficientViTHead',
        in_channels=[512, 256, 128],
        in_index=(3, 2, 1),
        strides=(32, 16, 8),
        head_width=256,
        head_depth=5,
        expand_ratio=1,
        middle_op='fmbconv',
        final_expand=None,
        act_cfg=act_cfg,
        norm_cfg=norm_cfg,
        num_classes=19,
        dropout_ratio=0.0,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                avg_non_ignore=True,
                loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=1.0)
        ]),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
