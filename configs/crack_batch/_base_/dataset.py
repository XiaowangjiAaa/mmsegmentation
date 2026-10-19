# Shared crack dataset for batch experiments.
# Model base configs do not define dataloaders, so these values are merged
# directly without any delete markers.
dataset_type = 'CrackDataset'
data_root = 'dataset'
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', imdecode_flag='grayscale'),
    dict(
        type='RandomResize',
        scale=(2048, 512),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 512), keep_ratio=True),
    dict(type='LoadAnnotations', imdecode_flag='grayscale'),
    # UNet-S5 needs both sides divisible by 16.  Non-square images such as
    # 512x910 are padded to 512x912; the seg padding value 255 is ignored by
    # the metric.
    dict(
        type='Pad',
        size_divisor=16,
        pad_val=dict(img=0, seg=255)),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=12,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='Image', seg_map_path='Mask'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        split='train',
        seed=42,
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='Image', seg_map_path='Mask'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        split='val',
        seed=42,
        pipeline=test_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='Image', seg_map_path='Mask'),
        img_suffix='.jpg',
        seg_map_suffix='.png',
        split='test',
        seed=42,
        pipeline=test_pipeline,
        test_mode=True))

val_evaluator = dict(
    type='CrackIoUMetric',
    iou_metrics=['mIoU', 'mDice', 'mFscore'])
test_evaluator = val_evaluator
