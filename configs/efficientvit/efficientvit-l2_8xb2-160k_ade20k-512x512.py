_base_ = [
    '../_base_/models/efficientvit-l2.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', './_base_/schedule_160k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        num_classes=150,
        head_width=128,
        head_depth=3,
        expand_ratio=4,
        final_expand=8))

train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
