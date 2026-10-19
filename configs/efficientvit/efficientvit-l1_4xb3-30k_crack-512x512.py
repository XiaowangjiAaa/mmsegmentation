_base_ = [
    '../_base_/models/efficientvit-l1.py',
    '../_base_/datasets/crack.py', '../_base_/default_runtime.py',
    './_base_/schedule_30k_crack.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=2, out_channels=2))

train_dataloader = dict(batch_size=3, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = dict(batch_size=1, num_workers=4)
