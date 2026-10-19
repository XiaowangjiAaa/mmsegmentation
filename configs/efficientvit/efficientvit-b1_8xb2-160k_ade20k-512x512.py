_base_ = [
    '../_base_/models/efficientvit-b1.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', './_base_/schedule_160k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=150, final_expand=None))

train_dataloader = dict(batch_size=2, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader
