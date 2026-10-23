# EfficientViT-B0 + coordinate attention, crack segmentation.
# The original EfficientViT-B0 config is kept as the baseline; this config
# only changes the context block used in stages 2/3.
_base_ = ['../crack_batch/efficientvit_b0.py']

model = dict(
    backbone=dict(
        block_type='coord',
        attn_kwargs=dict(coord_reduction=32)))
