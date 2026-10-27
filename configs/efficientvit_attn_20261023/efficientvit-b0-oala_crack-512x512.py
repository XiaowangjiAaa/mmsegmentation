# EfficientViT-B0 with ALL LiteMLA modules replaced by OA-LA.
# OA-LA uses a structure-tensor orientation field to weight horizontal,
# diagonal and vertical strip aggregators inside LiteMLA.
_base_ = ['../crack_batch/efficientvit_b0.py']

model = dict(
    backbone=dict(
        block_type='oala',
        attn_kwargs=dict(strip_kernel=7)))
