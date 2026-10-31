# EfficientViT-B0 CRA long (2026-10-31).
# Longer curves with more sample points for continuous cracks.
_base_ = ['../crack_batch/efficientvit_b0.py']

model = dict(
    backbone=dict(
        block_type='cra',
        attn_kwargs=dict(
            num_curves=16,
            points_per_curve=32,
            num_heads=4,
            curve_length=1.2,
            offset_scale=0.20)))
