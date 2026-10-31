# EfficientViT-B0 CRA dense (2026-10-31).
# More curves + more points per curve to improve recall / coverage.
_base_ = ['../crack_batch/efficientvit_b0.py']

model = dict(
    backbone=dict(
        block_type='cra',
        attn_kwargs=dict(
            num_curves=36,
            points_per_curve=24,
            num_heads=4,
            curve_length=0.7,
            offset_scale=0.10)))
