# EfficientViT-B0 CRA v1 (2026-10-31 baseline variant).
# 4x4 seeds, 16 points/curve, bilinear splat.
_base_ = ['../crack_batch/efficientvit_b0.py']

model = dict(
    backbone=dict(
        block_type='cra',
        attn_kwargs=dict(
            num_curves=16,
            points_per_curve=16,
            num_heads=4,
            curve_length=0.8,
            offset_scale=0.15)))
