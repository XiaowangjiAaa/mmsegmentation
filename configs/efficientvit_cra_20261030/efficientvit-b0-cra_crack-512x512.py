# EfficientViT-B0 with ALL LiteMLA modules replaced by Curve-Routed Attention.
# A 4x4 seed grid predicts deformable curves; features are sampled along the
# curves, mixed with curve attention, and splatted back.
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
