# EfficientViT-B0 + hybrid attention, crack segmentation.
# Stage 3: LiteMLA + residual global attention + edge gate.
# Stage 2: LiteMLA + coordinate attention.
_base_ = ['../crack_batch/efficientvit_b0.py']

model = dict(
    backbone=dict(
        block_type='hybrid',
        attn_kwargs=dict(
            global_stage=3,
            coord_stage=2,
            use_edge=True,
            attn_dim=64,
            num_heads=4,
            edge_kernel=7)))
