# EfficientViT-B0 + local window attention, crack segmentation.
# Stage 2/3: LiteMLA + residual 8x8 window attention.
# Stage 2 also gets coordinate attention; stage 3 gets an edge gate.
_base_ = ['../crack_batch/efficientvit_b0.py']

model = dict(
    backbone=dict(
        block_type='window',
        attn_kwargs=dict(
            window_size=8,
            coord_stage=2,
            use_edge=True,
            attn_dim=64,
            num_heads=4,
            edge_kernel=7)))
