# EfficientViT-B0 with ALL LiteMLA context modules replaced by hybrid attention.
# Stage 2: window attention; stage 3: global attention. No LiteMLA remains.
_base_ = ['../crack_batch/efficientvit_b0.py']

model = dict(
    backbone=dict(
        block_type='replace_hybrid',
        attn_kwargs=dict(
            window_size=8,
            attn_dim=64,
            num_heads=4)))
