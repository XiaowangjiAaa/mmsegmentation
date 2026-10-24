# EfficientViT-B0 with ALL LiteMLA context modules replaced by window attention.
# Stage 2 and stage 3 both get 8x8 window attention; no LiteMLA remains.
_base_ = ['../crack_batch/efficientvit_b0.py']

model = dict(
    backbone=dict(
        block_type='replace_window',
        attn_kwargs=dict(
            window_size=8,
            attn_dim=64,
            num_heads=4)))
