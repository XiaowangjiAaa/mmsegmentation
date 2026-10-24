# EfficientViT-B0 with ALL LiteMLA context modules replaced by global attention.
# Both stage 2 (32x32) and stage 3 (16x16) use full multi-head attention.
_base_ = ['../crack_batch/efficientvit_b0.py']

model = dict(
    backbone=dict(
        block_type='replace_global',
        attn_kwargs=dict(
            attn_dim=64,
            num_heads=4)))
