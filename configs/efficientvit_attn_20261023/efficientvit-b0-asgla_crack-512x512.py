# EfficientViT-B0 with ALL LiteMLA modules replaced by AS-GLA.
# AS-GLA keeps LiteMLA's qkv/proj and adds zero-gated horizontal/vertical
# strip aggregation for anisotropic cracks.
_base_ = ['../crack_batch/efficientvit_b0.py']

model = dict(
    backbone=dict(
        block_type='asgla',
        attn_kwargs=dict(strip_kernel=7)))
