# EfficientViT-B0 with ALL LiteMLA modules replaced by CDGL.
# Channels are split into a global LiteMLA branch and a local strip-conv branch.
_base_ = ['../crack_batch/efficientvit_b0.py']

model = dict(
    backbone=dict(
        block_type='cdgl',
        attn_kwargs=dict(
            local_ratio=0.25,
            strip_kernel=7)))
