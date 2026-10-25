# EfficientViT-B0 with ALL LiteMLA modules replaced by CDGL + AS-GLA.
# Global branch uses AS-GLA; local branch uses strip convs.
_base_ = ['../crack_batch/efficientvit_b0.py']

model = dict(
    backbone=dict(
        block_type='cdgl_asgla',
        attn_kwargs=dict(
            local_ratio=0.25,
            strip_kernel=7)))
