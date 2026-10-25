# EfficientViT-B0 with ALL LiteMLA modules replaced by a foveal-peripheral mixer.
# LiteMLA runs on a 2x-downsampled peripheral view; a full-resolution strip
# branch adds the foveal detail with a zero-initialised scale.
_base_ = ['../crack_batch/efficientvit_b0.py']

model = dict(
    backbone=dict(
        block_type='foveal',
        attn_kwargs=dict(strip_kernel=7)))
