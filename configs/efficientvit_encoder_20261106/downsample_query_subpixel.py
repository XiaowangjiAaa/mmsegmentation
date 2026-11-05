_base_ = ['./subpixel.py']
model = dict(
    backbone=dict(
        downsample_mode='structure',
        structure_downsample_stages=(1, 2),
        block_type='structure_query',
        attn_kwargs=dict(structure_width=8)))
