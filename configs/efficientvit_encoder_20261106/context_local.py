_base_ = ['./_base_.py']
model = dict(
    backbone=dict(
        block_type='context_local', attn_kwargs=dict(structure_width=8)))
