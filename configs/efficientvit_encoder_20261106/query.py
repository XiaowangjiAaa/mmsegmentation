_base_ = ['./_base_.py']
model = dict(
    backbone=dict(
        block_type='structure_query', attn_kwargs=dict(structure_width=8)))
