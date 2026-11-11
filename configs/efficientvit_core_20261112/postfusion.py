_base_ = ['./_base_.py']
model = dict(
    backbone=dict(
        block_type='core_postfusion',
        attn_kwargs=dict(routing_width=8, num_banks=2)))
