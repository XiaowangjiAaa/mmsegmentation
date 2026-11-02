_base_ = ['./_base_.py']
model = dict(backbone=dict(
    block_type='cra',
    attn_kwargs=dict(num_curves=16, points_per_curve=16,
                     num_heads=4, curve_length=0.8, offset_scale=0.15)))
# CRA has no matching pretrained attention: use a scratch baseline control.
load_from = None
