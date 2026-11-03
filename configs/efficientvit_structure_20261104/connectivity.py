_base_ = ['./_base_.py']
model = dict(
    decode_head=dict(
        type='EfficientViTStructureHead',
        structure_mode='connectivity',
        structure_width=16,
        structure_aux_weight=0.1))
