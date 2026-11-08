_base_ = ['./semantic.py']
model = dict(
    decode_head=dict(
        type='EfficientViTStructureHead',
        structure_mode='subpixel',
        structure_width=16,
        structure_aux_weight=0.0))
