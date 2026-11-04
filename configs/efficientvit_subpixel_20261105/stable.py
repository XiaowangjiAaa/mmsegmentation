_base_ = ['./subpixel.py']
model = dict(decode_head=dict(structure_mode='subpixel_stable'))
