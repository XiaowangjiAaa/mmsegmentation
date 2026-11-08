_base_ = ['./_base_.py']
load_from = 'work_dirs/pretrained/efficientvit_b0_semantic_transition.pth'
model = dict(
    backbone=dict(
        transition_mode='semantic_structure',
        transition_gate='semantic',
        transition_width=8))
