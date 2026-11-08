_base_ = ['./semantic.py']
model = dict(backbone=dict(transition_gate='local'))
