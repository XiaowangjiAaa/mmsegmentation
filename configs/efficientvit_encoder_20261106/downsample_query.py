_base_ = ['./query.py']
model = dict(
    backbone=dict(
        downsample_mode='structure', structure_downsample_stages=(1, 2)))
