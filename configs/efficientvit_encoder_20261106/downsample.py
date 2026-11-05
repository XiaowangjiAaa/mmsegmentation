_base_ = ['./_base_.py']
model = dict(
    backbone=dict(
        downsample_mode='structure', structure_downsample_stages=(1, 2)))
