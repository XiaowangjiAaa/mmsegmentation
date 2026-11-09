_base_ = ['./_base_.py']
load_from = 'work_dirs/pretrained/efficientvit_b0_adaptive_downsample.pth'
model = dict(
    backbone=dict(
        downsample_mode='adaptive_structure',
        adaptive_groups=4,
        adaptive_width=8))
