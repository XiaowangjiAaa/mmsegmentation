# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.structures import PixelData

from mmseg.models.decode_heads import EfficientViTHead
from mmseg.structures import SegDataSample


def _inputs():
    return [
        torch.randn(1, 16, 16, 16),
        torch.randn(1, 32, 8, 8),
        torch.randn(1, 64, 4, 4),
        torch.randn(1, 128, 2, 2),
    ]


def test_efficientvit_head():
    head = EfficientViTHead(
        in_channels=[128, 64, 32],
        in_index=(3, 2, 1),
        strides=(32, 16, 8),
        head_width=32,
        head_depth=1,
        expand_ratio=4,
        middle_op='mbconv',
        final_expand=4,
        num_classes=19,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    outputs = head(_inputs())
    assert outputs.shape == (1, 19, 8, 8)

    # FusedMBConv middle blocks and no final expansion (ADE20K style).
    head = EfficientViTHead(
        in_channels=[128, 64, 32],
        in_index=(3, 2, 1),
        strides=(32, 16, 8),
        head_width=32,
        head_depth=2,
        expand_ratio=4,
        middle_op='fmbconv',
        final_expand=None,
        num_classes=2,
        out_channels=2,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
    outputs = head(_inputs())
    assert outputs.shape == (1, 2, 8, 8)

    # Loss computation with the default CE + Dice setting.
    head = EfficientViTHead(
        in_channels=[128, 64, 32],
        in_index=(3, 2, 1),
        strides=(32, 16, 8),
        head_width=32,
        head_depth=1,
        expand_ratio=4,
        middle_op='mbconv',
        final_expand=4,
        num_classes=19,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            dict(type='DiceLoss', use_sigmoid=False, loss_weight=1.0)
        ])
    batch_data_samples = []
    for i in range(2):
        sample = SegDataSample()
        sample.gt_sem_seg = PixelData(
            data=torch.randint(0, 19, (1, 8, 8)))
        batch_data_samples.append(sample)
    losses = head.loss(
        [
            torch.randn(2, 16, 16, 16),
            torch.randn(2, 32, 8, 8),
            torch.randn(2, 64, 4, 4),
            torch.randn(2, 128, 2, 2)
        ], batch_data_samples, None)
    assert 'loss_ce' in losses and 'loss_dice' in losses
