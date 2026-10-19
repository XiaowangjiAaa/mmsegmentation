# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmseg.models.backbones import EfficientViT


def test_efficientvit_backbone():
    # EfficientViT-B0: four outputs at strides 4, 8, 16 and 32.
    model = EfficientViT(arch='b0')
    model.train()
    imgs = torch.randn(1, 3, 64, 64)
    feats = model(imgs)
    assert len(feats) == 4
    assert [feat.shape[1] for feat in feats] == [16, 32, 64, 128]
    assert [feat.shape[-1] for feat in feats] == [16, 8, 4, 2]

    # out_indices selects a subset of the stages.
    model = EfficientViT(arch='b1', out_indices=(1, 3))
    feats = model(torch.randn(1, 3, 64, 64))
    assert len(feats) == 2
    assert [feat.shape[1] for feat in feats] == [64, 256]

    # EfficientViT-L1 uses GELU and returns wider features.
    model = EfficientViT(arch='l1')
    feats = model(torch.randn(1, 3, 64, 64))
    assert len(feats) == 4
    assert [feat.shape[1] for feat in feats] == [64, 128, 256, 512]

    # Input sizes that are not multiples of 32 still work.
    model = EfficientViT(arch='b0')
    feats = model(torch.randn(2, 3, 63, 65))
    assert len(feats) == 4

    # Invalid arch / out_indices.
    with pytest.raises(AssertionError):
        EfficientViT(arch='b9')
    with pytest.raises(AssertionError):
        EfficientViT(arch='b0', out_indices=(0, 4))
