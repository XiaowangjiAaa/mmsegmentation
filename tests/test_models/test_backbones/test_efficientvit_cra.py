# Copyright (c) OpenMMLab. All rights reserved.
import unittest
import torch

from mmseg.models.backbones.efficientvit.attn_plugins.blocks import (
    EfficientViTBlockCRA)
from mmseg.models.backbones.efficientvit.attn_plugins.cra import CRAModule


def _check_curve_seed_coverage(num_curves, points, length):
    module = CRAModule(
        8, 8, num_curves=num_curves, points_per_curve=points,
        curve_length=length)
    with torch.no_grad():
        for parameter in module.offset_mlp.parameters():
            parameter.zero_()
        x = torch.randn(1, 8, 16, 16)
        _, coords = module._sample_curves(x)
        expected = (module.seed_grid[:, None, :] +
                    module.base_offsets).clamp(-1, 1)
        torch.testing.assert_close(coords[0], expected)
        # Every corner must contain samples; curves must not all start at
        # the center even when the learned offsets are zero.
        for sx in (-1, 1):
            for sy in (-1, 1):
                assert ((coords[..., 0] * sx > 0.5) &
                        (coords[..., 1] * sy > 0.5)).any()


def _check_single_identity_shortcut_and_backward():
    block = EfficientViTBlockCRA(in_channels=8, dim=8).eval()
    x = torch.randn(2, 8, 8, 8, requires_grad=True)
    # Test the integrated context wrapper, not only the inner module:
    # gamma=0 must give x, never 2*x.
    context = block.context_module(x)
    torch.testing.assert_close(context, x)
    module = block.context_module.main
    with torch.no_grad():
        module.gamma.fill_(0.1)
    block(x).square().mean().backward()
    assert torch.isfinite(x.grad).all()
    for name in ('gamma', 'offset_mlp.2.weight',
                 'attn.in_proj_weight', 'fuse.weight'):
        grad = dict(module.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all(), name
        assert grad.abs().sum() > 0, name


class TestEfficientViTCRA(unittest.TestCase):

    def test_curve_seed_coverage(self):
        for num_curves, points, length in (
                (16, 16, 0.8), (36, 24, 0.7), (16, 32, 1.2)):
            with self.subTest(num_curves=num_curves, length=length):
                _check_curve_seed_coverage(num_curves, points, length)

    def test_single_identity_shortcut_and_backward(self):
        _check_single_identity_shortcut_and_backward()


if __name__ == '__main__':
    unittest.main()
