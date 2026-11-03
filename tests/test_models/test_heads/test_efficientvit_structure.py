"""Mechanism invariants for the cross-scale structure heads."""
import unittest

import torch
import torch.nn.functional as F
from mmengine.structures import PixelData

from mmseg.models.decode_heads.efficientvit_head import EfficientViTHead
from mmseg.models.decode_heads.efficientvit_structure_head import (
    EfficientViTStructureHead, balanced_binary_loss, occupancy_targets)
from mmseg.models.decode_heads.efficientvit_structure_modules import (
    DIRECTIONS, ReciprocalConnectivity, StructurePartitionMemory,
    SubpixelStructureMemory, linear_read, neighbor)
from mmseg.structures import SegDataSample


class TestStructureMechanisms(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_compressed_read_matches_explicit_attention(self):
        q, k, v = torch.rand(2, 7, 4), torch.rand(2, 5,
                                                  4), torch.randn(2, 5, 4)
        weight = torch.rand(2, 5, 1) + 0.1
        attention = (q @ k.transpose(1, 2)) * weight.transpose(1, 2)
        expected = (attention @ v) / attention.sum(-1, keepdim=True)
        torch.testing.assert_close(linear_read(q, k, v, weight), expected)
        # Independent normalization makes a bank invariant to a uniform
        # multiplicative change in its population weights.
        torch.testing.assert_close(
            linear_read(q, k, v, weight), linear_read(q, k, v, weight * 3))

    def test_partition_routing_uses_detail_and_has_gradients(self):
        module = StructurePartitionMemory(8)
        coarse = torch.randn(2, 8, 4, 5, requires_grad=True)
        detail = torch.randn(2, 8, 8, 10, requires_grad=True)
        out, aux = module(coarse, detail)
        _, changed = module(coarse, detail + 1)
        self.assertGreater(
            float((aux['route_logits'] - changed['route_logits']).abs().max()),
            0)
        out.square().mean().backward()
        for name in ('route.0.weight', 'query.weight', 'key.weight',
                     'read_gate.weight'):
            grad = dict(module.named_parameters())[name].grad
            self.assertIsNotNone(grad, name)
            self.assertTrue(torch.isfinite(grad).all(), name)
            self.assertGreater(float(grad.abs().sum()), 0, name)

    def test_connections_are_reciprocal_and_do_not_wrap(self):
        logits, semantic = torch.randn(2, 8, 7, 9), torch.randn(2, 8, 7, 9)
        weights = ReciprocalConnectivity.reciprocal_weights(logits, semantic)
        for index, (dy, dx) in enumerate(DIRECTIONS):
            opposite = DIRECTIONS.index((-dy, -dx))
            torch.testing.assert_close(
                weights[:, index:index + 1],
                neighbor(weights[:, opposite:opposite + 1], dy, dx))
        self.assertEqual(float(weights[:, 0, 0].abs().sum()), 0)
        impulse = torch.zeros(1, 1, 5, 5)
        impulse[..., 0, 0] = 1
        self.assertEqual(float(neighbor(impulse, 0, 1)[..., -1].sum()), 0)

    def test_propagation_preserves_constant_state_at_borders(self):
        module = ReciprocalConnectivity(8)
        out, _ = module(torch.ones(2, 8, 3, 4), torch.ones(2, 8, 7, 9))
        torch.testing.assert_close(out, torch.full_like(out, 3))

    def test_phase_decomposition_reconstructs_even_and_odd_inputs(self):
        for h, w in ((8, 10), (7, 9), (1, 1)):
            detail = torch.randn(2, 8, h, w)
            mean, residual, size = SubpixelStructureMemory.split_phases(detail)
            torch.testing.assert_close(
                residual.sum(1), torch.zeros_like(mean), atol=1e-6, rtol=0)
            actual = SubpixelStructureMemory.merge_phases(
                mean[:, None] + residual, size)
            torch.testing.assert_close(actual, detail)

    def test_module_gradients_and_cpu_autocast(self):
        for module_type in (StructurePartitionMemory, ReciprocalConnectivity,
                            SubpixelStructureMemory):
            with self.subTest(module=module_type.__name__):
                module = module_type(8)
                coarse = torch.randn(2, 8, 4, 5, requires_grad=True)
                detail = torch.randn(2, 8, 7, 9, requires_grad=True)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    out, _ = module(coarse, detail)
                    loss = out.square().mean()
                self.assertEqual(out.shape, detail.shape)
                self.assertTrue(torch.isfinite(out).all())
                loss.backward()
                for name, p in module.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    self.assertGreater(float(p.grad.abs().sum()), 0, name)

    def test_auxiliary_targets_ignore_padding_and_empty_classes(self):
        gt = torch.full((2, 1, 8, 8), 255, dtype=torch.long)
        for touched in (False, True):
            target, valid = occupancy_targets(gt, (4, 4), touched=touched)
            logits = torch.randn_like(target, requires_grad=True)
            loss = balanced_binary_loss(logits, target, valid)
            self.assertEqual(float(loss), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
        for value in (0, 1):
            target, valid = occupancy_targets(
                torch.full_like(gt, value), (4, 4))
            self.assertTrue(
                torch.isfinite(
                    balanced_binary_loss(
                        torch.zeros_like(target), target, valid)))

    def test_head_checkpoint_compatibility_and_losses(self):
        kwargs = dict(
            in_channels=[128, 64, 32],
            head_width=32,
            head_depth=1,
            num_classes=2,
            out_channels=2,
            dropout_ratio=0,
            loss_decode=[
                dict(type='CrossEntropyLoss', avg_non_ignore=True),
                dict(type='DiceLoss', use_sigmoid=False)
            ])
        plain = EfficientViTHead(**kwargs).eval()
        features = [
            torch.randn(2, c, h, w)
            for c, h, w in ((16, 17, 19), (32, 9, 10), (64, 5, 5), (128, 3, 3))
        ]
        expected = F.interpolate(
            plain(features),
            size=(17, 19),
            mode='bilinear',
            align_corners=False)
        for mode in ('partition', 'connectivity', 'subpixel'):
            head = EfficientViTStructureHead(
                structure_mode=mode, **kwargs).eval()
            mismatch = head.load_state_dict(plain.state_dict(), strict=False)
            self.assertFalse(mismatch.unexpected_keys)
            with torch.no_grad():
                head.structure_scale.zero_()
                torch.testing.assert_close(head(features), expected)
                head.structure_scale.fill_(0.01)
            samples = []
            for _ in range(2):
                gt = torch.randint(0, 2, (1, 68, 76))
                gt[:, -4:, -4:] = 255
                sample = SegDataSample()
                sample.gt_sem_seg = PixelData(data=gt)
                samples.append(sample)
            losses = head.loss(features, samples, {})
            if mode == 'partition':
                self.assertIn('loss_structure_route', losses)
            elif mode == 'connectivity':
                self.assertIn('loss_structure_links', losses)
            else:
                self.assertNotIn('loss_structure_route', losses)
                self.assertNotIn('loss_structure_links', losses)
            sum(v for k, v in losses.items() if 'loss' in k).backward()
            grad = head.structure_classifier.weight.grad
            self.assertIsNotNone(grad)
            self.assertGreater(float(grad.abs().sum()), 0)
            # Inference takes features only; no cached labels or aux state.
            self.assertEqual(head(features).shape, (2, 2, 17, 19))


if __name__ == '__main__':
    unittest.main()
