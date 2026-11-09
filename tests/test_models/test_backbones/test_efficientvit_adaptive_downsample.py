"""Spatial aggregation correctness, routing, gradients and encoder placement."""
import unittest

import torch
from torch.nn import functional as F

from mmseg.models.backbones.efficientvit.adaptive_downsample import (
    AdaptiveDownsampleMBConv, LocalStructureAggregation)
from mmseg.models.backbones.efficientvit.backbone import EfficientViT
from mmseg.models.backbones.efficientvit.layers import LiteMLA


class TestAdaptiveDownsample(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(23)
        torch.set_num_threads(2)

    def test_uniform_matches_valid_neighbor_average(self):
        module = LocalStructureAggregation(8, mode='uniform')
        for h, w in [(6, 8), (5, 7), (1, 1), (1, 7), (7, 1)]:
            x = torch.randn(2, 8, h, w)
            torch.testing.assert_close(
                module(x), F.avg_pool2d(x, 3, 2, 1, count_include_pad=False))

    def test_weights_normalize_exclude_padding_and_preserve_constants(self):
        for mode in ['structure', 'content', 'static', 'uniform']:
            module = LocalStructureAggregation(8, mode=mode)
            for h, w in [(5, 7), (1, 1), (1, 7), (7, 1)]:
                x = torch.full((2, 8, h, w), 2.5)
                out, parts = module.forward_components(x)
                weights, valid = parts['weights'], parts['valid']
                torch.testing.assert_close(
                    weights.sum(2), torch.ones_like(weights[:, :, 0]))
                self.assertTrue((weights >= 0).all())
                self.assertEqual(
                    float(
                        weights.masked_select(
                            ~valid.expand_as(weights)).abs().sum()), 0)
                torch.testing.assert_close(out, torch.full_like(out, 2.5))
                self.assertEqual(out.shape[-2:], ((h + 1) // 2, (w + 1) // 2))

    def test_groups_can_select_different_neighbors(self):
        module = LocalStructureAggregation(2, groups=2, mode='static')
        with torch.no_grad():
            module.position_bias[0, 4] = 50  # center
            module.position_bias[1, 3] = 50  # left
        x = torch.arange(25.).reshape(1, 1, 5, 5).expand(1, 2, -1, -1)
        out = module(x)
        torch.testing.assert_close(out[0, :, 1, 1], torch.tensor([12., 11.]))

    def test_dynamic_weights_depend_on_structure_and_content(self):
        for mode in ['structure', 'content']:
            module = LocalStructureAggregation(8, mode=mode)
            constant = torch.ones(2, 8, 5, 7)
            perturbed = constant.clone()
            perturbed[:, :, 1, 1] += 2
            a, _ = module.routing_weights(constant)
            b, _ = module.routing_weights(perturbed)
            self.assertGreater(float((a - b).abs().max()), 0)
        structure = LocalStructureAggregation(8, mode='structure')
        content = LocalStructureAggregation(8, mode='content')
        self.assertEqual(
            sum(p.numel() for p in structure.parameters()),
            sum(p.numel() for p in content.parameters()))

    def test_trainable_parameters_and_inputs_receive_gradients(self):
        for mode in ['structure', 'content', 'static', 'uniform']:
            module = LocalStructureAggregation(8, mode=mode)
            for amp in [False, True]:
                module.zero_grad(set_to_none=True)
                x = torch.randn(
                    2, 8, 7, 9, requires_grad=True,
                    dtype=torch.bfloat16 if amp else torch.float32)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    y = module(x)
                    loss = (y.float() * torch.randn_like(y.float())).mean()
                loss.backward()
                self.assertTrue(torch.isfinite(x.grad).all())
                self.assertGreater(float(x.grad.abs().sum()), 0)
                for name, p in module.named_parameters():
                    self.assertIsNotNone(p.grad, (mode, name))
                    self.assertTrue(torch.isfinite(p.grad).all(), (mode, name))
                    self.assertGreater(
                        float(p.grad.abs().sum()), 0, (mode, name))

    def test_only_stride8_to16_spatial_kernel_changes(self):
        original = EfficientViT().eval()
        for mode in ['structure', 'content', 'static', 'uniform']:
            model = EfficientViT(downsample_mode='adaptive_' + mode).eval()
            locations = [
                n for n, m in model.named_modules()
                if isinstance(m, LocalStructureAggregation)
            ]
            self.assertEqual(locations,
                             ['stages.2.op_list.0.main.depth_conv.conv'])
            self.assertIsInstance(model.stages[2].op_list[0].main,
                                  AdaptiveDownsampleMBConv)
            self.assertEqual(
                sum(isinstance(m, LiteMLA) for m in model.modules()), 4)
            prefix = 'stages.2.op_list.0.main.depth_conv.conv.'
            state = {
                k: v
                for k, v in original.state_dict().items()
                if not k.startswith(prefix)
            }
            missing, unexpected = model.load_state_dict(state, strict=False)
            self.assertFalse(unexpected)
            self.assertTrue(all(k.startswith(prefix) for k in missing))
            for k, v in state.items():
                torch.testing.assert_close(model.state_dict()[k], v)
            with torch.no_grad():
                outputs = model(torch.randn(1, 3, 65, 97))
            self.assertEqual([f.shape[1] for f in outputs], [16, 32, 64, 128])
            self.assertEqual([tuple(f.shape[-2:]) for f in outputs], [(17, 25),
                                                                      (9, 13),
                                                                      (5, 7),
                                                                      (3, 4)])

    def test_reject_invalid_groups_and_combinations(self):
        with self.assertRaises(ValueError):
            LocalStructureAggregation(8, groups=3)
        with self.assertRaises(ValueError):
            AdaptiveDownsampleMBConv(8, 16, stride=1)
        with self.assertRaises(ValueError):
            EfficientViT(arch='l0', downsample_mode='adaptive_structure')
        with self.assertRaises(ValueError):
            EfficientViT(
                transition_mode='semantic_structure',
                downsample_mode='adaptive_structure')


if __name__ == '__main__':
    unittest.main()
