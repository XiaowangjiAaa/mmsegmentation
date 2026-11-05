"""Encoder mechanism, checkpoint compatibility and gradient checks."""
import unittest

import torch

from mmseg.models.backbones.efficientvit.backbone import EfficientViT
from mmseg.models.backbones.efficientvit.layers import EfficientViTBlock, LiteMLA, MBConv
from mmseg.models.backbones.efficientvit.structure_features import (
    ContextGuidedLocalBlock, StructureDownsample, StructureQueryBlock,
    StructureQueryLiteMLA, split_phases)


class TestStructureFeatures(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)

    def test_phase_decomposition_and_ordered_compression(self):
        x = torch.arange(35.).reshape(1, 1, 5, 7)
        mean, residual = split_phases(x)
        torch.testing.assert_close(residual.sum(2), torch.zeros_like(mean))
        phases = (mean[:, :, None] + residual).flatten(1, 2)
        torch.testing.assert_close(
            torch.nn.functional.pixel_shuffle(phases, 2)[..., :5, :7], x)
        module = StructureDownsample(1, 2, stride=2, norm=None)
        with torch.no_grad():
            module.structure_reduce.weight.fill_(1)
            module.structure_encode.weight.zero_()
            module.structure_encode.weight[:, 0].fill_(1)
        self.assertGreater(module.structure_update(x).abs().max().item(), 0)
        self.assertEqual(
            module.structure_update(torch.ones_like(x)).abs().max().item(), 0)
        for shape in [(2, 1, 5, 7), (2, 1, 1, 1), (2, 1, 6, 8)]:
            with torch.no_grad():
                out = module(torch.randn(shape))
            self.assertEqual(out.shape[-2:],
                             ((shape[-2] + 1) // 2, (shape[-1] + 1) // 2))

    def test_query_changes_queries_only_and_respects_constant_input(self):
        module = StructureQueryLiteMLA(16, 16, dim=4)
        x = torch.randn(2, 16, 5, 7)
        qkv = torch.randn(2, 96, 5, 7)
        modified = module.condition_queries(qkv, x)
        a, b = qkv.reshape(2, -1, 3, 4, 5,
                           7), modified.reshape(2, -1, 3, 4, 5, 7)
        torch.testing.assert_close(a[:, :, 1:], b[:, :, 1:])
        self.assertGreater((a[:, :, 0] - b[:, :, 0]).abs().max().item(), 0)
        torch.testing.assert_close(
            module.condition_queries(qkv, torch.ones_like(x)), qkv)

    def test_zero_scales_preserve_original_functions_and_parameter_keys(self):
        pairs = [
            (MBConv(8, 16, stride=2), StructureDownsample(8, 16, stride=2)),
            (LiteMLA(16, 16, dim=4), StructureQueryLiteMLA(16, 16, dim=4)),
            (EfficientViTBlock(16, dim=4), ContextGuidedLocalBlock(16, dim=4)),
            (EfficientViTBlock(16, dim=4), StructureQueryBlock(16, dim=4))
        ]
        for original, modified in pairs:
            missing, unexpected = modified.load_state_dict(
                original.state_dict(), strict=False)
            self.assertFalse(unexpected)
            self.assertTrue(missing)
            self.assertTrue(
                all('structure_' in key for key in missing), missing)
            for name, p in modified.named_parameters():
                if name.endswith('structure_scale'):
                    p.data.zero_()
            original.eval()
            modified.eval()
            channels = 8 if isinstance(modified, StructureDownsample) else 16
            x = torch.randn(2, channels, 7, 9)
            torch.testing.assert_close(original(x), modified(x))

    def test_context_actually_changes_spatial_routing(self):
        module = ContextGuidedLocalBlock(16, dim=4)
        x, context = torch.randn(2, 16, 5, 7), torch.randn(2, 16, 5, 7)
        a, wa = module.local_update(x, torch.zeros_like(context))
        b, wb = module.local_update(x, context)
        torch.testing.assert_close(wb.sum(1), torch.ones_like(wb[:, 0]))
        self.assertGreater((wa - wb).abs().max().item(), 0)
        self.assertGreater((a - b).abs().max().item(), 0)

    def test_new_parameters_receive_finite_nonzero_gradients(self):
        for cls in [
                StructureDownsample, StructureQueryLiteMLA,
                ContextGuidedLocalBlock
        ]:
            module = (
                cls(16, 16, stride=2)
                if cls is StructureDownsample else cls(16, 16, dim=4)
                if cls is StructureQueryLiteMLA else cls(16, dim=4))
            for amp in [False, True]:
                module.zero_grad(set_to_none=True)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    output = module(torch.randn(2, 16, 7, 9))
                    loss = (output.float() *
                            torch.randn_like(output.float())).mean()
                loss.backward()
                for name, p in module.named_parameters():
                    if 'structure_' in name:
                        self.assertIsNotNone(p.grad, name)
                        self.assertTrue(torch.isfinite(p.grad).all(), name)
                        self.assertGreater(p.grad.abs().sum().item(), 0, name)

    def test_backbone_outputs_checkpoint_keys_and_stage_placement(self):
        original = EfficientViT()
        for args in [
                dict(downsample_mode='structure'),
                dict(block_type='structure_query'),
                dict(block_type='context_local'),
                dict(
                    downsample_mode='structure', block_type='structure_query')
        ]:
            model = EfficientViT(**args).eval()
            missing, unexpected = model.load_state_dict(
                original.state_dict(), strict=False)
            self.assertFalse(unexpected)
            self.assertTrue(all('structure_' in key for key in missing))
            if args.get('block_type'):
                cls = StructureQueryBlock if args[
                    'block_type'] == 'structure_query' else ContextGuidedLocalBlock
                self.assertEqual(
                    sum(isinstance(m, cls) for m in model.modules()), 4)
            if args.get('downsample_mode'):
                locations = [
                    name for name, m in model.named_modules()
                    if isinstance(m, StructureDownsample)
                ]
                self.assertEqual(
                    locations,
                    ['stages.1.op_list.0.main', 'stages.2.op_list.0.main'])
            for shape in [(64, 64), (65, 97)]:
                with torch.no_grad():
                    features = model(torch.randn(1, 3, *shape))
                self.assertEqual([f.shape[1] for f in features],
                                 [16, 32, 64, 128])
                self.assertEqual([f.shape[-2:] for f in features],
                                 [((shape[0] + s - 1) // s,
                                   (shape[1] + s - 1) // s)
                                  for s in [4, 8, 16, 32]])
        with self.assertRaises(ValueError):
            EfficientViT(arch='l0', downsample_mode='structure')


if __name__ == '__main__':
    unittest.main()
