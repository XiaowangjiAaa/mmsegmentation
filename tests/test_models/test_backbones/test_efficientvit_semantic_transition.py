"""Replacement transition: decomposition, conditioning and stage integration."""
import unittest

import torch
from torch.nn import functional as F

from mmseg.models.backbones.efficientvit.backbone import EfficientViT
from mmseg.models.backbones.efficientvit.layers import IdentityLayer, LiteMLA
from mmseg.models.backbones.efficientvit.semantic_transition import (
    SemanticStructureTransition, phase_components)


class TestSemanticTransition(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(19)
        torch.set_num_threads(2)

    def test_components_reconstruct_odd_even_and_single_pixel_inputs(self):
        for h, w in [(6, 8), (5, 7), (1, 1), (1, 7), (5, 1)]:
            x = torch.randn(2, 3, h, w)
            m, dh, dv, dd = phase_components(x)
            phases = torch.stack([
                m + dh + dv + dd, m - dh + dv - dd, m + dh - dv - dd,
                m - dh - dv + dd
            ], 2)
            actual = F.pixel_shuffle(phases.flatten(1, 2), 2)[..., :h, :w]
            torch.testing.assert_close(actual, x)

    def test_directional_components_and_constant_regions(self):
        fixtures = [
            torch.tensor([[1., -1.], [1., -1.]]),
            torch.tensor([[1., 1.], [-1., -1.]]),
            torch.tensor([[1., -1.], [-1., 1.]])
        ]
        for direction, x in enumerate(fixtures):
            parts = phase_components(x[None, None])
            self.assertEqual(float(parts[0]), 0)
            for i, part in enumerate(parts[1:]):
                self.assertEqual(float(part), float(i == direction))
        for part in phase_components(torch.ones(1, 2, 5, 7))[1:]:
            self.assertEqual(float(part.abs().max()), 0)

    def test_global_context_changes_semantic_weights_only(self):
        mean = torch.randn(2, 8, 3, 5)
        differences = [torch.randn_like(mean) for _ in range(3)]
        a, b = torch.zeros(2, 16, 3, 5), torch.randn(2, 16, 3, 5)
        for mode in ['semantic', 'local', 'fixed']:
            module = SemanticStructureTransition(
                8, 16, dim=4, gate_mode=mode).eval()
            _, first = module.aggregate(mean, differences, a)
            _, second = module.aggregate(mean, differences, b)
            self.assertTrue((second['weights'] >= 0.05).all())
            self.assertTrue((second['weights'] <= 1).all())
            if mode == 'semantic':
                self.assertGreater(
                    float((first['weights'] - second['weights']).abs().max()),
                    0)
            else:
                torch.testing.assert_close(first['weights'], second['weights'])
            if mode == 'fixed':
                torch.testing.assert_close(
                    second['weights'], torch.full_like(second['weights'],
                                                       .525))

    def test_signed_structure_is_not_replaced_by_absolute_values(self):
        module = SemanticStructureTransition(8, 16, dim=4).eval()
        mean = torch.randn(2, 8, 3, 5)
        differences = [torch.randn_like(mean) for _ in range(3)]
        context = torch.randn(2, 16, 3, 5)
        out_a, a = module.aggregate(mean, differences, context)
        out_b, b = module.aggregate(mean, [-d for d in differences], context)
        torch.testing.assert_close(a['weights'], b['weights'])
        torch.testing.assert_close(a['structure'], -b['structure'])
        self.assertGreater(float((out_a - out_b).abs().max()), 0)

    def test_all_trainable_parameters_have_finite_nonzero_gradients(self):
        for mode in ['semantic', 'local', 'fixed', 'pooled']:
            module = SemanticStructureTransition(8, 16, dim=4, gate_mode=mode)
            for amp in [False, True]:
                module.zero_grad(set_to_none=True)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    output = module(torch.randn(2, 8, 7, 9))
                    loss = (output.float() *
                            torch.randn_like(output.float())).mean()
                loss.backward()
                for name, parameter in module.named_parameters():
                    self.assertIsNotNone(parameter.grad, (mode, name))
                    self.assertTrue(
                        torch.isfinite(parameter.grad).all(), (mode, name))
                    self.assertGreater(
                        float(parameter.grad.abs().sum()), 0, (mode, name))

    def test_only_one_transition_replaces_original_downsample_and_attention(
            self):
        original = EfficientViT().eval()
        model = EfficientViT(transition_mode='semantic_structure').eval()
        self.assertIsInstance(model.stages[2].op_list[0],
                              SemanticStructureTransition)
        self.assertIsInstance(model.stages[2].op_list[1].context_module,
                              IdentityLayer)
        self.assertFalse(hasattr(model.stages[2].op_list[0], 'main'))
        for item in [original, model]:
            self.assertEqual(
                sum(isinstance(m, LiteMLA) for m in item.modules()), 4)
        counts = {}
        handles = []
        for name, m in model.named_modules():
            if isinstance(m, LiteMLA):
                counts[name] = 0

                def record(module, inputs, output, key=name):
                    counts[key] += 1

                handles.append(m.register_forward_hook(record))
        with torch.no_grad():
            features = model(torch.randn(1, 3, 65, 97))
        self.assertEqual([f.shape[1] for f in features], [16, 32, 64, 128])
        self.assertEqual([tuple(f.shape[-2:]) for f in features], [(17, 25),
                                                                   (9, 13),
                                                                   (5, 7),
                                                                   (3, 4)])
        self.assertTrue(all(value == 1 for value in counts.values()))
        for handle in handles:
            handle.remove()

    def test_rejects_unimplemented_combinations(self):
        for kwargs in [
                dict(arch='l0'),
                dict(block_type='structure_query'),
                dict(downsample_mode='structure'),
                dict(transition_width=0)
        ]:
            with self.assertRaises(ValueError):
                EfficientViT(transition_mode='semantic_structure', **kwargs)


if __name__ == '__main__':
    unittest.main()
