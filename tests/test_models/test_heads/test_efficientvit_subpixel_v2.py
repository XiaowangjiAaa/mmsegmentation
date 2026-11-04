"""Subpixel write/read and reconstruction invariants (unittest/pytest)."""
import unittest

import torch

from mmseg.models.decode_heads.efficientvit_structure_modules import (
    SubpixelStructureMemory, resize_like)
from mmseg.models.decode_heads.efficientvit_subpixel_v2 import (
    StableSubpixelMemory, SubpixelContrastMemory, SubpixelMemoryV2,
    SubpixelWriteContrastMemory, SubpixelWriteMemory, memory_state,
    phase_tokens)


class TestSubpixelV2(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(11)
        torch.set_num_threads(2)

    def test_original_checkpoint_and_function_are_preserved(self):
        original = SubpixelStructureMemory(4)
        adapter = SubpixelMemoryV2(4)
        adapter.load_state_dict(original.state_dict(), strict=True)
        coarse, detail = torch.randn(2, 4, 4, 5), torch.randn(2, 4, 7, 9)
        torch.testing.assert_close(
            adapter(coarse, detail)[0],
            original(coarse, detail)[0])
        # Stable sampling and centered reads add no learned parameters.
        for cls in (StableSubpixelMemory, SubpixelContrastMemory):
            cls(4).load_state_dict(original.state_dict(), strict=True)

    def test_nonlinear_residual_keys_prevent_guaranteed_cancellation(self):
        module = SubpixelWriteMemory(2)
        with torch.no_grad():
            for projection in (module.structure_query, module.structure_key,
                               module.structure_value):
                projection.weight.copy_(torch.eye(2).reshape(2, 2, 1, 1))
            for gate in (module.write_confidence, module.read_confidence):
                gate.weight.zero_()
                gate.bias.zero_()
        detail = torch.tensor([[[[1., -1.], [0., 0.]], [[0., 0.], [1., -1.]]]])
        _, residual, _ = module.split_phases(detail)
        values = phase_tokens(residual)
        naive_state, _ = memory_state(torch.ones_like(values), values)
        self.assertEqual(float(naive_state.abs().sum()), 0)
        _, components = module.forward_components(
            torch.zeros(1, 2, 1, 1), detail)
        self.assertGreater(float(components['geometry_read'].abs().sum()), 0)

    def test_constant_phase_features_write_no_geometry(self):
        module = SubpixelWriteMemory(4)
        _, components = module.forward_components(
            torch.randn(2, 4, 3, 4), torch.ones(2, 4, 6, 8))
        self.assertEqual(float(components['geometry_read'].abs().sum()), 0)

    def test_centered_reads_preserve_the_shared_semantic_mean(self):
        for cls in (SubpixelContrastMemory, SubpixelWriteContrastMemory):
            module = cls(4)
            _, c = module.forward_components(
                torch.randn(2, 4, 4, 5), torch.randn(2, 4, 7, 9))
            torch.testing.assert_close(
                c['contrast'].mean(1),
                torch.zeros_like(c['shared_read']),
                atol=1e-6,
                rtol=0)
            torch.testing.assert_close(
                c['phases'].mean(1), c['shared_read'], atol=1e-6, rtol=1e-5)

    def test_filtered_split_reconstructs_and_retains_high_frequencies(self):
        module = StableSubpixelMemory(4)
        for h, w in ((8, 10), (7, 9), (1, 1)):
            detail = torch.randn(2, 4, h, w)
            low, residual = module.split_low_residual(detail)
            torch.testing.assert_close(
                resize_like(low, detail) + residual, detail)
            constant = torch.ones_like(detail)
            _, constant_residual = module.split_low_residual(constant)
            self.assertEqual(float(constant_residual.abs().sum()), 0)
        checkerboard = ((torch.arange(12)[:, None] + torch.arange(12)[None]) %
                        2)
        detail = (checkerboard.float() * 2 - 1)[None,
                                                None].expand(1, 4, -1, -1)
        low, residual = module.split_low_residual(detail)
        self.assertEqual(float(low[..., 1:-1, 1:-1].abs().max()), 0)
        self.assertGreater(float(residual.square().mean()), 0.5)

    def test_new_mechanisms_backpropagate_under_cpu_autocast(self):
        classes = (SubpixelWriteMemory, SubpixelContrastMemory,
                   SubpixelWriteContrastMemory, StableSubpixelMemory)
        for cls in classes:
            with self.subTest(module=cls.__name__):
                module = cls(4)
                coarse = torch.randn(2, 4, 4, 5, requires_grad=True)
                detail = torch.randn(2, 4, 7, 9, requires_grad=True)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    out, aux = module(coarse, detail)
                    loss = out.square().mean()
                self.assertFalse(aux)
                self.assertEqual(out.shape, detail.shape)
                self.assertTrue(torch.isfinite(out).all())
                loss.backward()
                for name, p in module.named_parameters():
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
                    self.assertGreater(float(p.grad.abs().sum()), 0, name)
                self.assertGreater(float(detail.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
