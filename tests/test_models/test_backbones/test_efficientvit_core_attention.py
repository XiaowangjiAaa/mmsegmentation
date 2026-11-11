"""Dense-oracle and integration tests for LiteMLA core replacements."""
import unittest

import torch

from mmseg.models.backbones.efficientvit.backbone import EfficientViT
from mmseg.models.backbones.efficientvit.layers import LiteMLA
from mmseg.models.backbones.efficientvit.attn_plugins.core_attention import (
    CoreLiteMLA, bank_read, joint_read, local_global_statistics)


class TestCoreAttention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(29)
        torch.set_num_threads(2)

    def tensors(self, h=3, w=5):
        q = torch.rand(2, 2, h * w, 4, dtype=torch.float64)
        k = torch.rand_like(q)
        v = torch.randn_like(q)
        return q, k, v

    def test_bank_read_matches_explicit_pairwise_relation_kernel(self):
        q, k, v = self.tensors()
        routes = torch.randn(2, 15, 2, dtype=torch.float64).softmax(-1)
        out, _ = bank_read(q, k, v, routes)
        affinity = (q @ k.transpose(-1, -2)) * (
            routes @ routes.transpose(-1, -2))[:, None]
        expected = (affinity @ v) / (affinity.sum(-1, keepdim=True) + 1e-15)
        torch.testing.assert_close(out, expected)

    def test_uniform_routes_recover_original_global_read(self):
        q, k, v = self.tensors()
        expected = q @ (k.transpose(-1, -2) @ v) / (
            q @ k.sum(-2, keepdim=True).transpose(-1, -2) + 1e-15)
        for banks in [1, 2, 3]:
            routes = torch.full((2, 15, banks), 1 / banks, dtype=torch.float64)
            torch.testing.assert_close(bank_read(q, k, v, routes)[0], expected)
        original = LiteMLA(16, 16, dim=4).eval()
        controlled = CoreLiteMLA(16, 16, dim=4, mode='bank_uniform').eval()
        controlled.load_state_dict(original.state_dict(), strict=True)
        x = torch.randn(2, 16, 5, 7)
        torch.testing.assert_close(original(x), controlled(x))

    def test_local_global_statistics_match_dense_mask_without_overlap(self):
        for h, w in [(3, 5), (1, 1), (1, 4), (2, 2)]:
            q, k, v = self.tensors(h, w)
            stats = local_global_statistics(q, k, v, h, w)
            positions = torch.cartesian_prod(torch.arange(h), torch.arange(w))
            local = (positions[:, None] - positions[None]).abs().amax(-1) <= 1
            affinity = q @ k.transpose(-1, -2)
            for name, mask in [('local', local), ('global', ~local)]:
                scores = affinity * mask
                torch.testing.assert_close(stats[name + '_num'], scores @ v)
                torch.testing.assert_close(stats[name + '_mass'],
                                           scores.sum(-1, keepdim=True))
                torch.testing.assert_close(stats[name + '_count'][0, 0, :, 0],
                                           mask.sum(-1).to(q.dtype))

    def test_joint_and_postfusion_match_distinct_dense_definitions(self):
        q, k, v = self.tensors()
        stats = local_global_statistics(q, k, v, 3, 5)
        weight = torch.rand(2, 2, 15, 1, dtype=torch.float64) + .1
        positions = torch.cartesian_prod(torch.arange(3), torch.arange(5))
        local = (positions[:, None] - positions[None]).abs().amax(-1) <= 1
        base = q @ k.transpose(-1, -2)
        scores = base * (
            local / stats['local_count'] + weight *
            (~local) / stats['global_count'].clamp_min(1))
        expected = scores @ v / (scores.sum(-1, keepdim=True) + 1e-15)
        joint = joint_read(stats, weight)
        torch.testing.assert_close(joint, expected)
        fraction = weight / (1 + weight)
        expected_post = (1 - fraction) * stats['local_num'] / (
            stats['local_mass'] + 1e-15) + fraction * stats['global_num'] / (
                stats['global_mass'] + 1e-15)
        torch.testing.assert_close(
            joint_read(stats, weight, postfusion=True), expected_post)
        self.assertGreater(float((joint - expected_post).abs().max()), 0)
        # Compensating the count balance recovers the original all-token read.
        all_weight = stats['global_count'] / stats['local_count']
        original = base @ v / (base.sum(-1, keepdim=True) + 1e-15)
        torch.testing.assert_close(joint_read(stats, all_weight), original)

    def test_empty_complement_and_zero_kernel_are_finite(self):
        for h, w in [(1, 1), (1, 2), (2, 2)]:
            q, k, v = self.tensors(h, w)
            stats = local_global_statistics(q, k, v, h, w)
            self.assertEqual(float(stats['global_mass'].abs().max()), 0)
            for post in [False, True]:
                output = joint_read(
                    stats,
                    torch.ones_like(stats['local_mass']),
                    postfusion=post)
                torch.testing.assert_close(
                    output, stats['local_num'] / (stats['local_mass'] + 1e-15))
        q, k, v = self.tensors()
        q.zero_()
        k.zero_()
        routes = torch.full((2, 15, 2), .5, dtype=q.dtype)
        self.assertEqual(float(bank_read(q, k, v, routes)[0].abs().max()), 0)
        stats = local_global_statistics(q, k, v, 3, 5)
        for post in [False, True]:
            out = joint_read(
                stats, torch.ones_like(stats['local_mass']), postfusion=post)
            self.assertTrue(torch.isfinite(out).all())
            self.assertEqual(float(out.abs().max()), 0)

    def test_modes_have_finite_gradients_in_fp32_and_bfloat16(self):
        for mode in CoreLiteMLA.MODES:
            model = CoreLiteMLA(16, 16, dim=4, mode=mode)
            for amp in [False, True]:
                model.zero_grad(set_to_none=True)
                with torch.autocast('cpu', dtype=torch.bfloat16, enabled=amp):
                    out = model(torch.randn(2, 16, 5, 7))
                    loss = (out.float() * torch.randn_like(out.float())).mean()
                loss.backward()
                for name, p in model.named_parameters():
                    self.assertIsNotNone(p.grad, (mode, name))
                    self.assertTrue(torch.isfinite(p.grad).all(), (mode, name))
                    if name.startswith('routing_'):
                        self.assertGreater(
                            float(p.grad.abs().sum()), 0, (mode, name))
        a = CoreLiteMLA(16, 16, dim=4, mode='bank_structure')
        b = CoreLiteMLA(16, 16, dim=4, mode='bank_content')
        self.assertEqual(
            sum(p.numel() for p in a.parameters()),
            sum(p.numel() for p in b.parameters()))
        a = CoreLiteMLA(16, 16, dim=4, mode='joint')
        b = CoreLiteMLA(16, 16, dim=4, mode='postfusion')
        self.assertEqual(
            sum(p.numel() for p in a.parameters()),
            sum(p.numel() for p in b.parameters()))

    def test_all_four_core_operators_replace_attention_and_keep_old_keys(self):
        torch.manual_seed(101)
        original = EfficientViT().eval()
        for mode in CoreLiteMLA.MODES:
            torch.manual_seed(101)
            model = EfficientViT(block_type='core_' + mode).eval()
            self.assertEqual(
                sum(isinstance(m, CoreLiteMLA) for m in model.modules()), 4)
            for key, value in original.state_dict().items():
                torch.testing.assert_close(
                    model.state_dict()[key], value, rtol=0, atol=0)
            missing, unexpected = model.load_state_dict(
                original.state_dict(), strict=False)
            self.assertFalse(unexpected)
            self.assertTrue(
                all('.routing_' in key for key in missing), missing)
            state = model.state_dict()
            for key, value in original.state_dict().items():
                torch.testing.assert_close(state[key], value)
            with torch.no_grad():
                features = model(torch.randn(1, 3, 65, 97))
            self.assertEqual([f.shape[1] for f in features], [16, 32, 64, 128])
            self.assertEqual([tuple(f.shape[-2:]) for f in features],
                             [(17, 25), (9, 13), (5, 7), (3, 4)])


if __name__ == '__main__':
    unittest.main()
