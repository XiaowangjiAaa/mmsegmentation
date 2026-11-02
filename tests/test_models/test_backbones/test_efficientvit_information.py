"""Functional invariants for the lightweight ablations (unittest/pytest)."""
import unittest

import torch

from mmseg.models.backbones.efficientvit.attn_plugins.blocks import (
    EfficientViTBlockInformation)
from mmseg.models.backbones.efficientvit.attn_plugins.information_mixers import (
    InformationLiteMLA)
from mmseg.models.backbones.efficientvit.layers import LiteMLA
from mmseg.models.decode_heads.efficientvit_head import EfficientViTHead


class TestInformationMixers(unittest.TestCase):

    def test_zero_init_preserves_pretrained_function(self):
        torch.manual_seed(4)
        original = LiteMLA(16, 16, dim=8).eval()
        x = torch.randn(2, 16, 7, 9)
        for mode in ('magnitude', 'polarity', 'kv_reweight'):
            with self.subTest(mode=mode):
                mixer = InformationLiteMLA(16, 16, dim=8, mode=mode).eval()
                missing = mixer.load_state_dict(original.state_dict(), strict=False)
                self.assertEqual(missing.missing_keys, ['mix_gamma'])
                self.assertFalse(missing.unexpected_keys)
                torch.testing.assert_close(mixer(x), original(x), rtol=1e-5, atol=1e-6)

    def test_nonzero_gate_changes_output_and_backpropagates(self):
        torch.manual_seed(6)
        for mode in ('magnitude', 'polarity', 'kv_reweight', 'channel'):
            with self.subTest(mode=mode):
                mixer = InformationLiteMLA(16, 16, dim=8, mode=mode).eval()
                if mode != 'channel':
                    with torch.no_grad():
                        mixer.mix_gamma.fill_(0.2)
                x = torch.randn(2, 16, 7, 9, requires_grad=True)
                output = mixer(x)
                output.square().mean().backward()
                for parameter in (x, mixer.qkv.conv.weight, mixer.proj.conv.weight):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(float(parameter.grad.abs().sum()), 0)
                if mode != 'channel':
                    self.assertGreater(float(mixer.mix_gamma.grad.abs().sum()), 0)
                    with torch.no_grad():
                        mixer.mix_gamma.zero_()
                        self.assertGreater(float((mixer(x) - output).abs().max()), 0)

    def test_late_channel_only_changes_last_stage(self):
        early = EfficientViTBlockInformation(16, dim=8, stage_idx=2,
                                             mixer_type='late_channel')
        late = EfficientViTBlockInformation(16, dim=8, stage_idx=3,
                                            mixer_type='late_channel')
        self.assertIs(type(early.context_module.main), LiteMLA)
        self.assertEqual(late.context_module.main.mode, 'channel')

    def test_zero_and_low_precision_attention(self):
        for mode in ('magnitude', 'polarity', 'kv_reweight', 'channel'):
            mixer = InformationLiteMLA(16, 16, dim=8, mode=mode)
            for dtype in (torch.float32, torch.float16, torch.bfloat16):
                qkv = torch.zeros(1, 48, 3, 5, dtype=dtype)
                output = mixer.relu_linear_att(qkv)
                self.assertTrue(torch.isfinite(output).all())
                self.assertEqual(float(output.abs().sum()), 0)

    def test_detail_preserves_main_logits_at_initialization(self):
        kwargs = dict(in_channels=[128, 64, 32], head_width=32,
                      head_depth=1, num_classes=2)
        plain = EfficientViTHead(**kwargs).eval()
        detail = EfficientViTHead(**kwargs, detail_channels=8).eval()
        detail.load_state_dict(plain.state_dict(), strict=False)
        features = [torch.randn(2, c, h, w) for c, h, w in
                    ((16, 17, 19), (32, 9, 10), (64, 5, 5), (128, 3, 3))]
        expected = torch.nn.functional.interpolate(
            plain(features), size=(17, 19), mode='bilinear', align_corners=False)
        torch.testing.assert_close(detail(features), expected)
        with torch.no_grad():
            detail.detail_gamma.fill_(0.1)
        detail(features).square().mean().backward()
        self.assertGreater(float(detail.detail_branch[0].weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
