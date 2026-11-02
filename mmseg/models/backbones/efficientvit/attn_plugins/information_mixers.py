"""Checkpoint-compatible lightweight information-mixing ablations.

These are EfficientViT adaptations inspired by MALA, PolaFormer, RALA and
stage-specific channel mixing, not reproductions of their full models.
All modes retain LiteMLA's qkv, multi-scale aggregation and projection keys.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import LiteMLA


class InformationLiteMLA(LiteMLA):
    """Change only the mixer; never materialize a spatial N-by-N matrix.

    magnitude: zero-gated, centered KV covariance restores query scale.
    polarity: zero-gated same-sign minus opposite-sign interaction.
    kv_reweight: globally conditioned positive token weights; uniform at init.
    channel: channel-to-channel interaction instead of spatial attention.
    """

    def __init__(self, *args, mode='magnitude', **kwargs):
        super().__init__(*args, **kwargs)
        if mode not in ('magnitude', 'polarity', 'kv_reweight', 'channel'):
            raise ValueError(f'Unknown information mixer: {mode}')
        self.mode = mode
        if mode != 'channel':
            self.mix_gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _statistics(q, k, v):
        numerator = q @ (k.transpose(-1, -2) @ v)
        mass = q @ k.sum(dim=-2, keepdim=True).transpose(-1, -2)
        return numerator, mass

    @torch.cuda.amp.autocast(enabled=False)
    def relu_linear_att(self, qkv):
        b, _, h, w = qkv.shape
        # Use fp32 for both float16 and bfloat16 accumulation.
        if qkv.dtype in (torch.float16, torch.bfloat16):
            qkv = qkv.float()
        packed = qkv.reshape(b, -1, 3 * self.dim, h * w).transpose(-1, -2)
        q_raw, k_raw, v = packed.split(self.dim, dim=-1)
        if self.mode == 'channel':
            q = F.normalize(q_raw, dim=-2, eps=1e-6)
            k = F.normalize(k_raw, dim=-2, eps=1e-6)
            weights = (k.transpose(-1, -2) @ q).softmax(dim=-1)
            out = v @ weights
        else:
            q, k = F.relu(q_raw), F.relu(k_raw)
            if self.mode == 'kv_reweight':
                score = (q_raw.mean(dim=-2, keepdim=True) * k).sum(-1)
                score = score / self.dim**0.5
                # gamma=0 gives exactly uniform, non-negative token weights.
                weight = (self.mix_gamma * score).softmax(dim=-1)
                k = k * (weight * (h * w)).unsqueeze(-1)
            numerator, mass = self._statistics(q, k, v)
            out = numerator / (mass + self.eps)
            if self.mode == 'magnitude':
                # This centered covariance is linear in query magnitude.
                # Divide by N*d so its scale does not grow with resolution.
                covariance = (numerator - mass * v.mean(-2, keepdim=True))
                covariance = covariance / ((h * w) * self.dim)
                out = out + self.mix_gamma.tanh() * covariance
            elif self.mode == 'polarity':
                q_neg, k_neg = F.relu(-q_raw), F.relu(-k_raw)
                nn_num, nn_mass = self._statistics(q_neg, k_neg, v)
                pn_num, pn_mass = self._statistics(q, k_neg, v)
                np_num, np_mass = self._statistics(q_neg, k, v)
                same = (numerator + nn_num) / (mass + nn_mass + self.eps)
                opposite = (pn_num + np_num) / (pn_mass + np_mass + self.eps)
                out = out + self.mix_gamma.tanh() * (same - opposite)
        return out.transpose(-1, -2).reshape(b, -1, h, w)
