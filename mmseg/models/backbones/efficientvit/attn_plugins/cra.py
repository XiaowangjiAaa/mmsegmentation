"""Curve-Routed Attention (CRA).

CRA replaces the dense grid token mixer with a small set of learned curves:

1. a fixed 4x4 set of seed points covers the image;
2. each seed predicts a K-point curve (base line + learned offsets);
3. features are sampled along the curves with ``grid_sample``;
4. curve tokens are mixed with curve-aware self-attention;
5. curve tokens are splatted back to the feature map.

The curve branch is zero-gated, so the module starts as an identity residual.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CRAModule(nn.Module):
    """Curve tokenizer + curve attention + splatting."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 num_curves: int = 16,
                 points_per_curve: int = 16,
                 num_heads: int = 4,
                 curve_length: float = 0.8,
                 offset_scale: float = 0.15,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 **kwargs):
        super().__init__()
        norm_cfg = norm_cfg or dict(type='BN')
        act_cfg = act_cfg or dict(type='Hardswish')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_curves = num_curves
        self.points_per_curve = points_per_curve
        self.offset_scale = offset_scale

        self.local = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                3,
                padding=1,
                groups=in_channels,
                bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Hardswish())

        self.seed_embed = nn.Parameter(
            torch.randn(num_curves, in_channels) * 0.02)
        self.point_embed = nn.Parameter(
            torch.randn(points_per_curve, in_channels) * 0.02)
        self.offset_mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, 2))
        self.attn = nn.MultiheadAttention(
            in_channels, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(in_channels)
        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 2),
            nn.GELU(),
            nn.Linear(in_channels * 2, in_channels))
        self.norm2 = nn.LayerNorm(in_channels)
        self.fuse = nn.Conv2d(in_channels * 2, out_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        seeds, base_offsets = self._build_curves(
            num_curves, points_per_curve, curve_length)
        self.register_buffer('seed_grid', seeds, persistent=False)
        self.register_buffer('base_offsets', base_offsets, persistent=False)

    @staticmethod
    def _build_curves(num_curves: int, points: int, curve_length: float):
        side = int(math.ceil(math.sqrt(num_curves)))
        positions = torch.linspace(-0.75, 0.75, side)
        seeds = torch.stack(
            torch.meshgrid(positions, positions, indexing='ij'),
            dim=-1).reshape(-1, 2)[:num_curves]
        t = torch.linspace(-0.5, 0.5, points)
        offsets = []
        for index in range(num_curves):
            angle = (index % 4) * math.pi / 4.0
            direction = torch.tensor(
                [math.cos(angle), math.sin(angle)], dtype=torch.float32)
            offsets.append(t.unsqueeze(1) * direction * curve_length)
        return seeds, torch.stack(offsets, dim=0)

    def _sample_curves(self, x: torch.Tensor):
        b, _, h, w = x.shape
        global_context = x.mean(dim=(2, 3))
        embedding = (self.seed_embed.unsqueeze(1) +
                     self.point_embed.unsqueeze(0) +
                     global_context[:, None, None, :])
        offsets = torch.tanh(self.offset_mlp(embedding)) * self.offset_scale
        coords = self.base_offsets.unsqueeze(0) + offsets
        coords = coords.clamp(-1.0, 1.0)
        sampled = F.grid_sample(
            x, coords, mode='bilinear', padding_mode='border',
            align_corners=True)
        tokens = sampled.flatten(2).transpose(1, 2)
        return tokens, coords

    @staticmethod
    def _splat(tokens: torch.Tensor, coords: torch.Tensor, h: int,
               w: int) -> torch.Tensor:
        b, c, m, k = tokens.shape
        n = m * k
        values = tokens.reshape(b, c, n)
        x = (coords[..., 0] + 1.0) * 0.5 * (w - 1)
        y = (coords[..., 1] + 1.0) * 0.5 * (h - 1)
        x = x.reshape(b, n)
        y = y.reshape(b, n)
        x0 = torch.floor(x).long()
        y0 = torch.floor(y).long()
        wx = x - x0
        wy = y - y0
        out = values.new_zeros(b, c, h * w)
        norm = values.new_zeros(b, 1, h * w)
        weights = (
            (1.0 - wx) * (1.0 - wy),
            wx * (1.0 - wy),
            (1.0 - wx) * wy,
            wx * wy,
        )
        for dx, dy, weight in zip((0, 1, 0, 1), (0, 0, 1, 1), weights):
            xi = (x0 + dx).clamp(0, w - 1)
            yi = (y0 + dy).clamp(0, h - 1)
            index = (yi * w + xi).unsqueeze(1)
            out.scatter_add_(
                2, index.expand(-1, c, -1),
                values * weight.unsqueeze(1))
            norm.scatter_add_(2, index, weight.unsqueeze(1))
        out = out / norm.clamp_min(1.0e-6)
        return out.view(b, c, h, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        tokens, coords = self._sample_curves(x)
        seed = self.seed_embed[:, None, :].expand(
            -1, self.points_per_curve, -1).reshape(1, -1, self.in_channels)
        point = self.point_embed[None, :, :].expand(
            self.num_curves, -1, -1).reshape(1, -1, self.in_channels)
        tokens = tokens + seed + point

        attn, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        tokens = self.norm1(tokens + attn)
        tokens = self.norm2(tokens + self.ffn(tokens))

        tokens = tokens.transpose(1, 2).reshape(
            b, self.in_channels, self.num_curves, self.points_per_curve)
        splat = self._splat(tokens, coords, h, w)
        local = self.local(x)
        fused = self.fuse(torch.cat([local, splat], dim=1))
        return x + self.gamma * fused
