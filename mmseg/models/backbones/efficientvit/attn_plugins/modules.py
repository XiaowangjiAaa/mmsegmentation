"""Lightweight attention plugins for EfficientViT-B0.

The modules here are deliberately small: they are meant to be inserted as
zero-initialised residual gates so that a pretrained EfficientViT checkpoint
still behaves like the original model at step 0.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import build_act, build_norm


class CoordAttention(nn.Module):
    """Coordinate attention with a zero-initialised residual gate.

    Args:
        channels (int): Number of input/output channels.
        reduction (int): Channel reduction ratio for the shared 1x1 conv.
        norm_cfg (dict): Norm config used by the small bottleneck.
        act_cfg (dict | str): Activation config.
        gamma_init (float): Initial value of the residual scale. 0 keeps the
            module as an identity at initialisation.
    """

    def __init__(self,
                 channels: int,
                 reduction: int = 32,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 gamma_init: float = 0.0):
        super().__init__()
        norm_cfg = norm_cfg or dict(type='BN')
        act_cfg = act_cfg or dict(type='Hardswish')
        mid = max(8, channels // reduction)

        self.conv1 = nn.Conv2d(channels, mid, 1, bias=False)
        self.bn1 = build_norm(norm_cfg, mid)
        self.act = build_act(act_cfg)
        self.conv_h = nn.Conv2d(mid, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(mid, channels, 1, bias=False)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        # (B, C, H, 1) and (B, C, W, 1)
        x_h = x.mean(dim=3, keepdim=True)
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)

        y = self.conv1(y)
        if self.bn1 is not None:
            y = self.bn1(y)
        if self.act is not None:
            y = self.act(y)

        y_h, y_w = torch.split(y, [h, w], dim=2)
        a_h = torch.sigmoid(self.conv_h(y_h))
        a_w = torch.sigmoid(self.conv_w(y_w)).permute(0, 1, 3, 2)
        gate = a_h * a_w
        return x * (1.0 + self.gamma * (gate - 1.0))


class EdgeGate(nn.Module):
    """Direction-aware strip-convolution gate.

    Inspired by the DEGConv idea in MixerCSeg (CVPR 2026): horizontal and
    vertical strip convolutions are cheap and match long, thin cracks.
    """

    def __init__(self,
                 channels: int,
                 mid_channels: Optional[int] = None,
                 kernel_size: int = 7,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 gamma_init: float = 0.0):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd')
        norm_cfg = norm_cfg or dict(type='BN')
        act_cfg = act_cfg or dict(type='Hardswish')
        mid = mid_channels or max(8, channels // 2)

        self.in_proj = nn.Conv2d(channels, mid, 1, bias=False)
        self.act = build_act(act_cfg)
        self.conv_h = nn.Conv2d(
            mid,
            mid,
            (1, kernel_size),
            padding=(0, kernel_size // 2),
            groups=mid,
            bias=False)
        self.conv_w = nn.Conv2d(
            mid,
            mid,
            (kernel_size, 1),
            padding=(kernel_size // 2, 0),
            groups=mid,
            bias=False)
        self.out_proj = nn.Conv2d(2 * mid, channels, 1, bias=False)
        self.norm = build_norm(norm_cfg, channels)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.in_proj(x)
        if self.act is not None:
            y = self.act(y)
        y = torch.cat([self.conv_h(y), self.conv_w(y)], dim=1)
        gate = self.out_proj(y)
        if self.norm is not None:
            gate = self.norm(gate)
        gate = torch.sigmoid(gate)
        return x * (1.0 + self.gamma * (gate - 1.0))


class ResidualGlobalAttention(nn.Module):
    """Small global multi-head attention branch with a zero-init residual.

    Only intended for the low-resolution stage (16x16 at 512 input), where a
    256-token full attention is cheap. A depthwise 3x3 conv injects a local
    positional inductive bias before Q/K/V.
    """

    def __init__(self,
                 channels: int,
                 attn_dim: Optional[int] = None,
                 num_heads: int = 4,
                 gamma_init: float = 0.0):
        super().__init__()
        attn_dim = attn_dim or max(32, channels // 2)
        attn_dim = max(num_heads, (attn_dim // num_heads) * num_heads)
        self.channels = channels
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.LayerNorm(channels)
        self.local = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=True)
        self.qkv = nn.Conv2d(channels, 3 * attn_dim, 1, bias=True)
        self.proj = nn.Conv2d(attn_dim, channels, 1, bias=True)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        b, c, h, w = x.shape
        y = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        y = y + self.local(y)

        qkv = self.qkv(y)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b, self.num_heads, self.head_dim, h * w)
        k = k.reshape(b, self.num_heads, self.head_dim, h * w)
        v = v.reshape(b, self.num_heads, self.head_dim, h * w)

        # Full attention on 256 tokens: keep the matmul in fp32 for stability.
        scores = torch.matmul(q.transpose(-1, -2), k) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v.transpose(-1, -2))
        out = out.transpose(-1, -2).reshape(b, self.attn_dim, h, w)
        return (x + self.gamma * self.proj(out)).to(dtype)


class ResidualWindowAttention(nn.Module):
    """Non-overlapping window attention, cheap enough for stage 2/3."""

    def __init__(self,
                 channels: int,
                 window_size: int = 8,
                 attn_dim: Optional[int] = None,
                 num_heads: int = 4,
                 gamma_init: float = 0.0):
        super().__init__()
        attn_dim = attn_dim or max(32, channels // 2)
        attn_dim = max(num_heads, (attn_dim // num_heads) * num_heads)
        self.channels = channels
        self.window_size = window_size
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.scale = self.head_dim**-0.5

        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Conv2d(channels, 3 * attn_dim, 1, bias=True)
        self.proj = nn.Conv2d(attn_dim, channels, 1, bias=True)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        b, c, h, w = x.shape
        y = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        ws = self.window_size
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h or pad_w:
            y = F.pad(y, (0, pad_w, 0, pad_h))
        hp, wp = y.shape[-2:]
        nh, nw = hp // ws, wp // ws

        windows = y.view(b, c, nh, ws, nw, ws).permute(
            0, 2, 4, 1, 3, 5).reshape(b * nh * nw, c, ws, ws)
        qkv = self.qkv(windows)
        q, k, v = qkv.chunk(3, dim=1)
        q = q.reshape(b * nh * nw, self.num_heads, self.head_dim, ws * ws)
        k = k.reshape(b * nh * nw, self.num_heads, self.head_dim, ws * ws)
        v = v.reshape(b * nh * nw, self.num_heads, self.head_dim, ws * ws)

        scores = torch.matmul(q.transpose(-1, -2), k) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v.transpose(-1, -2))
        out = out.transpose(-1, -2).reshape(
            b * nh * nw, self.attn_dim, ws, ws)
        out = out.view(b, nh, nw, self.attn_dim, ws, ws)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            b, self.attn_dim, hp, wp)
        out = out[:, :, :h, :w]
        return (x + self.gamma * self.proj(out)).to(dtype)
