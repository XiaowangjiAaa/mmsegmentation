"""Orientation-Adaptive Linear Attention (OA-LA).

OA-LA keeps LiteMLA's qkv / multi-scale aggregation / projection, but adds a
structure-tensor orientation field and four oriented strip aggregators. The
oriented branch is zero-gated, so a pretrained LiteMLA checkpoint behaves like
the original model at initialisation.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import LiteMLA


class _OrientedBranch(nn.Module):
    """Depthwise oriented aggregation + grouped 1x1 projection."""

    def __init__(self,
                 channels: int,
                 groups: int,
                 kernel_size: int = 7,
                 angle: int = 0):
        super().__init__()
        self.angle = angle
        if angle in (0, 90):
            kernel = (1, kernel_size) if angle == 0 else (kernel_size, 1)
            padding = (0, kernel_size // 2) if angle == 0 else (
                kernel_size // 2, 0)
            self.depth = nn.Conv2d(
                channels,
                channels,
                kernel,
                padding=padding,
                groups=channels,
                bias=False)
            self.mask = None
        else:
            self.depth = nn.Conv2d(
                channels,
                channels,
                kernel_size,
                padding=kernel_size // 2,
                groups=channels,
                bias=False)
            mask = torch.zeros(1, 1, kernel_size, kernel_size)
            for i in range(kernel_size):
                if angle == 45:
                    mask[0, 0, i, i] = 1.0
                else:
                    mask[0, 0, i, kernel_size - 1 - i] = 1.0
            self.register_buffer('mask', mask, persistent=False)
        self.point = nn.Conv2d(
            channels, channels, 1, groups=groups, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask is None:
            y = self.depth(x)
        else:
            weight = (self.depth.weight * self.mask).to(x.dtype)
            y = F.conv2d(
                x,
                weight,
                bias=None,
                stride=1,
                padding=self.depth.padding,
                groups=self.depth.groups)
        return self.point(y)


class OALALiteMLA(LiteMLA):
    """LiteMLA with a learned-free structure-tensor orientation prior."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 strip_kernel: int = 7,
                 orient_temp: float = 0.1,
                 **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         **kwargs)
        total_dim = self.qkv.conv.out_channels // 3
        heads = max(1, total_dim // self.dim)
        self.strip_kernel = strip_kernel
        self.orient_temp = float(orient_temp)
        self.branches = nn.ModuleList([
            _OrientedBranch(
                3 * total_dim,
                3 * heads,
                kernel_size=strip_kernel,
                angle=angle) for angle in (0, 45, 90, 135)
        ])
        self.gamma = nn.Parameter(torch.zeros(1))

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        self.register_buffer(
            'sobel_x', sobel_x.view(1, 1, 3, 3), persistent=False)
        self.register_buffer(
            'sobel_y', sobel_y.view(1, 1, 3, 3), persistent=False)

    @torch.no_grad()
    def orientation_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Return (B, 4, H, W) orientation weights from a structure tensor."""
        dtype = x.dtype
        gray = x.mean(dim=1, keepdim=True).float()
        dx = F.conv2d(gray, self.sobel_x, padding=1)
        dy = F.conv2d(gray, self.sobel_y, padding=1)
        jxx = F.avg_pool2d(dx * dx, 7, stride=1, padding=3)
        jyy = F.avg_pool2d(dy * dy, 7, stride=1, padding=3)
        jxy = F.avg_pool2d(dx * dy, 7, stride=1, padding=3)
        denom = jxx + jyy + 1.0e-6
        coherence = torch.sqrt((jxx - jyy)**2 + 4.0 * jxy**2) / denom
        coherence = coherence.clamp(0.0, 1.0)
        # Structure-tensor major axis is normal to an edge; use its tangent.
        theta = 0.5 * torch.atan2(2.0 * jxy, jxx - jyy) + math.pi / 2.0
        angles = torch.tensor(
            [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0],
            device=x.device,
            dtype=torch.float32).view(1, 4, 1, 1)
        scores = torch.cos(2.0 * (theta.unsqueeze(1) - angles))
        scores = scores * coherence.unsqueeze(1) / self.orient_temp
        # (B, 1, 4, H, W) -> (B, 4, H, W)
        scores = scores.squeeze(1)
        return torch.softmax(scores, dim=1).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        if len(multi_scale_qkv) > 1:
            weights = self.orientation_weights(x)
            oriented = None
            for index, branch in enumerate(self.branches):
                term = weights[:, index:index + 1] * branch(qkv)
                oriented = term if oriented is None else oriented + term
            oriented = oriented.to(qkv.dtype)
            multi_scale_qkv[1] = (
                multi_scale_qkv[1] + self.gamma.to(qkv.dtype) * oriented)
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)
        out = self.relu_linear_att(multi_scale_qkv)
        return self.proj(out)
