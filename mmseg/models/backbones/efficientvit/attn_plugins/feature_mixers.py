"""Feature-extractor mixers for the EfficientViT attention study.

This module contains three *backbone-level* variants:

* ``ASGLALiteMLA``: LiteMLA with anisotropic strip aggregation.
* ``CDGLLiteMLA``: channel-decoupled global/local linear attention.
* ``CDGLASGLALiteMLA``: CDGL whose global branch is AS-GLA.

All variants keep the same input/output contract ``(B, C, H, W)`` as LiteMLA
and are intended to replace *every* LiteMLA in the B0 backbone.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers import LiteMLA, build_act, build_norm


class ASGLALiteMLA(LiteMLA):
    """LiteMLA + horizontal/vertical strip aggregation.

    The original LiteMLA ``qkv``, ``aggreg`` and ``proj`` keep their parameter
    names, so a converted EfficientViT checkpoint can still load them. The new
    strip branches are zero-gated and start as no-ops.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 strip_kernel: int = 7,
                 **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         **kwargs)
        total_dim = self.qkv.conv.out_channels // 3
        heads = max(1, total_dim // self.dim)
        self.strip_kernel = strip_kernel
        self.strip_h = nn.Sequential(
            nn.Conv2d(
                3 * total_dim,
                3 * total_dim,
                (1, strip_kernel),
                padding=(0, strip_kernel // 2),
                groups=3 * total_dim,
                bias=False),
            nn.Conv2d(
                3 * total_dim,
                3 * total_dim,
                1,
                groups=3 * heads,
                bias=False))
        self.strip_w = nn.Sequential(
            nn.Conv2d(
                3 * total_dim,
                3 * total_dim,
                (strip_kernel, 1),
                padding=(strip_kernel // 2, 0),
                groups=3 * total_dim,
                bias=False),
            nn.Conv2d(
                3 * total_dim,
                3 * total_dim,
                1,
                groups=3 * heads,
                bias=False))
        self.gamma_h = nn.Parameter(torch.zeros(1))
        self.gamma_w = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        if len(multi_scale_qkv) > 1:
            strip = (self.gamma_h * self.strip_h(qkv) +
                     self.gamma_w * self.strip_w(qkv))
            multi_scale_qkv[1] = multi_scale_qkv[1] + strip
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)
        out = self.relu_linear_att(multi_scale_qkv)
        return self.proj(out)


class CDGLLiteMLA(nn.Module):
    """Channel-decoupled global/local token mixer.

    A fraction of channels goes through LiteMLA (or AS-GLA), the remaining
    channels go through horizontal/vertical depthwise strip convolutions. The
    two paths are concatenated and fused with a 1x1 conv.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 local_ratio: float = 0.25,
                 use_asgla: bool = False,
                 strip_kernel: int = 7,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 **kwargs):
        super().__init__()
        norm_cfg = norm_cfg or dict(type='BN')
        act_cfg = act_cfg or dict(type='Hardswish')
        local_channels = max(8, int(round(in_channels * local_ratio)))
        local_channels = min(local_channels, in_channels - 8)
        global_channels = in_channels - local_channels
        self.local_channels = local_channels
        self.global_channels = global_channels

        global_cls = ASGLALiteMLA if use_asgla else LiteMLA
        global_kwargs = dict(kwargs)
        if use_asgla:
            global_kwargs['strip_kernel'] = strip_kernel
        self.global_branch = global_cls(
            in_channels=global_channels,
            out_channels=global_channels,
            **global_kwargs)

        self.local_branch = nn.Sequential(
            nn.Conv2d(
                local_channels,
                local_channels,
                (1, strip_kernel),
                padding=(0, strip_kernel // 2),
                groups=local_channels,
                bias=False),
            build_norm(norm_cfg, local_channels),
            build_act(act_cfg),
            nn.Conv2d(
                local_channels,
                local_channels,
                (strip_kernel, 1),
                padding=(strip_kernel // 2, 0),
                groups=local_channels,
                bias=False),
            build_norm(norm_cfg, local_channels),
            build_act(act_cfg),
        )
        self.fuse = nn.Conv2d(in_channels, out_channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_global, x_local = torch.split(
            x, [self.global_channels, self.local_channels], dim=1)
        y_global = self.global_branch(x_global)
        y_local = self.local_branch(x_local)
        return self.fuse(torch.cat([y_global, y_local], dim=1))


class CDGLASGLALiteMLA(CDGLLiteMLA):
    """CDGL with AS-GLA in the global branch."""

    def __init__(self, *args, **kwargs):
        kwargs['use_asgla'] = True
        super().__init__(*args, **kwargs)


class FovealLiteMLA(LiteMLA):
    """Foveal-peripheral feature mixer.

    The original LiteMLA runs on a 2x downsampled peripheral view; a
    full-resolution strip-convolution foveal branch is added with a
    zero-initialised scale. This keeps the pretrained global path and adds
    high-resolution detail without full-resolution attention.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 local_kernel: int = 7,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 **kwargs):
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         **kwargs)
        norm_cfg = norm_cfg or dict(type='BN')
        act_cfg = act_cfg or dict(type='Hardswish')
        self.local = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                (1, local_kernel),
                padding=(0, local_kernel // 2),
                groups=in_channels,
                bias=False),
            build_norm(norm_cfg, in_channels),
            build_act(act_cfg),
            nn.Conv2d(
                in_channels,
                in_channels,
                (local_kernel, 1),
                padding=(local_kernel // 2, 0),
                groups=in_channels,
                bias=False),
            build_norm(norm_cfg, in_channels),
            build_act(act_cfg))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        peripheral = F.avg_pool2d(x, 2)
        y_global = super().forward(peripheral)
        y_global = F.interpolate(
            y_global, size=size, mode='bilinear', align_corners=False)
        y_local = self.local(x)
        return y_global + self.gamma * y_local
