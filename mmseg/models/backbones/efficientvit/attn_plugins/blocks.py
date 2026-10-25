"""EfficientViT blocks with optional lightweight attention plugins."""
from typing import Optional

import torch
import torch.nn as nn

from ..layers import EfficientViTBlock, IdentityLayer, ResidualBlock
from .feature_mixers import (ASGLALiteMLA, CDGLASGLALiteMLA,
                             CDGLLiteMLA, FovealLiteMLA)
from .modules import CoordAttention, EdgeGate, ResidualGlobalAttention, \
    ResidualWindowAttention


class EfficientViTBlockCoord(EfficientViTBlock):
    """LiteMLA -> coordinate attention -> MBConv."""

    def __init__(self,
                 in_channels: int,
                 dim: int = 32,
                 expand_ratio: float = 4,
                 norm: dict = dict(type='BN'),
                 act_func='hswish',
                 stage_idx: int = 2,
                 coord_reduction: int = 32,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            dim=dim,
            expand_ratio=expand_ratio,
            norm=norm,
            act_func=act_func,
            **kwargs)
        self.coord = CoordAttention(
            in_channels,
            reduction=coord_reduction,
            norm_cfg=norm or dict(type='BN'),
            act_cfg=act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.coord(x)
        x = self.local_module(x)
        return x


class EfficientViTBlockHybrid(EfficientViTBlock):
    """Stage 3: global attention + edge gate; stage 2: coordinate attention."""

    def __init__(self,
                 in_channels: int,
                 dim: int = 32,
                 expand_ratio: float = 4,
                 norm: dict = dict(type='BN'),
                 act_func='hswish',
                 stage_idx: int = 2,
                 global_stage: int = 3,
                 coord_stage: int = 2,
                 use_edge: bool = True,
                 attn_dim: Optional[int] = None,
                 num_heads: int = 4,
                 edge_kernel: int = 7,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            dim=dim,
            expand_ratio=expand_ratio,
            norm=norm,
            act_func=act_func,
            **kwargs)
        self.global_attn = (
            ResidualGlobalAttention(
                in_channels,
                attn_dim=attn_dim,
                num_heads=num_heads) if stage_idx >= global_stage else None)
        self.coord = (
            CoordAttention(
                in_channels,
                norm_cfg=norm or dict(type='BN'),
                act_cfg=act_func) if stage_idx == coord_stage else None)
        self.edge = (
            EdgeGate(
                in_channels,
                kernel_size=edge_kernel,
                norm_cfg=norm or dict(type='BN'),
                act_cfg=act_func) if use_edge and stage_idx >= global_stage
            else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        if self.global_attn is not None:
            x = self.global_attn(x)
        if self.coord is not None:
            x = self.coord(x)
        x = self.local_module(x)
        if self.edge is not None:
            x = self.edge(x)
        return x


class EfficientViTBlockWindow(EfficientViTBlock):
    """LiteMLA -> local window attention -> optional edge gate -> MBConv."""

    def __init__(self,
                 in_channels: int,
                 dim: int = 32,
                 expand_ratio: float = 4,
                 norm: dict = dict(type='BN'),
                 act_func='hswish',
                 stage_idx: int = 2,
                 window_size: int = 8,
                 coord_stage: int = 2,
                 use_edge: bool = True,
                 attn_dim: Optional[int] = None,
                 num_heads: int = 4,
                 edge_kernel: int = 7,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            dim=dim,
            expand_ratio=expand_ratio,
            norm=norm,
            act_func=act_func,
            **kwargs)
        self.window_attn = ResidualWindowAttention(
            in_channels,
            window_size=window_size,
            attn_dim=attn_dim,
            num_heads=num_heads)
        self.coord = (
            CoordAttention(
                in_channels,
                norm_cfg=norm or dict(type='BN'),
                act_cfg=act_func) if stage_idx == coord_stage else None)
        self.edge = (
            EdgeGate(
                in_channels,
                kernel_size=edge_kernel,
                norm_cfg=norm or dict(type='BN'),
                act_cfg=act_func) if use_edge and stage_idx >= 3 else None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.window_attn(x)
        if self.coord is not None:
            x = self.coord(x)
        x = self.local_module(x)
        if self.edge is not None:
            x = self.edge(x)
        return x


class EfficientViTBlockReplace(EfficientViTBlock):
    """Replace LiteMLA in *every* context block with a token mixer.

    Unlike the additive variants, this version removes LiteMLA completely.
    ``mixer_type`` can be ``window``, ``global`` or ``hybrid``; hybrid uses
    window attention at stage 2 and global attention at stage 3.
    """

    def __init__(self,
                 in_channels: int,
                 dim: int = 32,
                 expand_ratio: float = 4,
                 norm: dict = dict(type='BN'),
                 act_func='hswish',
                 stage_idx: int = 2,
                 mixer_type: str = 'window',
                 attn_dim: Optional[int] = None,
                 num_heads: int = 4,
                 window_size: int = 8,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            dim=dim,
            expand_ratio=expand_ratio,
            norm=norm,
            act_func=act_func,
            **kwargs)
        if mixer_type == 'global':
            mixer = ResidualGlobalAttention(
                in_channels,
                attn_dim=attn_dim,
                num_heads=num_heads,
                residual=False)
        elif mixer_type == 'window':
            mixer = ResidualWindowAttention(
                in_channels,
                window_size=window_size,
                attn_dim=attn_dim,
                num_heads=num_heads,
                residual=False)
        elif mixer_type == 'hybrid':
            if stage_idx >= 3:
                mixer = ResidualGlobalAttention(
                    in_channels,
                    attn_dim=attn_dim,
                    num_heads=num_heads,
                    residual=False)
            else:
                mixer = ResidualWindowAttention(
                    in_channels,
                    window_size=window_size,
                    attn_dim=attn_dim,
                    num_heads=num_heads,
                    residual=False)
        else:
            raise ValueError(f'Unknown mixer_type: {mixer_type}')
        # Replace the LiteMLA context module; MBConv local module is unchanged.
        self.context_module = ResidualBlock(mixer, IdentityLayer())


class EfficientViTBlockFeatureMixer(EfficientViTBlock):
    """Replace LiteMLA with AS-GLA or a channel-decoupled global/local mixer."""

    def __init__(self,
                 in_channels: int,
                 dim: int = 32,
                 expand_ratio: float = 4,
                 norm: dict = dict(type='BN'),
                 act_func='hswish',
                 stage_idx: int = 2,
                 mixer_type: str = 'asgla',
                 local_ratio: float = 0.25,
                 strip_kernel: int = 7,
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            dim=dim,
            expand_ratio=expand_ratio,
            norm=norm,
            act_func=act_func,
            **kwargs)
        if mixer_type == 'asgla':
            mixer = ASGLALiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                dim=dim,
                norm=(None, norm),
                act_func=(None, None),
                strip_kernel=strip_kernel)
        elif mixer_type == 'cdgl':
            mixer = CDGLLiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                local_ratio=local_ratio,
                use_asgla=False,
                strip_kernel=strip_kernel,
                norm_cfg=norm,
                act_cfg=act_func,
                dim=dim)
        elif mixer_type == 'cdgl_asgla':
            mixer = CDGLASGLALiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                local_ratio=local_ratio,
                strip_kernel=strip_kernel,
                norm_cfg=norm,
                act_cfg=act_func,
                dim=dim)
        elif mixer_type == 'foveal':
            mixer = FovealLiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                dim=dim,
                norm=(None, norm),
                act_func=(None, None),
                local_kernel=strip_kernel,
                norm_cfg=norm,
                act_cfg=act_func)
        else:
            raise ValueError(f'Unknown feature mixer: {mixer_type}')
        self.context_module = ResidualBlock(mixer, IdentityLayer())


def build_efficientvit_block(block_type: str = 'v1', **kwargs):
    """Build a standard or plugin EfficientViT block.

    ``stage_idx`` is consumed by the plugin classes and is ignored by the
    original ``EfficientViTBlock``.
    """
    stage_idx = kwargs.pop('stage_idx', 2)
    if block_type in (None, 'v1', 'baseline'):
        return EfficientViTBlock(**kwargs)
    if block_type == 'coord':
        return EfficientViTBlockCoord(stage_idx=stage_idx, **kwargs)
    if block_type == 'hybrid':
        return EfficientViTBlockHybrid(stage_idx=stage_idx, **kwargs)
    if block_type == 'window':
        return EfficientViTBlockWindow(stage_idx=stage_idx, **kwargs)
    if block_type == 'replace_window':
        return EfficientViTBlockReplace(
            stage_idx=stage_idx, mixer_type='window', **kwargs)
    if block_type == 'replace_global':
        return EfficientViTBlockReplace(
            stage_idx=stage_idx, mixer_type='global', **kwargs)
    if block_type == 'replace_hybrid':
        return EfficientViTBlockReplace(
            stage_idx=stage_idx, mixer_type='hybrid', **kwargs)
    if block_type in ('asgla', 'cdgl', 'cdgl_asgla', 'foveal'):
        return EfficientViTBlockFeatureMixer(
            stage_idx=stage_idx, mixer_type=block_type, **kwargs)
    raise ValueError(
        f'Unknown EfficientViT block_type {block_type!r}; '
        "expected one of 'v1', 'coord', 'hybrid', 'window', "
        "'replace_window', 'replace_global', 'replace_hybrid', "
        "'asgla', 'cdgl', 'cdgl_asgla', 'foveal'")
