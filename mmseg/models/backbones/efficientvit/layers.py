# Copyright (c) EfficientViT authors. All rights reserved.
"""Building blocks of EfficientViT.

This module is a PyTorch/MMSeg port of the EfficientViT model code released
by the MIT Han Lab (https://github.com/mit-han-lab/efficientvit) and the
segmentation-oriented fork at https://github.com/CVHub520/efficientvit.

Only the modules needed by the semantic-segmentation backbone and decode head
are kept.  The module structure and the parameter names intentionally follow
the reference implementation, so that official EfficientViT checkpoints can
be converted to MMSeg with a pure key rewrite.
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_activation_layer, build_norm_layer

__all__ = [
    'LayerNorm2d', 'ConvLayer', 'UpSampleLayer', 'IdentityLayer', 'DSConv',
    'MBConv', 'FusedMBConv', 'ResBlock', 'LiteMLA', 'EfficientViTBlock',
    'ResidualBlock', 'OpSequential', 'build_act', 'build_norm',
    'get_same_padding', 'val2tuple', 'resize'
]


def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]
                     ) -> Union[int, Tuple[int, ...]]:
    """Return the padding that keeps the spatial size unchanged."""
    if isinstance(kernel_size, tuple):
        return tuple(get_same_padding(ks) for ks in kernel_size)
    assert kernel_size % 2 > 0, 'kernel size should be odd number'
    return kernel_size // 2


def val2tuple(val, min_len: int = 1) -> list:
    """Broadcast a scalar (or extend a short sequence) to ``min_len`` items."""
    if isinstance(val, (list, tuple)):
        assert len(val) <= min_len
        return list(val) + [val[-1]] * (min_len - len(val))
    return [val] * min_len


def resize(x: torch.Tensor,
           size: Optional[Union[int, Tuple[int, int]]] = None,
           scale_factor: Optional[Union[int, Tuple[int, int]]] = None,
           mode: str = 'bicubic',
           align_corners: bool = False) -> torch.Tensor:
    """Resize helper that mimics ``efficientvit.models.utils.resize``."""
    if mode in {'bilinear', 'bicubic'}:
        return F.interpolate(
            x,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners)
    elif mode in {'nearest', 'area'}:
        return F.interpolate(x, size=size, scale_factor=scale_factor,
                             mode=mode)
    raise NotImplementedError(f'resize(mode={mode}) is not implemented.')


class LayerNorm2d(nn.LayerNorm):
    """Channel-wise LayerNorm used by the EfficientViT-L models."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(
            torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(
                1, -1, 1, 1)
        return out


def build_norm(norm_cfg: Optional[dict],
               num_features: int) -> Optional[nn.Module]:
    """Build a normalization layer from an MMSeg style ``norm_cfg``.

    ``None`` means "no normalization" and returns ``None`` (this is used by
    the reference implementation to build bias-only convolutions).
    """
    if norm_cfg is None:
        return None
    cfg = dict(norm_cfg)
    norm_type = cfg.pop('type')
    key = norm_type.lower()
    if key in ('bn', 'bn2d', 'batchnorm2d'):
        return nn.BatchNorm2d(num_features, **cfg)
    if key in ('syncbn', 'syncbatchnorm'):
        return nn.SyncBatchNorm(num_features, **cfg)
    if key in ('ln2d', 'layernorm2d'):
        return LayerNorm2d(num_features, **cfg)
    # Fall back to MMCV for GroupNorm/LayerNorm and other registered norms.
    return build_norm_layer(dict(type=norm_type, **cfg), num_features)[1]


def build_act(act_cfg: Optional[Union[str, dict]]) -> Optional[nn.Module]:
    """Build an activation layer.

    Both the reference names (``hswish``, ``relu6``, ``gelu`` ...) and the
    MMSeg names (``Hardswish``, ``ReLU6`` ...) are accepted.  ``gelu`` follows
    the reference implementation and uses ``approximate='tanh'``.
    """
    if act_cfg is None:
        return None
    if isinstance(act_cfg, str):
        act_cfg = dict(type=act_cfg)
    cfg = dict(act_cfg)
    act_type = cfg.pop('type')
    key = act_type.lower()
    if key == 'relu':
        return nn.ReLU(**cfg)
    if key == 'relu6':
        return nn.ReLU6(**cfg)
    if key in ('hswish', 'hardswish'):
        return nn.Hardswish(**cfg)
    if key == 'silu':
        return nn.SiLU(**cfg)
    if key == 'gelu':
        cfg.setdefault('approximate', 'tanh')
        return nn.GELU(**cfg)
    return build_activation_layer(dict(type=act_type, **cfg))


class ConvLayer(nn.Module):
    """Conv2d (+ optional dropout, norm and activation)."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 dilation: int = 1,
                 groups: int = 1,
                 use_bias: bool = False,
                 dropout: float = 0,
                 norm: Optional[dict] = None,
                 act_func: Optional[Union[str, dict]] = None):
        super().__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) \
            if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias)
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class UpSampleLayer(nn.Module):
    """Bicubic upsampling used by the SegHead input projections."""

    def __init__(self,
                 mode: str = 'bicubic',
                 size: Optional[Union[int, Tuple[int, int], List[int]]] = None,
                 factor: int = 2,
                 align_corners: bool = False):
        super().__init__()
        self.mode = mode
        self.size = tuple(val2tuple(size, 2)) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.size is not None
                and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class IdentityLayer(nn.Module):
    """Identity shortcut."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class DSConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 use_bias: Union[bool, Tuple[bool, bool]] = False,
                 norm: Union[dict, Tuple[dict, dict]] = (dict(type='BN'),
                                                         dict(type='BN')),
                 act_func: Union[None, str, dict, Tuple] = ('relu6', None)):
        super().__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0])
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class MBConv(nn.Module):
    """Inverted residual block used by the EfficientViT stages."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 mid_channels: Optional[int] = None,
                 expand_ratio: float = 6,
                 use_bias: Union[bool, Tuple[bool, bool, bool]] = False,
                 norm: Union[dict, Tuple] = (dict(type='BN'), dict(type='BN'),
                                             dict(type='BN')),
                 act_func: Union[None, str, dict, Tuple] = ('relu6', 'relu6',
                                                            None)):
        super().__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0])
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1])
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inverted_conv(x)
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class FusedMBConv(nn.Module):
    """Fused MBConv block used by the EfficientViT-L stages."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 mid_channels: Optional[int] = None,
                 expand_ratio: float = 6,
                 groups: int = 1,
                 use_bias: Union[bool, Tuple[bool, bool]] = False,
                 norm: Union[dict, Tuple] = (dict(type='BN'), dict(type='BN')),
                 act_func: Union[None, str, dict, Tuple] = ('relu6', None)):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0])
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial_conv(x)
        x = self.point_conv(x)
        return x


class ResBlock(nn.Module):
    """Two-convolution residual block used by the EfficientViT-L stem."""

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 mid_channels: Optional[int] = None,
                 expand_ratio: float = 1,
                 use_bias: Union[bool, Tuple[bool, bool]] = False,
                 norm: Union[dict, Tuple] = (dict(type='BN'), dict(type='BN')),
                 act_func: Union[None, str, dict, Tuple] = ('relu6', None)):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0])
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention.

    This is the core EfficientViT context module.  Attention is computed with
    a ReLU kernel and a global ``k^T v`` state, which keeps the complexity
    linear in the number of pixels.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 heads: Optional[int] = None,
                 heads_ratio: float = 1.0,
                 dim: int = 8,
                 use_bias: Union[bool, Tuple[bool, bool]] = False,
                 norm: Union[None, dict, Tuple] = (None, dict(type='BN')),
                 act_func: Tuple = (None, None),
                 kernel_func: str = 'relu',
                 scales: Tuple[int, ...] = (5, ),
                 eps: float = 1.0e-15):
        super().__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0])
        self.aggreg = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    3 * total_dim,
                    3 * total_dim,
                    scale,
                    padding=get_same_padding(scale),
                    groups=3 * total_dim,
                    bias=use_bias[0]),
                nn.Conv2d(
                    3 * total_dim,
                    3 * total_dim,
                    1,
                    groups=3 * heads,
                    bias=use_bias[0]))
            for scale in scales
        ])
        self.kernel_func = build_act(kernel_func)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1])

    @torch.cuda.amp.autocast(enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (qkv[..., 0:self.dim], qkv[..., self.dim:2 * self.dim],
                   qkv[..., 2 * self.dim:])

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 1), mode='constant', value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)

        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        out = self.relu_linear_att(multi_scale_qkv)
        out = self.proj(out)

        return out


class EfficientViTBlock(nn.Module):
    """Context (linear attention) + local (MBConv) block."""

    def __init__(self,
                 in_channels: int,
                 heads_ratio: float = 1.0,
                 dim: int = 32,
                 expand_ratio: float = 4,
                 norm: dict = dict(type='BN'),
                 act_func: Union[str, dict] = 'hswish'):
        super().__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
            ),
            IdentityLayer(),
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.context_module(x)
        x = self.local_module(x)
        return x


class ResidualBlock(nn.Module):
    """Generic residual wrapper used throughout EfficientViT."""

    def __init__(self,
                 main: Optional[nn.Module],
                 shortcut: Optional[nn.Module],
                 post_act: Optional[Union[str, dict]] = None,
                 pre_norm: Optional[nn.Module] = None):
        super().__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x)
        return self.main(self.pre_norm(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x)
        else:
            res = self.forward_main(x) + self.shortcut(x)
            if self.post_act is not None:
                res = self.post_act(res)
        return res


class OpSequential(nn.Module):
    """Sequential container that silently skips ``None`` entries."""

    def __init__(self, op_list: List[Optional[nn.Module]]):
        super().__init__()
        self.op_list = nn.ModuleList([op for op in op_list if op is not None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for op in self.op_list:
            x = op(x)
        return x
