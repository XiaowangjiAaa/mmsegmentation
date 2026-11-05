"""Experimental structure-aware encoder operators for EfficientViT-B.

All additions preserve the original operator parameter names. They introduce
feature-derived routing only; no segmentation labels enter encoder forwards.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .layers import EfficientViTBlock, LiteMLA, MBConv, ResidualBlock, IdentityLayer


def split_phases(x):
    """Replicate-pad odd borders, then return mean and ordered residuals."""
    h, w = x.shape[-2:]
    x = F.pad(x, (0, w % 2, 0, h % 2), mode='replicate')
    b, c, hp, wp = x.shape
    phases = F.pixel_unshuffle(x, 2).reshape(b, c, 4, hp // 2, wp // 2)
    mean = phases.mean(2)
    return mean, phases - mean[:, :, None]


class StructureDownsample(MBConv):
    """Original strided MBConv plus semantic-gated ordered phase compression."""

    def __init__(self, in_channels, out_channels, structure_width=8, **kwargs):
        if kwargs.get('stride') != 2:
            raise ValueError('StructureDownsample requires stride=2')
        super().__init__(in_channels, out_channels, **kwargs)
        width = min(structure_width, in_channels)
        self.structure_reduce = nn.Conv2d(in_channels, width, 1, bias=False)
        # Flatten C then phase: four different weights per channel avoid the
        # guaranteed cancellation of summing zero-mean residuals directly.
        self.structure_encode = nn.Conv2d(
            4 * width, out_channels, 1, bias=False)
        self.structure_gate = nn.Conv2d(2 * width, out_channels, 1)
        self.structure_scale = nn.Parameter(torch.tensor(0.01))

    def structure_update(self, x):
        mean, residual = split_phases(self.structure_reduce(x))
        magnitude = residual.abs().mean(2)
        gate = self.structure_gate(torch.cat([mean, magnitude], 1)).sigmoid()
        encoded = self.structure_encode(residual.flatten(1, 2))
        return gate * encoded

    def forward(self, x):
        return super().forward(
            x) + self.structure_scale.tanh() * self.structure_update(x)


class StructureQueryLiteMLA(LiteMLA):
    """Add a local structure vector to each multi-scale semantic query.

    Keys/values and global state construction retain the original LiteMLA
    definition. The update is vector-valued, not a scalar that cancels in the
    normalized attention ratio.
    """

    def __init__(self, in_channels, out_channels, structure_width=8, **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        width = min(structure_width, in_channels)
        query_channels = self.proj.conv.in_channels
        self.structure_reduce = nn.Conv2d(in_channels, width, 1, bias=False)
        self.structure_query = nn.Conv2d(width, query_channels, 1, bias=False)
        self.structure_scale = nn.Parameter(torch.tensor(0.01))

    def condition_queries(self, qkv, x):
        descriptor = self.structure_reduce(x)
        residual = descriptor - F.avg_pool2d(
            F.pad(descriptor, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
        delta = self.structure_query(residual)
        b, _, h, w = qkv.shape
        packed = qkv.reshape(b, -1, 3, self.dim, h, w)
        q, k, v = packed.unbind(2)
        q = q + self.structure_scale.tanh() * delta.reshape_as(q)
        return torch.stack([q, k, v], 2).reshape_as(qkv)

    def forward(self, x):
        qkv = self.qkv(x)
        multi = torch.cat([qkv] + [op(qkv) for op in self.aggreg], 1)
        return self.proj(
            self.relu_linear_att(self.condition_queries(multi, x)))


class StructureQueryBlock(EfficientViTBlock):

    def __init__(self,
                 in_channels,
                 dim=32,
                 expand_ratio=4,
                 norm=dict(type='BN'),
                 act_func='hswish',
                 structure_width=8,
                 **kwargs):
        super().__init__(
            in_channels,
            dim=dim,
            expand_ratio=expand_ratio,
            norm=norm,
            act_func=act_func,
            **kwargs)
        self.context_module = ResidualBlock(
            StructureQueryLiteMLA(
                in_channels,
                in_channels,
                dim=dim,
                heads_ratio=kwargs.get('heads_ratio', 1.0),
                norm=(None, norm),
                structure_width=structure_width), IdentityLayer())


class ContextGuidedLocalBlock(EfficientViTBlock):
    """Global context conditions spatially varying local feature refinement.

    Preserve the pretrained MBConv and add a narrow mixture of horizontal,
    vertical and 3x3 depthwise operators. Dense soft routing saves no branch
    FLOPs: all three experts are evaluated.
    """

    def __init__(self, in_channels, structure_width=8, **kwargs):
        super().__init__(in_channels, **kwargs)
        width = min(structure_width, in_channels)
        self.structure_reduce = nn.Conv2d(in_channels, width, 1, bias=False)
        self.structure_experts = nn.ModuleList([
            nn.Conv2d(
                width, width, (1, 5), padding=(0, 2), groups=width,
                bias=False),
            nn.Conv2d(
                width, width, (5, 1), padding=(2, 0), groups=width,
                bias=False),
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False)
        ])
        self.structure_router = nn.Conv2d(2 * width, 3, 1)
        self.structure_expand = nn.Conv2d(width, in_channels, 1, bias=False)
        self.structure_scale = nn.Parameter(torch.tensor(0.01))

    def local_update(self, x, context):
        local = self.structure_reduce(x)
        semantic = self.structure_reduce(context)
        weights = self.structure_router(torch.cat([local, semantic],
                                                  1)).softmax(1)
        experts = torch.stack([op(local) for op in self.structure_experts], 1)
        update = self.structure_expand((weights[:, :, None] * experts).sum(1))
        return update, weights

    def forward(self, x):
        contextual = self.context_module(x)
        update, _ = self.local_update(contextual, contextual - x)
        return self.local_module(
            contextual) + self.structure_scale.tanh() * update
