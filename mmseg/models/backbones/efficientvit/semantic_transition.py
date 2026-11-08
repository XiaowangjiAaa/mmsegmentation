"""Semantic-conditioned structure aggregation at a B-series stage transition.

This operator replaces the strided MBConv and consumes the next block's
LiteMLA. It is not a correction branch around the old downsampling operator.
"""
import torch
from torch import nn
from torch.nn import functional as F

from .layers import ConvLayer, IdentityLayer, LiteMLA, ResidualBlock


def phase_components(x):
    """Mean and signed horizontal/vertical/diagonal differences, scaled by 1/4.

    The four components exactly reconstruct the replicate-padded input before
    learned projection. The subsequent channel compression is not lossless.
    """
    h, w = x.shape[-2:]
    x = F.pad(x, (0, w % 2, 0, h % 2), mode='replicate')
    a, b = x[..., 0::2, 0::2], x[..., 0::2, 1::2]
    c, d = x[..., 1::2, 0::2], x[..., 1::2, 1::2]
    return ((a + b + c + d) / 4, (a - b + c - d) / 4, (a + b - c - d) / 4,
            (a - b - c + d) / 4)


class SemanticStructureTransition(nn.Module):
    """Build coarse tokens using global semantics and signed structure.

    ``semantic`` gates see mean, contextualized coarse features and structural
    magnitudes. ``local`` removes the context input, ``fixed`` uses constant
    0.525 weights, and ``pooled`` removes all detail components. These are
    mechanism controls, not additional claimed innovations.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 dim=16,
                 structure_width=8,
                 gate_mode='semantic',
                 norm=dict(type='BN'),
                 act_func='hswish'):
        super().__init__()
        if gate_mode not in ('semantic', 'local', 'fixed', 'pooled'):
            raise ValueError(f'Unknown transition gate mode: {gate_mode}')
        if structure_width < 1:
            raise ValueError('structure_width must be positive')
        self.gate_mode = gate_mode
        self.structure_width = structure_width
        self.mean_projection = ConvLayer(
            in_channels, out_channels, 1, norm=norm, act_func=act_func)
        self.context_module = ResidualBlock(
            LiteMLA(out_channels, out_channels, dim=dim, norm=(None, norm)),
            IdentityLayer())
        if gate_mode != 'pooled':
            self.structure_projection = nn.Conv2d(
                in_channels, structure_width, 1, bias=False)
            if gate_mode in ('semantic', 'local'):
                gate_channels = 4 * structure_width
                if gate_mode == 'semantic':
                    self.context_projection = nn.Conv2d(
                        out_channels, structure_width, 1, bias=False)
                    gate_channels += structure_width
                self.router = nn.Conv2d(gate_channels, 3 * structure_width, 1)
        fusion_channels = out_channels + (0 if gate_mode == 'pooled' else 3 *
                                          structure_width)
        self.fusion = ConvLayer(
            fusion_channels, out_channels, 1, norm=norm, act_func=None)

    def aggregate(self, mean, differences, context):
        """Expose conditional aggregation for diagnostics without mutable caches."""
        if self.gate_mode == 'pooled':
            return self.fusion(context), {}
        structure = torch.stack(
            [self.structure_projection(d) for d in differences], 1)
        if self.gate_mode == 'fixed':
            weights = torch.full_like(structure, 0.525)
        else:
            condition = [
                self.structure_projection(mean),
                structure.abs().flatten(1, 2)
            ]
            if self.gate_mode == 'semantic':
                condition.append(self.context_projection(context))
            logits = self.router(torch.cat(condition, 1))
            # A fixed positive floor avoids hard removal of weak components.
            weights = (0.05 + 0.95 * logits.sigmoid()).reshape_as(structure)
        weighted = (weights * structure).flatten(1, 2)
        output = self.fusion(torch.cat([context, weighted], 1))
        return output, dict(structure=structure, weights=weights)

    def forward_components(self, x):
        mean, dh, dv, dd = phase_components(x)
        coarse = self.mean_projection(mean)
        context = self.context_module(coarse)
        output, details = self.aggregate(mean, (dh, dv, dd), context)
        return output, dict(mean=mean, context=context, **details)

    def forward(self, x):
        return self.forward_components(x)[0]
