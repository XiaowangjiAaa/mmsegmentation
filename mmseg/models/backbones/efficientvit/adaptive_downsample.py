"""Group-wise local structure routing replaces a strided depthwise kernel."""
import torch
from torch import nn
from torch.nn import functional as F

from .layers import MBConv


class LocalStructureAggregation(nn.Module):
    """Content-dependent convex aggregation over valid 3x3, stride-2 grids.

    Each channel group shares a different spatial distribution. No offsets,
    global features, labels or additional residual branch are used. Unfold
    materializes local patches; linear complexity does not imply low latency.
    """

    def __init__(self, channels, groups=4, width=8, mode='structure'):
        super().__init__()
        if mode not in ('structure', 'content', 'static', 'uniform'):
            raise ValueError(f'Unknown aggregation mode: {mode}')
        if groups < 1 or channels % groups or width < 1:
            raise ValueError(
                'groups must divide channels, and width must be positive')
        self.channels, self.groups, self.mode = channels, groups, mode
        if mode in ('structure', 'content'):
            self.descriptor = nn.Conv2d(channels, width, 1, bias=False)
            self.router = nn.Sequential(
                nn.Conv2d(3 * width, width, 1),
                nn.Hardswish(),
                # A per-group bias shared by all neighbors cancels in softmax.
                nn.Conv2d(width, groups, 1, bias=False))
        if mode != 'uniform':
            self.position_bias = nn.Parameter(torch.zeros(groups, 9))

    def routing_weights(self, x):
        b, _, h, w = x.shape
        ho, wo = (h + 1) // 2, (w + 1) // 2
        length = ho * wo
        valid = F.unfold(
            x.new_ones(1, 1, h, w), 3, padding=1,
            stride=2).reshape(1, 1, 9, length).bool()
        if self.mode in ('structure', 'content'):
            descriptor = self.descriptor(x)
            neighbors = F.unfold(descriptor, 3, padding=1, stride=2)
            neighbors = neighbors.reshape(b, descriptor.shape[1], 9, length)
            center = descriptor[..., ::2, ::2].flatten(2)[:, :, None]
            center = center.expand(-1, -1, 9, -1)
            if self.mode == 'structure':
                difference = neighbors - center
                condition = torch.cat(
                    [center, difference, difference.abs()], 1)
            else:
                # Capacity-matched content control; no explicit differences.
                condition = torch.cat([center, neighbors, center * neighbors],
                                      1)
            logits = self.router(condition) + self.position_bias[None, :, :,
                                                                 None]
        elif self.mode == 'static':
            logits = self.position_bias[None, :, :,
                                        None].expand(b, -1, -1, length)
        else:
            logits = x.new_zeros(b, self.groups, 9, length)
        # Exclude padded cells instead of treating zeros as real image values.
        weights = logits.float().masked_fill(~valid, float('-inf')).softmax(2)
        return weights, valid

    def aggregate(self, x, weights):
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(
                'Input channels do not match the aggregation operator')
        ho, wo = (h + 1) // 2, (w + 1) // 2
        patches = F.unfold(x, 3, padding=1, stride=2)
        patches = patches.reshape(b, self.groups, c // self.groups, 9, ho * wo)
        output = (patches.float() * weights[:, :, None]).sum(3)
        return output.reshape(b, c, ho, wo).to(x.dtype)

    def forward_components(self, x):
        weights, valid = self.routing_weights(x)
        return self.aggregate(x, weights), dict(weights=weights, valid=valid)

    def forward(self, x):
        return self.forward_components(x)[0]


class AdaptiveDownsampleMBConv(MBConv):
    """Retain expansion/projection; replace the spatial aggregation itself."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 adaptive_mode='structure',
                 adaptive_groups=4,
                 adaptive_width=8,
                 **kwargs):
        if kwargs.get('stride') != 2 or kwargs.get('kernel_size', 3) != 3:
            raise ValueError(
                'Adaptive downsampling requires kernel=3, stride=2')
        super().__init__(in_channels, out_channels, **kwargs)
        channels = self.depth_conv.conv.in_channels
        self.depth_conv.conv = LocalStructureAggregation(
            channels,
            groups=adaptive_groups,
            width=adaptive_width,
            mode=adaptive_mode)
