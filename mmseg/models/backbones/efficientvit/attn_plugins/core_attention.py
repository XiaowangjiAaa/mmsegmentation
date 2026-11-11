"""Experimental replacements of LiteMLA's normalized global read operator.

QKV, multi-scale aggregation, projection and the surrounding MBConv retain
their original layouts. No spatial all-pairs attention matrix is constructed.
"""
import torch
from torch import nn
from torch.nn import functional as F

from ..layers import EfficientViTBlock, IdentityLayer, LiteMLA, ResidualBlock


def bank_read(q, k, v, routes, eps=1e-15):
    """Shared structure routes factor a positive pairwise relation kernel.

    q/k/v: B,heads,N,dim; routes: B,N,banks. One joint normalization is used
    across banks, not a sum of separately normalized memory outputs.
    """
    memory = torch.einsum('bhnd,bnr,bhne->bhrde', k, routes, v)
    mass = torch.einsum('bhnd,bnr->bhrd', k, routes)
    numerator = torch.einsum('bhnd,bhrde,bnr->bhne', q, memory, routes)
    denominator = torch.einsum('bhnd,bhrd,bnr->bhn', q, mass, routes)[...,
                                                                      None]
    return numerator / (denominator + eps), dict(memory=memory, mass=mass)


def local_global_statistics(q, k, v, height, width):
    """Separate valid 3x3 neighbors and their exact global complement."""
    b, heads, n, dim = q.shape
    if height * width != n:
        raise ValueError('Token count must match height*width')

    def neighborhoods(tokens):
        maps = tokens.transpose(-1, -2).reshape(b, heads * dim, height, width)
        patches = F.unfold(maps, 3, padding=1)
        return patches.reshape(b, heads, dim, 9, n).permute(0, 1, 4, 3, 2)

    valid = F.unfold(q.new_ones(1, 1, height, width), 3, padding=1)
    valid = valid.reshape(1, 1, 9, n).transpose(-1, -2)
    local_k, local_v = neighborhoods(k), neighborhoods(v)
    scores = (q[..., None, :] * local_k).sum(-1) * valid
    local_num = (scores[..., None] * local_v).sum(-2)
    local_mass = scores.sum(-1, keepdim=True)
    total_num = q @ (k.transpose(-1, -2) @ v)
    total_mass = q @ k.sum(-2, keepdim=True).transpose(-1, -2)
    local_count = valid.sum(-1, keepdim=True)
    global_count = n - local_count
    # Empty complements occur on very small grids; force exact zero instead
    # of retaining floating-point subtraction noise.
    global_num = torch.where(global_count > 0, total_num - local_num, 0.)
    global_mass = torch.where(global_count > 0,
                              (total_mass - local_mass).clamp_min(0), 0.)
    return dict(
        local_num=local_num,
        local_mass=local_mass,
        global_num=global_num,
        global_mass=global_mass,
        local_count=local_count,
        global_count=global_count)


def joint_read(stats, global_weight, eps=1e-15, postfusion=False):
    """Count-balanced joint normalization, or a separate-output control."""
    ln = stats['local_num'] / stats['local_count']
    lm = stats['local_mass'] / stats['local_count']
    gn = stats['global_num'] / stats['global_count'].clamp_min(1)
    gm = stats['global_mass'] / stats['global_count'].clamp_min(1)
    if postfusion:
        fraction = global_weight / (1 + global_weight)
        fraction = torch.where(stats['global_count'] > 0, fraction, 0.)
        return (1 - fraction) * ln / (lm + eps) + fraction * gn / (gm + eps)
    return (ln + global_weight * gn) / (lm + global_weight * gm + eps)


class CoreLiteMLA(LiteMLA):
    """Replace memory construction/readout without an output correction path."""

    MODES = ('bank_structure', 'bank_content', 'bank_uniform', 'joint',
             'joint_fixed', 'postfusion')

    def __init__(self,
                 in_channels,
                 out_channels,
                 mode='bank_structure',
                 routing_width=8,
                 num_banks=2,
                 **kwargs):
        super().__init__(in_channels, out_channels, **kwargs)
        if mode not in self.MODES or routing_width < 1 or num_banks < 1:
            raise ValueError(
                'Unknown core attention mode or invalid routing dimensions')
        self.mode, self.num_banks = mode, num_banks
        self.packed_heads = self.proj.conv.in_channels // self.dim
        if mode not in ('bank_uniform', 'joint_fixed'):
            self.routing_descriptor = nn.Conv2d(
                in_channels, routing_width, 1, bias=False)
            outputs = num_banks if mode.startswith(
                'bank_') else self.packed_heads
            self.routing_logits = nn.Conv2d(3 * routing_width, outputs, 1)

    def routing_features(self, x):
        descriptor = self.routing_descriptor(x)
        mean = F.avg_pool2d(
            F.pad(descriptor, (1, 1, 1, 1), mode='replicate'), 3, stride=1)
        if self.mode == 'bank_content':
            return torch.cat([descriptor, mean, descriptor * mean], 1)
        difference = descriptor - mean
        return torch.cat([descriptor, difference, difference.abs()], 1)

    def forward_components(self, x):
        qkv = self.qkv(x)
        packed = torch.cat([qkv] + [op(qkv) for op in self.aggreg], 1)
        b, _, h, w = packed.shape
        tokens = packed.reshape(b, -1, 3 * self.dim, h * w).transpose(-1, -2)
        q_raw, k_raw, v = tokens.split(self.dim, -1)
        logits = None if self.mode in ('bank_uniform',
                                       'joint_fixed') else self.routing_logits(
                                           self.routing_features(x))
        # The compressed statistics and local/global subtraction accumulate in
        # float32 under either CPU/CUDA mixed precision.
        with torch.autocast(device_type=x.device.type, enabled=False):
            q, k, v = self.kernel_func(q_raw.float()), self.kernel_func(
                k_raw.float()), v.float()
            if self.mode.startswith('bank_'):
                routes = (
                    x.new_full((b, h * w, self.num_banks), 1 /
                               self.num_banks).float() if logits is None else
                    logits.float().flatten(2).transpose(1, 2).softmax(-1))
                output, details = bank_read(q, k, v, routes, self.eps)
                details['routes'] = routes
            else:
                weight = (
                    q.new_ones(b, self.packed_heads, h * w, 1) if logits
                    is None else F.softplus(logits.float()).flatten(2)[...,
                                                                       None])
                details = local_global_statistics(q, k, v, h, w)
                output = joint_read(details, weight, self.eps,
                                    self.mode == 'postfusion')
                details['global_weight'] = weight
                lm = details['local_mass'] / details['local_count']
                gm = details['global_mass'] / details[
                    'global_count'].clamp_min(1)
                details['effective_global_fraction'] = (
                    torch.where(details['global_count'] > 0, weight /
                                (1 + weight), 0.) if self.mode == 'postfusion'
                    else weight * gm / (lm + weight * gm + self.eps))
        maps = output.transpose(-1, -2).reshape(b, -1, h, w)
        return self.proj(maps.to(packed.dtype)), details

    def forward(self, x):
        return self.forward_components(x)[0]


class EfficientViTBlockCore(EfficientViTBlock):

    def __init__(self,
                 in_channels,
                 dim=32,
                 expand_ratio=4,
                 norm=dict(type='BN'),
                 act_func='hswish',
                 mode='bank_structure',
                 routing_width=8,
                 num_banks=2,
                 **kwargs):
        super().__init__(
            in_channels,
            dim=dim,
            expand_ratio=expand_ratio,
            norm=norm,
            act_func=act_func,
            **kwargs)
        original = self.context_module.main
        # CPU model construction keeps the original parameter initialization
        # and subsequent RNG sequence identical to baseline under the same seed.
        # New routing weights are still random, not zero-gated.
        with torch.random.fork_rng(devices=[]):
            mixer = CoreLiteMLA(
                in_channels,
                in_channels,
                dim=dim,
                norm=(None, norm),
                heads_ratio=kwargs.get('heads_ratio', 1.0),
                mode=mode,
                routing_width=routing_width,
                num_banks=num_banks)
        mixer.load_state_dict(original.state_dict(), strict=False)
        self.context_module = ResidualBlock(mixer, IdentityLayer())
