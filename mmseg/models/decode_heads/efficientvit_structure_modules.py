"""Three lightweight cross-scale structure mechanisms.

All inputs are narrow features, not labels. Auxiliary predictions are returned
explicitly, never cached on the module; inference needs no ground truth.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_like(x, reference):
    return F.interpolate(
        x, size=reference.shape[-2:], mode='bilinear', align_corners=False)


def tokens(x):
    return x.flatten(2).transpose(1, 2)


def positive(x):
    return F.elu(x) + 1


def linear_read(q, k, v, weight=None):
    """Normalized cross-resolution read, O((Nq+Nk)*d^2), no Nq*Nk map."""
    if weight is not None:
        k = k * weight
    state = k.transpose(-1, -2) @ v
    mass = k.sum(dim=1, keepdim=True).transpose(-1, -2)
    return (q @ state) / (q @ mass).clamp_min(1e-6)


class StructurePartitionMemory(nn.Module):
    """Detail-conditioned complementary memories and query-specific contrast.

    The two banks cover every token with soft assignments. Each bank has its
    own denominator, so their readouts are not mixed by bank population alone.
    Their semantics are anchored by a training-only occupancy auxiliary loss.
    """

    def __init__(self, channels=16):
        super().__init__()
        self.route = nn.Sequential(
            nn.Conv2d(3 * channels, channels, 1), nn.Hardswish(),
            nn.Conv2d(channels, 1, 1))
        self.query = nn.Conv2d(2 * channels, channels, 1, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)
        self.read_gate = nn.Conv2d(2 * channels, channels, 1)

    def forward(self, coarse, detail):
        up = resize_like(coarse, detail)
        size = coarse.shape[-2:]
        avg = F.adaptive_avg_pool2d(detail, size)
        # A max statistic retains sparse responses lost by averaging.
        peak = F.adaptive_max_pool2d(detail.abs(), size)
        route_logits = self.route(torch.cat([coarse, avg, peak], dim=1))
        query_input = torch.cat([up, detail], dim=1)
        q_raw, k_raw = self.query(query_input), self.key(coarse)
        value = self.value(coarse)
        gate_raw = self.read_gate(query_input)
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            q, k, v = positive(tokens(q_raw.float())), positive(
                tokens(k_raw.float())), tokens(value.float())
            # No hard pruning; both banks retain a small support everywhere.
            r = 0.02 + 0.96 * tokens(route_logits.float()).sigmoid()
            structure = linear_read(q, k, v, r)
            reference = linear_read(q, k, v, 1 - r)
            global_read = linear_read(q, k, v)
            gate = tokens(gate_raw.float()).tanh()
            out = global_read + gate * (structure - reference)
        out = out.transpose(1, 2).reshape_as(detail)
        return detail + out, {'route_logits': route_logits}


# Outgoing edge order; opposite directions are determined geometrically.
DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0),
              (1, 1))


def neighbor(x, dy, dx):
    """At (y,x), return the neighbor (y+dy,x+dx); zero outside the image."""
    h, w = x.shape[-2:]
    padded = F.pad(x, (1, 1, 1, 1))
    return padded[..., 1 + dy:1 + dy + h, 1 + dx:1 + dx + w]


class ReciprocalConnectivity(nn.Module):
    """Propagate only across mutually supported local connections.

    Shared two-step propagation uses a self-loop and normalized edge mass.
    It is a soft graph on the regular grid, not sparse-kernel acceleration.
    """

    def __init__(self, channels=16):
        super().__init__()
        self.links = nn.Sequential(
            nn.Conv2d(2 * channels, channels, 1), nn.Hardswish(),
            nn.Conv2d(channels, 8, 3, padding=1))
        self.semantic = nn.Conv2d(channels, channels, 1, bias=False)

    @staticmethod
    def reciprocal_weights(logits, semantic):
        probability = logits.float().sigmoid()
        sem = F.normalize(semantic.float(), dim=1, eps=1e-6)
        valid = torch.ones_like(probability[:, :1])
        edges = []
        for index, (dy, dx) in enumerate(DIRECTIONS):
            opposite = DIRECTIONS.index((-dy, -dx))
            reciprocal = probability[:, index:index + 1] * neighbor(
                probability[:, opposite:opposite + 1], dy, dx)
            agreement = (sem * neighbor(sem, dy, dx)).sum(1, keepdim=True)
            edges.append(reciprocal * (2 * agreement).sigmoid() *
                         neighbor(valid, dy, dx))
        return torch.cat(edges, dim=1)

    def forward(self, coarse, detail):
        up = resize_like(coarse, detail)
        logits = self.links(torch.cat([up, detail], dim=1))
        semantic = self.semantic(up)
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            weights = self.reciprocal_weights(logits, semantic)
            h = detail.float() + up.float()
            denominator = 1 + weights.sum(1, keepdim=True)
            for _ in range(2):
                message = h
                for index, (dy, dx) in enumerate(DIRECTIONS):
                    message = message + weights[:, index:index + 1] * neighbor(
                        h, dy, dx)
                h = message / denominator
        return detail + h, {'link_logits': logits}


class SubpixelStructureMemory(nn.Module):
    """Preserve four subpositions and let their residuals query global context.

    The mean/residual decomposition is exact for the projected detail feature
    before learned processing. No claim of lossless image encoding or strict
    shift equivariance is made.
    """

    def __init__(self, channels=16):
        super().__init__()
        self.semantic = nn.Conv2d(2 * channels, channels, 1)
        self.query = nn.Conv2d(channels, channels, 1, bias=False)
        self.residual_query = nn.Conv2d(channels, channels, 1, bias=False)
        self.key = nn.Conv2d(channels, channels, 1, bias=False)
        self.value = nn.Conv2d(channels, channels, 1, bias=False)

    @staticmethod
    def split_phases(detail):
        b, c, h, w = detail.shape
        padded = F.pad(detail, (0, w % 2, 0, h % 2), mode='replicate')
        packed = F.pixel_unshuffle(padded, 2)
        phases = packed.reshape(b, c, 4, padded.shape[-2] // 2,
                                padded.shape[-1] // 2).transpose(1, 2)
        mean = phases.mean(dim=1)
        return mean, phases - mean[:, None], (h, w)

    @staticmethod
    def merge_phases(phases, original_size):
        b, _, c, h, w = phases.shape
        packed = phases.transpose(1, 2).reshape(b, 4 * c, h, w)
        return F.pixel_shuffle(packed,
                               2)[..., :original_size[0], :original_size[1]]

    def forward(self, coarse, detail):
        mean, residual, size = self.split_phases(detail)
        semantic = self.semantic(
            torch.cat([resize_like(coarse, mean), mean], dim=1))
        b, _, c, h, w = residual.shape
        residual_query = self.residual_query(residual.reshape(4 * b, c, h, w))
        residual_query = residual_query.reshape(b, 4, c, h, w)
        query = self.query(semantic)[:, None] + residual_query
        key, value = self.key(semantic), self.value(semantic)
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            q = positive(query.float()).permute(0, 1, 3, 4,
                                                2).reshape(b, -1, c)
            k, v = positive(tokens(key.float())), tokens(value.float())
            context = linear_read(q, k, v)
            context = context.reshape(b, 4, h, w, c).permute(0, 1, 4, 2, 3)
            # A direct residual path preserves the distinction among phases.
            phases = context + residual.float()
        return self.merge_phases(phases, size), {}
