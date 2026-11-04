"""Subpixel upgrades: residual memory writes, centered reads and stable sampling.

The mechanisms share narrow projections and compressed global states. They do
not construct a spatial all-pairs attention map or use ground truth at inference.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .efficientvit_structure_modules import (SubpixelStructureMemory, positive,
                                             resize_like, tokens)


def memory_state(key, value, weight=None):
    if weight is not None:
        key = key * weight
    return key.transpose(-1, -2) @ value, key.sum(
        1, keepdim=True).transpose(-1, -2)


def memory_read(query, state):
    matrix, mass = state
    return (query @ matrix) / (query @ mass).clamp_min(1e-6)


def phase_tokens(x):
    """B,4,C,H,W -> B,4HW,C, preserving the phase order."""
    b, p, c, h, w = x.shape
    return x.permute(0, 1, 3, 4, 2).reshape(b, p * h * w, c)


def phase_maps(x, shape):
    b, p, c, h, w = shape
    return x.reshape(b, p, h, w, c).permute(0, 1, 4, 2, 3)


def phase_project(projection, x):
    b, p, c, h, w = x.shape
    return projection(x.reshape(b * p, c, h, w)).reshape(b, p, c, h, w)


class SubpixelMemoryV2(SubpixelStructureMemory):
    """Optional structural memory and mean-preserving contrast reconstruction.

    A shared semantic key times sum of zero-mean residual values would vanish.
    Residual-dependent nonlinear keys therefore write signed residual values
    into an additional structure state, with semantic-conditioned soft weights.
    """

    def __init__(self, channels=16, write_structure=False, center_reads=False):
        super().__init__(channels)
        self.write_structure = write_structure
        self.center_reads = center_reads
        if write_structure:
            self.structure_query = nn.Conv2d(channels, channels, 1, bias=False)
            self.structure_key = nn.Conv2d(channels, channels, 1, bias=False)
            self.structure_value = nn.Conv2d(channels, channels, 1, bias=False)
            self.write_confidence = nn.Conv2d(2 * channels, 1, 1)
            self.read_confidence = nn.Conv2d(2 * channels, 1, 1)

    def forward_components(self, coarse, detail):
        mean, residual, size = self.split_phases(detail)
        semantic = self.semantic(
            torch.cat([resize_like(coarse, mean), mean], 1))
        shared_query = self.query(semantic)
        queries = shared_query[:, None] + phase_project(
            self.residual_query, residual)
        key, value = self.key(semantic), self.value(semantic)
        if self.write_structure:
            structural_query = phase_project(self.structure_query, residual)
            structural_key = phase_project(self.structure_key, residual)
            structural_value = phase_project(self.structure_value, residual)
            b, p, c, h, w = residual.shape
            condition = torch.cat(
                [semantic[:, None].expand(-1, p, -1, -1, -1),
                 residual.abs()], 2).reshape(b * p, 2 * c, h, w)
            write_weight = self.write_confidence(condition).reshape(
                b, p, 1, h, w)
            read_weight = self.read_confidence(condition).reshape(
                b, p, 1, h, w)
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            state = memory_state(
                positive(tokens(key.float())), tokens(value.float()))
            semantic_read = phase_maps(
                memory_read(positive(phase_tokens(queries.float())), state),
                residual.shape)
            shared_read = memory_read(
                positive(tokens(shared_query.float())), state)
            shared_read = shared_read.transpose(1, 2).reshape_as(mean)
            geometry_read = torch.zeros_like(semantic_read)
            if self.write_structure:
                # Nonlinear, residual-dependent keys prevent guaranteed
                # cancellation of the zero-sum residual values.
                structural_state = memory_state(
                    positive(phase_tokens(structural_key.float())),
                    phase_tokens(structural_value.float()),
                    0.02 + 0.98 * phase_tokens(write_weight.float()).sigmoid())
                geometry_read = phase_maps(
                    memory_read(
                        positive(phase_tokens(structural_query.float())),
                        structural_state), residual.shape)
                geometry_read = geometry_read * read_weight.float().sigmoid()
            reads = semantic_read + geometry_read
            if self.center_reads:
                contrast = reads - reads.mean(1, keepdim=True)
                phases = shared_read[:, None] + residual.float() + contrast
            else:
                contrast = reads - reads.mean(1, keepdim=True)
                phases = residual.float() + reads
            output = self.merge_phases(phases, size)
        return output, dict(
            residual=residual,
            semantic_read=semantic_read,
            geometry_read=geometry_read,
            contrast=contrast,
            shared_read=shared_read,
            phases=phases)

    def forward(self, coarse, detail):
        output, _ = self.forward_components(coarse, detail)
        return output, {}


class SubpixelWriteMemory(SubpixelMemoryV2):

    def __init__(self, channels=16):
        super().__init__(channels, write_structure=True)


class SubpixelContrastMemory(SubpixelMemoryV2):

    def __init__(self, channels=16):
        super().__init__(channels, center_reads=True)


class SubpixelWriteContrastMemory(SubpixelMemoryV2):

    def __init__(self, channels=16):
        super().__init__(channels, write_structure=True, center_reads=True)


class StableSubpixelMemory(SubpixelStructureMemory):
    """Filtered coarse sampling paired with a retained full-resolution residual.

    Queries are formed directly on the detail grid, removing the fixed four-way
    query grouping. The strided memory remains sampled and is not guaranteed
    shift-equivariant. Filtering does not erase detail: D=up(low)+residual.
    """

    def __init__(self, channels=16):
        super().__init__(channels)
        kernel = torch.tensor([1., 2., 1.])
        kernel = torch.outer(kernel, kernel) / 16
        self.register_buffer(
            'lowpass_kernel', kernel[None, None], persistent=False)

    def split_low_residual(self, detail):
        channels = detail.shape[1]
        # Filtering before decimation, including full border support.
        kernel = self.lowpass_kernel.to(detail).expand(channels, 1, 3, 3)
        low = F.conv2d(
            F.pad(detail, (1, 1, 1, 1), mode='replicate'),
            kernel,
            stride=2,
            groups=channels)
        residual = detail - resize_like(low, detail)
        return low, residual

    def forward_components(self, coarse, detail):
        low, residual = self.split_low_residual(detail)
        semantic = self.semantic(torch.cat([resize_like(coarse, low), low], 1))
        query = resize_like(self.query(semantic),
                            detail) + self.residual_query(residual)
        key, value = self.key(semantic), self.value(semantic)
        with torch.autocast(device_type=coarse.device.type, enabled=False):
            state = memory_state(
                positive(tokens(key.float())), tokens(value.float()))
            context = memory_read(positive(tokens(query.float())), state)
            context = context.transpose(1, 2).reshape_as(detail)
            output = context + residual.float()
        return output, dict(low=low, residual=residual, context=context)

    def forward(self, coarse, detail):
        output, _ = self.forward_components(coarse, detail)
        return output, {}
