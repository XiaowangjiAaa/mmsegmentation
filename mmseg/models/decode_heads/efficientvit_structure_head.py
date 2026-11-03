"""Cross-scale structure modules on the pretrained EfficientViT fusion head."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
from .efficientvit_head import EfficientViTHead
from .efficientvit_structure_modules import (DIRECTIONS,
                                             ReciprocalConnectivity,
                                             StructurePartitionMemory,
                                             SubpixelStructureMemory, neighbor,
                                             resize_like)


def occupancy_targets(gt, size, ignore_index=255, touched=False):
    """Training-only structure targets, excluding invalid source regions."""
    valid = (gt != ignore_index).float()
    foreground = ((gt == 1) & (gt != ignore_index)).float()
    valid_fraction = F.adaptive_avg_pool2d(valid, size)
    if touched:
        target = F.adaptive_max_pool2d(foreground, size)
        # For edge supervision, a partially padded cell is excluded entirely.
        mask = (valid_fraction > 1 - 1e-6).float()
    else:
        target = F.adaptive_avg_pool2d(foreground,
                                       size) / valid_fraction.clamp_min(1e-6)
        mask = (valid_fraction > 0).float()
    return target, mask


def balanced_binary_loss(logits, target, valid):
    """Separate foreground/background averages; finite for absent classes."""
    logits = logits.float()
    pos, neg = target * valid, (1 - target) * valid
    pos_count, neg_count = pos.sum(), neg.sum()
    positive_loss = (F.softplus(-logits) * pos).sum() / pos_count.clamp_min(1)
    negative_loss = (F.softplus(logits) * neg).sum() / neg_count.clamp_min(1)
    present = (pos_count > 0).float() + (neg_count > 0).float()
    return (positive_loss + negative_loss) / present.clamp_min(1)


@MODELS.register_module()
class EfficientViTStructureHead(EfficientViTHead):
    """Preserve pretrained fusion; replace its detail path with one mechanism.

    The original main classifier provides a dense semantic path. A narrow
    cross-scale module returns stride-4 features and a learned logit update.
    Partition/link supervision is training-only and explicitly included in loss.
    All three modes use the same width and output resolution.
    """

    def __init__(self,
                 structure_mode='partition',
                 structure_width=16,
                 structure_in_channels=16,
                 structure_in_index=0,
                 structure_aux_weight=0.1,
                 **kwargs):
        if kwargs.get('detail_channels', 0) or kwargs.get(
                'detail_upsample_only', False):
            raise ValueError(
                'Use one structure module without the old detail path')
        if structure_width < 1 or structure_aux_weight < 0:
            raise ValueError(
                'width must be positive and auxiliary weight non-negative')
        super().__init__(**kwargs)
        if self.num_classes != 2 or self.out_channels != 2:
            raise ValueError(
                'Structure supervision currently requires binary 2-logit segmentation'
            )
        modules = dict(
            partition=StructurePartitionMemory,
            connectivity=ReciprocalConnectivity,
            subpixel=SubpixelStructureMemory)
        if structure_mode not in modules:
            raise ValueError(f'Unknown structure mode: {structure_mode}')
        self.structure_mode = structure_mode
        self.structure_in_index = structure_in_index
        self.structure_aux_weight = structure_aux_weight
        self.coarse_projection = nn.Sequential(
            nn.Conv2d(self.channels, structure_width, 1, bias=False),
            nn.BatchNorm2d(structure_width), nn.Hardswish())
        self.structure_projection = nn.Sequential(
            nn.Conv2d(structure_in_channels, structure_width, 1, bias=False),
            nn.BatchNorm2d(structure_width), nn.Hardswish())
        self.structure_module = modules[structure_mode](structure_width)
        self.structure_classifier = nn.Conv2d(structure_width,
                                              self.out_channels, 1)
        # Small nonzero scale lets routing, messages and readout learn from
        # the first step; no zero-gate/symmetric-bank dead initialization.
        self.structure_scale = nn.Parameter(torch.tensor(0.01))

    def forward_with_aux(self, inputs):
        coarse_feature = self.forward_features(inputs)
        main_logits = self.cls_seg(coarse_feature)
        detail = self.structure_projection(inputs[self.structure_in_index])
        coarse = self.coarse_projection(coarse_feature)
        structure, aux = self.structure_module(coarse, detail)
        correction = self.structure_classifier(structure.to(detail.dtype))
        logits = resize_like(main_logits, detail)
        return logits + self.structure_scale.tanh() * correction, aux

    def forward(self, inputs):
        logits, _ = self.forward_with_aux(inputs)
        return logits

    def loss(self, inputs, batch_data_samples, train_cfg):
        logits, aux = self.forward_with_aux(inputs)
        losses = self.loss_by_feat(logits, batch_data_samples)
        if not aux or self.structure_aux_weight == 0:
            return losses
        gt = self._stack_batch_gt(batch_data_samples)
        if 'route_logits' in aux:
            route = aux['route_logits']
            target, valid = occupancy_targets(gt, route.shape[-2:],
                                              self.ignore_index)
            losses[
                'loss_structure_route'] = self.structure_aux_weight * balanced_binary_loss(
                    route, target, valid)
        if 'link_logits' in aux:
            link = aux['link_logits']
            target, valid = occupancy_targets(
                gt, link.shape[-2:], self.ignore_index, touched=True)
            edge_targets, edge_valid = [], []
            for dy, dx in DIRECTIONS:
                edge_targets.append(target * neighbor(target, dy, dx))
                edge_valid.append(valid * neighbor(valid, dy, dx))
            losses[
                'loss_structure_links'] = self.structure_aux_weight * balanced_binary_loss(
                    link, torch.cat(edge_targets, dim=1),
                    torch.cat(edge_valid, dim=1))
        return losses
