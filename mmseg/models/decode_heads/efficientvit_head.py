# Copyright (c) EfficientViT authors. All rights reserved.
"""EfficientViT segmentation head.

This is a port of the ``SegHead`` used by the official EfficientViT
segmentation models.  It fuses the stride-32, stride-16 and stride-8 features
with 1x1 projections + bicubic upsampling, refines the fused feature with a
few MBConv blocks, and predicts the segmentation logits at stride 8.
"""
from typing import Optional, Sequence

import torch.nn as nn

from mmseg.registry import MODELS
from ..backbones.efficientvit.layers import (ConvLayer, FusedMBConv,
                                             IdentityLayer, MBConv,
                                             OpSequential, ResidualBlock,
                                             UpSampleLayer, resize)
from .decode_head import BaseDecodeHead


@MODELS.register_module()
class EfficientViTHead(BaseDecodeHead):
    """EfficientViT segmentation head (official ``SegHead``).

    Args:
        in_channels (Sequence[int]): Channels of the selected backbone
            features, ordered as ``in_index``.
        head_width (int): Width of the fused feature.
        head_depth (int): Number of MBConv/FusedMBConv blocks after fusion.
        expand_ratio (float): Expansion ratio of the middle blocks.
        middle_op (str): Middle block type, ``'mbconv'`` or ``'fmbconv'``.
        final_expand (int, optional): Expansion ratio of the 1x1 conv before
            the classifier.  ``None`` means no extra expansion.
        in_index (Sequence[int]): Indices of the backbone outputs to fuse.
            The default ``(3, 2, 1)`` selects strides 32, 16 and 8 and matches
            the reference parameter order.
        strides (Sequence[int]): Strides of the selected features.  Used to
            compute the upsampling factors relative to ``head_stride``.
        head_stride (int): Output stride of the head. Default: 8.
        act_cfg (dict): Activation config. Default: Hardswish.
        norm_cfg (dict): Normalization config. Default: ``dict(type='BN')``.
        **kwargs: Other arguments for :class:`BaseDecodeHead`.
    """

    def __init__(self,
                 in_channels: Sequence[int],
                 head_width: int,
                 head_depth: int,
                 expand_ratio: float = 4,
                 middle_op: str = 'mbconv',
                 final_expand: Optional[int] = 4,
                 in_index: Sequence[int] = (3, 2, 1),
                 strides: Sequence[int] = (32, 16, 8),
                 head_stride: int = 8,
                 act_cfg: Optional[dict] = None,
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 dropout_ratio: float = 0.0,
                 init_cfg: Optional[dict] = None,
                 **kwargs):
        assert middle_op in ('mbconv', 'fmbconv'), \
            f'middle_op should be "mbconv" or "fmbconv", got {middle_op}'
        assert len(in_channels) == len(in_index) == len(strides), \
            'in_channels, in_index and strides must have the same length'
        assert head_stride in strides, \
            f'head_stride {head_stride} must be one of strides {strides}'

        self.head_width = head_width
        self.head_depth = head_depth
        self.expand_ratio = expand_ratio
        self.middle_op = middle_op
        self.final_expand = final_expand
        self.strides = tuple(strides)
        self.head_stride = head_stride
        self.act_cfg = act_cfg or dict(type='Hardswish')
        self.norm_cfg = norm_cfg

        # ``final_expand`` is part of the classifier projection, so the
        # channel count of ``conv_seg`` is known before calling super().
        channels = head_width * (final_expand or 1)
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            in_index=in_index,
            input_transform='multiple_select',
            act_cfg=self.act_cfg,
            norm_cfg=self.norm_cfg,
            dropout_ratio=dropout_ratio,
            init_cfg=init_cfg,
            **kwargs)

        # 1x1 projection + upsampling for every selected feature.  The module
        # layout follows the reference implementation so that official
        # checkpoints can be loaded after a key rename.
        self.input_ops = nn.ModuleList()
        for in_ch, stride in zip(in_channels, strides):
            factor = stride // head_stride
            conv = ConvLayer(
                in_ch,
                head_width,
                1,
                norm=self.norm_cfg,
                act_func=None)
            if factor == 1:
                self.input_ops.append(conv)
            else:
                self.input_ops.append(
                    OpSequential([conv, UpSampleLayer(factor=factor)]))

        middle = []
        for _ in range(head_depth):
            if middle_op == 'mbconv':
                block = MBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=self.norm_cfg,
                    act_func=(self.act_cfg, self.act_cfg, None))
            else:
                block = FusedMBConv(
                    head_width,
                    head_width,
                    expand_ratio=expand_ratio,
                    norm=self.norm_cfg,
                    act_func=(self.act_cfg, None))
            middle.append(ResidualBlock(block, IdentityLayer()))
        self.middle = OpSequential(middle)

        if final_expand is not None:
            self.final_expand_conv = ConvLayer(
                head_width,
                head_width * final_expand,
                1,
                norm=self.norm_cfg,
                act_func=self.act_cfg)
        else:
            self.final_expand_conv = None

    def forward(self, inputs):
        """Forward function.

        Args:
            inputs (list[Tensor]): Multi-level features from the backbone.

        Returns:
            Tensor: Segmentation logits with stride ``head_stride``.
        """
        feats = self._transform_inputs(inputs)

        target_size = feats[self.strides.index(self.head_stride)].shape[-2:]
        fused = None
        for op, feat in zip(self.input_ops, feats):
            out = op(feat)
            # The reference implementation uses integer factors.  Resizing to
            # the head stride keeps the output correct for inputs whose size
            # is not a multiple of 32 as well.
            if out.shape[-2:] != target_size:
                out = resize(
                    out, target_size, mode='bicubic', align_corners=False)
            fused = out if fused is None else fused + out

        out = self.middle(fused)
        if self.final_expand_conv is not None:
            out = self.final_expand_conv(out)
        return self.cls_seg(out)
