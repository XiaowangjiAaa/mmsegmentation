# Copyright (c) EfficientViT authors. All rights reserved.
"""EfficientViT backbones for semantic segmentation.

Ported from the official implementation
(https://github.com/mit-han-lab/efficientvit) and the segmentation fork
(https://github.com/CVHub520/efficientvit).

The backbone returns four feature maps with strides 4, 8, 16 and 32, which
makes it a drop-in replacement for the encoders used by the MMSeg decode
heads.  For the official EfficientViT segmentation head use
``EfficientViTHead`` with ``in_index=(3, 2, 1)``.
"""
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmseg.registry import MODELS
from .attn_plugins import build_efficientvit_block
from .layers import (ConvLayer, DSConv, FusedMBConv,
                     IdentityLayer, MBConv, OpSequential, ResBlock,
                     ResidualBlock)

__all__ = ['EfficientViT']


@MODELS.register_module()
class EfficientViT(BaseModule):
    """EfficientViT backbone.

    Args:
        arch (str): Model variant, one of ``b0``, ``b1``, ``b2``, ``b3``,
            ``l0``, ``l1`` and ``l2``.  The ``b`` series uses Hardswish and
            BatchNorm; the ``l`` series uses GELU and a smaller batch-norm
            epsilon, following the reference implementation.
        in_channels (int): Number of input channels. Default: 3.
        out_indices (Sequence[int]): Output from which stages. The four
            stages correspond to strides 4, 8, 16 and 32.
            Default: ``(0, 1, 2, 3)``.
        norm_cfg (dict): Config of normalization layers.
            Default: ``dict(type='BN')``.
        act_cfg (dict): Config of activation layers. If not specified,
            Hardswish is used for the ``b`` series and GELU for the ``l``
            series.
        init_cfg (dict, optional): Initialization config.
    """

    arch_settings = {
        # name: width_list, depth_list, dim / qkv_dim, is_large
        'b0': dict(
            width_list=[8, 16, 32, 64, 128],
            depth_list=[1, 2, 2, 2, 2],
            dim=16,
            large=False),
        'b1': dict(
            width_list=[16, 32, 64, 128, 256],
            depth_list=[1, 2, 3, 3, 4],
            dim=16,
            large=False),
        'b2': dict(
            width_list=[24, 48, 96, 192, 384],
            depth_list=[1, 3, 4, 4, 6],
            dim=32,
            large=False),
        'b3': dict(
            width_list=[32, 64, 128, 256, 512],
            depth_list=[1, 4, 6, 6, 9],
            dim=32,
            large=False),
        'l0': dict(
            width_list=[32, 64, 128, 256, 512],
            depth_list=[1, 1, 1, 4, 4],
            dim=32,
            large=True),
        'l1': dict(
            width_list=[32, 64, 128, 256, 512],
            depth_list=[1, 1, 1, 6, 6],
            dim=32,
            large=True),
        'l2': dict(
            width_list=[32, 64, 128, 256, 512],
            depth_list=[1, 2, 2, 8, 8],
            dim=32,
            large=True),
    }

    def __init__(self,
                 arch: str = 'b0',
                 in_channels: int = 3,
                 out_indices: Sequence[int] = (0, 1, 2, 3),
                 norm_cfg: Optional[dict] = dict(type='BN'),
                 act_cfg: Optional[dict] = None,
                 block_type: str = 'v1',
                 attn_kwargs: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg=init_cfg)
        assert arch in self.arch_settings, \
            f'Unknown EfficientViT arch {arch!r}. ' \
            f'Supported: {list(self.arch_settings)}'
        assert max(out_indices) < 4 and min(out_indices) >= 0, \
            f'out_indices should be in [0, 3], but got {out_indices}'

        setting = self.arch_settings[arch]
        self.arch = arch
        self.block_type = block_type
        self.attn_kwargs = dict(attn_kwargs or {})
        self.is_large = setting['large']
        self.out_indices = tuple(out_indices)
        self.width_list: List[int] = []

        if norm_cfg is None:
            norm_cfg = dict(type='BN')
        else:
            norm_cfg = dict(norm_cfg)

        if act_cfg is None:
            act_cfg = dict(type='GELU', approximate='tanh') \
                if self.is_large else dict(type='Hardswish')
        else:
            act_cfg = dict(act_cfg)

        if self.is_large:
            # The reference implementation uses eps=1e-7 for the L models.
            norm_cfg.setdefault('eps', 1e-7)
            self._build_large_backbone(
                width_list=setting['width_list'],
                depth_list=setting['depth_list'],
                qkv_dim=setting['dim'],
                in_channels=in_channels,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        else:
            self._build_branch_backbone(
                width_list=setting['width_list'],
                depth_list=setting['depth_list'],
                dim=setting['dim'],
                in_channels=in_channels,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

    def _build_branch_backbone(self, width_list: List[int],
                               depth_list: List[int], dim: int,
                               in_channels: int, norm_cfg: dict,
                               act_cfg: dict) -> None:
        """Build the EfficientViT-B encoder."""
        # input stem
        input_stem = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm_cfg,
                act_func=act_cfg)
        ]
        for _ in range(depth_list[0]):
            block = self._build_local_block(
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm_cfg,
                act_func=act_cfg)
            input_stem.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        self.input_stem = OpSequential(input_stem)
        self.width_list.append(in_channels)

        # stages with local blocks only
        stages = []
        for w, d in zip(width_list[1:3], depth_list[1:3]):
            stage = []
            for i in range(d):
                stride = 2 if i == 0 else 1
                block = self._build_local_block(
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=4,
                    norm=norm_cfg,
                    act_func=act_cfg)
                block = ResidualBlock(
                    block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        # stages with linear-attention blocks
        for stage_idx, (w, d) in enumerate(
                zip(width_list[3:], depth_list[3:]), start=2):
            stage = []
            block = self._build_local_block(
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=4,
                norm=norm_cfg,
                act_func=act_cfg,
                fewer_norm=True)
            stage.append(ResidualBlock(block, None))
            in_channels = w
            for _ in range(d):
                stage.append(
                    build_efficientvit_block(
                        self.block_type,
                        in_channels=in_channels,
                        dim=dim,
                        expand_ratio=4,
                        norm=norm_cfg,
                        act_func=act_cfg,
                        stage_idx=stage_idx,
                        **self.attn_kwargs))
            stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        self.stages = nn.ModuleList(stages)

    def _build_large_backbone(self, width_list: List[int],
                              depth_list: List[int], qkv_dim: int,
                              in_channels: int, norm_cfg: dict,
                              act_cfg: dict) -> None:
        """Build the EfficientViT-L encoder."""
        # stage 0 (stride 2), kept in ``stages`` to match the reference
        # parameter names (``stages.0...``).
        stages = []
        stage0 = [
            ConvLayer(
                in_channels=in_channels,
                out_channels=width_list[0],
                stride=2,
                norm=norm_cfg,
                act_func=act_cfg)
        ]
        for _ in range(depth_list[0]):
            block = self._build_local_block_large(
                stage_id=0,
                in_channels=width_list[0],
                out_channels=width_list[0],
                stride=1,
                expand_ratio=1,
                norm=norm_cfg,
                act_func=act_cfg)
            stage0.append(ResidualBlock(block, IdentityLayer()))
        in_channels = width_list[0]
        stages.append(OpSequential(stage0))
        self.width_list.append(in_channels)

        # stages 1-3 (strides 4, 8, 16)
        for stage_id, (w, d) in enumerate(
                zip(width_list[1:4], depth_list[1:4]), start=1):
            stage = []
            for i in range(d + 1):
                stride = 2 if i == 0 else 1
                block = self._build_local_block_large(
                    stage_id=stage_id,
                    in_channels=in_channels,
                    out_channels=w,
                    stride=stride,
                    expand_ratio=4 if stride == 1 else 16,
                    norm=norm_cfg,
                    act_func=act_cfg,
                    fewer_norm=stage_id > 2)
                block = ResidualBlock(
                    block, IdentityLayer() if stride == 1 else None)
                stage.append(block)
                in_channels = w
            stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        # stage 4 (stride 32), with linear-attention blocks
        for stage_id, (w, d) in enumerate(
                zip(width_list[4:], depth_list[4:]), start=4):
            stage = []
            block = self._build_local_block_large(
                stage_id=stage_id,
                in_channels=in_channels,
                out_channels=w,
                stride=2,
                expand_ratio=24,
                norm=norm_cfg,
                act_func=act_cfg,
                fewer_norm=True)
            stage.append(ResidualBlock(block, None))
            in_channels = w
            for _ in range(d):
                stage.append(
                    build_efficientvit_block(
                        self.block_type,
                        in_channels=in_channels,
                        dim=qkv_dim,
                        expand_ratio=6,
                        norm=norm_cfg,
                        act_func=act_cfg,
                        stage_idx=4,
                        **self.attn_kwargs))
            stages.append(OpSequential(stage))
            self.width_list.append(in_channels)

        self.stages = nn.ModuleList(stages)

    @staticmethod
    def _build_local_block(in_channels: int,
                           out_channels: int,
                           stride: int,
                           expand_ratio: float,
                           norm: dict,
                           act_func: dict,
                           fewer_norm: bool = False) -> nn.Module:
        if expand_ratio == 1:
            block = DSConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None))
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None))
        return block

    @staticmethod
    def _build_local_block_large(stage_id: int,
                                 in_channels: int,
                                 out_channels: int,
                                 stride: int,
                                 expand_ratio: float,
                                 norm: dict,
                                 act_func: dict,
                                 fewer_norm: bool = False) -> nn.Module:
        if expand_ratio == 1:
            block = ResBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None))
        elif stage_id <= 2:
            block = FusedMBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, False) if fewer_norm else False,
                norm=(None, norm) if fewer_norm else norm,
                act_func=(act_func, None))
        else:
            block = MBConv(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                use_bias=(True, True, False) if fewer_norm else False,
                norm=(None, None, norm) if fewer_norm else norm,
                act_func=(act_func, act_func, None))
        return block

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward with four outputs at strides 4, 8, 16 and 32."""
        if self.is_large:
            # ``stages[0]`` produces the stride-2 feature map.
            x = self.stages[0](x)
            stages = self.stages[1:]
        else:
            x = self.input_stem(x)
            stages = self.stages

        outs = []
        for stage in stages:
            x = stage(x)
            outs.append(x)

        return tuple(outs[i] for i in self.out_indices)
