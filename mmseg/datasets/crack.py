# Copyright (c) OpenMMLab. All rights reserved.
"""Dataset for binary crack segmentation."""
import os.path as osp
import random

from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class CrackDataset(BaseSegDataset):
    """Binary crack segmentation dataset.

    Images and masks are paired by their file stem, e.g.
    ``Image/xxx.jpg`` and ``Mask/xxx.png``.  Masks are expected to be
    grayscale binary images where ``0`` is background and either ``1`` or
    ``255`` is crack; ``255`` is mapped to the foreground class id ``1`` while
    loading annotations, so it is not mistaken for the MMSeg ``ignore_index``.

    When ``ann_file`` is empty (the default), the dataset reproduces the
    deterministic 7:2:1 split of the EfficientViT crack pipeline: samples are
    sorted by stem, shuffled with ``seed`` and assigned to val / test / train.
    Pass ``split='train' | 'val' | 'test'`` to select a subset.  If
    ``ann_file`` is given, it is used as an explicit list of image stems and
    ``split`` is ignored.

    Args:
        split (str): Which subset to use when ``ann_file`` is empty.
            Default: ``'train'``.
        val_ratio (float): Validation ratio. Default: 0.2.
        test_ratio (float): Test ratio. Default: 0.1.
        seed (int): Random seed of the deterministic split. Default: 42.
        **kwargs: Other arguments for :class:`BaseSegDataset`.
    """

    METAINFO = dict(
        classes=('background', 'crack'),
        palette=[[0, 0, 0], [255, 255, 255]])

    def __init__(self,
                 split: str = 'train',
                 val_ratio: float = 0.2,
                 test_ratio: float = 0.1,
                 seed: int = 42,
                 **kwargs):
        assert split in ('train', 'val', 'test'), \
            f'split should be train/val/test, but got {split}'
        assert 0 < val_ratio < 1 and 0 < test_ratio < 1 \
            and val_ratio + test_ratio < 1, \
            'val_ratio/test_ratio should be positive and sum to less than 1'
        self.split = split
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed
        super().__init__(**kwargs)

    def _split_data_list(self, data_list):
        """Deterministically split all samples into train/val/test."""
        records = sorted(
            data_list,
            key=lambda info: osp.splitext(info['img_path'].replace(
                '\\', '/'))[0])
        random.Random(self.seed).shuffle(records)
        num_val = max(1, int(len(records) * self.val_ratio))
        num_test = max(1, int(len(records) * self.test_ratio))
        if num_val + num_test >= len(records):
            raise ValueError('val/test ratios leave no training samples.')
        groups = dict(
            val=records[:num_val],
            test=records[num_val:num_val + num_test],
            train=records[num_val + num_test:])
        return groups[self.split]

    def load_data_list(self):
        data_list = super().load_data_list()
        if not self.ann_file:
            data_list = self._split_data_list(data_list)
        for data_info in data_list:
            # ``LoadAnnotations`` applies this mapping after reading the
            # mask: 255 (white/crack) -> 1, while 0 (background) is kept.
            data_info['label_map'] = {255: 1}
            data_info['reduce_zero_label'] = False
        return data_list
