# Copyright (c) OpenMMLab. All rights reserved.
"""Evaluation metric for binary crack segmentation."""
from typing import Dict, List

import numpy as np

from mmseg.registry import METRICS
from .iou_metric import IoUMetric


@METRICS.register_module()
class CrackIoUMetric(IoUMetric):
    """IoU metric with explicit foreground metrics for crack segmentation.

    In addition to the macro-averaged metrics returned by :class:`IoUMetric`
    (``aAcc``, ``mIoU``, ``mDice``, ``mFscore``, ``mPrecision``, ``mRecall``,
    ``mAcc``), this metric returns the foreground (crack) values as scalars:

    - ``IoU/crack``
    - ``Dice/crack``
    - ``F1/crack``
    - ``Precision/crack``
    - ``Recall/crack``

    This makes the crack metrics visible in the log/JSON scalars and usable
    for ``CheckpointHook(save_best=...)``.  Background/crack per-class values
    are still printed as the usual MMSeg table by the parent class.

    Args:
        foreground_index (int): Class index of the crack class.
            Default: 1.
        **kwargs: Arguments for :class:`IoUMetric`.
    """

    def __init__(self, foreground_index: int = 1, **kwargs) -> None:
        kwargs.setdefault('iou_metrics', ['mIoU', 'mDice', 'mFscore'])
        super().__init__(**kwargs)
        self.foreground_index = foreground_index

    def compute_metrics(self, results: list) -> Dict[str, float]:
        metrics = super().compute_metrics(results)
        if self.format_only:
            return metrics

        # Recompute the per-class areas to expose the foreground metrics.
        totals = [sum(area) for area in zip(*results)]

        def _to_numpy(total):
            if hasattr(total, 'detach'):
                total = total.detach().cpu()
            return np.asarray(total, dtype=np.float64)

        area_intersect, area_union, area_pred, area_label = [
            _to_numpy(total) for total in totals
        ]
        idx = self.foreground_index
        if not 0 <= idx < len(area_intersect):
            raise ValueError(
                f'foreground_index={idx} is out of range for '
                f'{len(area_intersect)} classes')

        with np.errstate(divide='ignore', invalid='ignore'):
            iou = area_intersect[idx] / area_union[idx]
            dice = 2 * area_intersect[idx] / (
                area_pred[idx] + area_label[idx])
            precision = area_intersect[idx] / area_pred[idx]
            recall = area_intersect[idx] / area_label[idx]
            f1 = 2 * precision * recall / (precision + recall)

        values = {
            'IoU': iou,
            'Dice': dice,
            'F1': f1,
            'Precision': precision,
            'Recall': recall
        }
        class_name = 'crack'
        classes = (self.dataset_meta or {}).get('classes')
        if classes is not None and idx < len(classes):
            class_name = classes[idx]
        for name, value in values.items():
            metrics[f'{name}/{class_name}'] = round(
                float(np.nan_to_num(value)) * 100, 2)
        return metrics
