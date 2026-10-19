# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmengine.structures import PixelData

from mmseg.evaluation import CrackIoUMetric
from mmseg.structures import SegDataSample


def _data_sample(gt, pred):
    data_sample = SegDataSample()
    data_sample.gt_sem_seg = PixelData(
        data=torch.LongTensor(gt).unsqueeze(0))
    data_sample.pred_sem_seg = PixelData(
        data=torch.LongTensor(pred).unsqueeze(0))
    data_sample.img_path = 'dummy.png'
    return data_sample.to_dict()


def test_crack_iou_metric():
    # gt:   [0, 0]    pred: [0, 0]
    #       [1, 1]          [0, 1]
    gt = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    pred = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    metric = CrackIoUMetric(iou_metrics=['mIoU', 'mDice', 'mFscore'])
    metric.dataset_meta = dict(
        classes=['background', 'crack'], label_map=dict(),
        reduce_zero_label=False)
    metric.process([0], [_data_sample(gt, pred)])
    results = metric.evaluate(1)

    # Foreground: intersection=1, union=2, pred=1, label=2.
    assert results['IoU/crack'] == 50.0
    assert results['Dice/crack'] == 66.67
    assert results['F1/crack'] == 66.67
    assert results['Precision/crack'] == 100.0
    assert results['Recall/crack'] == 50.0

    # Macro metrics include background and crack.
    assert results['mIoU'] == 58.33
    assert results['mDice'] == 73.33
    assert results['mFscore'] == 73.33
    assert results['mPrecision'] == 83.33
    assert results['mRecall'] == 75.0
    assert results['aAcc'] == 75.0
    assert results['mAcc'] == 75.0
