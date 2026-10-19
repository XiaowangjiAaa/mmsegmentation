#!/usr/bin/env python
"""Verify the MMSegmentation + EfficientViT environment.

Run from the repository root (the setup scripts do this automatically):

    python tools/verify_env.py

The script checks the Python/torch/MMCV/MMEngine/MMSeg versions, runs a small
EfficientViT forward pass, builds one experiment config and, when ``dataset/``
is present, builds the crack train/val/test datasets.
"""
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _version_ok(version, minimum, maximum=None):
    from packaging.version import parse
    version = parse(version)
    if version < parse(minimum):
        return False
    if maximum is not None and version >= parse(maximum):
        return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--skip-config', action='store_true', help='skip config build check')
    parser.add_argument(
        '--skip-dataset', action='store_true', help='skip dataset check')
    return parser.parse_args()


def main():
    args = parse_args()
    os.chdir(REPO_ROOT)
    # Make sure the checkout (and not a stale site-packages copy) is verified.
    sys.path.insert(0, str(REPO_ROOT))

    import numpy
    import setuptools
    import torch
    import mmcv
    import mmengine
    import mmseg

    print(f'python     : {sys.version.split()[0]}')
    print(f'numpy      : {numpy.__version__}')
    print(f'torch      : {torch.__version__}')
    print(f'mmcv       : {mmcv.__version__}')
    print(f'mmengine   : {mmengine.__version__}')
    print(f'setuptools : {setuptools.__version__}')
    print(f'mmseg      : {mmseg.__version__} ({Path(mmseg.__file__).parent})')
    try:
        import pkg_resources  # noqa: F401
    except ImportError:
        raise SystemExit(
            'setuptools>=81 removed pkg_resources, but '
            'torch.utils.cpp_extension still imports it. '
            'Run: pip install "setuptools<81"')
    print(f'cuda avail : {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'cuda       : {torch.version.cuda} | '
              f'{torch.cuda.get_device_name(0)}')
    try:
        import mmdet
        import mmpretrain
        print(f'mmdet      : {mmdet.__version__}')
        print(f'mmpretrain : {mmpretrain.__version__}')
    except ImportError:
        print('mmdet/mmpretrain : missing '
              '(needed by Mask2Former / ConvNeXt batch configs)')

    assert sys.version_info[:2] in ((3, 10), (3, 11)), (
        'Python 3.10 or 3.11 is required for the pinned MMCV wheels, '
        f'got {sys.version.split()[0]}')
    assert _version_ok(mmcv.__version__, '2.0.0rc4', '2.2.0'), \
        f'incompatible mmcv {mmcv.__version__}, need >=2.0.0rc4,<2.2.0'
    assert _version_ok(mmengine.__version__, '0.7.1', '1.0.0'), \
        f'incompatible mmengine {mmengine.__version__}, need >=0.7.1,<1.0.0'
    assert int(numpy.__version__.split('.')[0]) == 1, \
        ('numpy 2.x is not compatible with mmcv 2.1.0; '
         'install numpy==1.26.4')

    import mmseg.datasets  # noqa: F401
    import mmseg.models  # noqa: F401
    from mmseg.models.backbones import EfficientViT
    from mmseg.models.decode_heads import EfficientViTHead

    backbone = EfficientViT(arch='b0').eval()
    head = EfficientViTHead(
        in_channels=[128, 64, 32],
        head_width=32,
        head_depth=1,
        final_expand=4,
        num_classes=2,
        out_channels=2,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            avg_non_ignore=True,
            loss_weight=1.0)).eval()
    with torch.no_grad():
        logits = head(backbone(torch.randn(1, 3, 64, 64)))
    assert tuple(logits.shape) == (1, 2, 8, 8), tuple(logits.shape)
    print('EfficientViT forward : OK', tuple(logits.shape))

    if args.skip_config:
        print('config build        : skipped')
        print('\nENVIRONMENT_OK')
        return

    from mmengine.config import Config
    from mmengine.registry import init_default_scope
    from mmseg.registry import DATASETS, METRICS, MODELS

    config_path = ('configs/efficientvit/'
                   'efficientvit-b0_4xb3-30k_crack-512x512.py')
    cfg = Config.fromfile(config_path)
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    model = MODELS.build(cfg.model)
    metric = METRICS.build(cfg.val_evaluator)
    print('config build         : OK', config_path)
    print('evaluator            : OK', type(metric).__name__)

    data_root = Path(cfg.train_dataloader.dataset.data_root)
    if (args.skip_dataset or not (data_root / 'Image').is_dir()
            or not (data_root / 'Mask').is_dir()):
        print('dataset check        : skipped (dataset/Image or Mask missing)')
    else:
        for split in ('train', 'val', 'test'):
            dataset_cfg = (cfg.train_dataloader.dataset
                           if split == 'train' else
                           cfg.val_dataloader.dataset).copy()
            dataset_cfg['split'] = split
            if split != 'train':
                dataset_cfg['pipeline'] = \
                    cfg.val_dataloader.dataset.pipeline
                dataset_cfg['test_mode'] = True
            dataset = DATASETS.build(dataset_cfg)
            print(f'dataset {split:<11}: {len(dataset)} samples')
            assert len(dataset) > 0

    del model
    print('\nENVIRONMENT_OK')


if __name__ == '__main__':
    main()
