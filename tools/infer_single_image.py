#!/usr/bin/env python
"""Run a trained MMSeg crack model on a single image.

Example:
    python tools/infer_single_image.py \
        --config configs/crack_batch/mask2former_r50.py \
        --checkpoint work_dirs/crack_batch/mask2former_r50/best_mIoU_iter_33000.pth \
        --image "image for test/8.jpg" \
        --out-dir work_dirs/inference/8
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.registry import init_default_scope
import mmseg.datasets  # noqa: F401
import mmseg.evaluation  # noqa: F401
import mmseg.models  # noqa: F401
from mmseg.apis import init_model
from mmseg.registry import TRANSFORMS


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--out-dir', default='work_dirs/inference')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def build_inference_pipeline(cfg):
    pipeline_cfg = []
    for transform in cfg.test_pipeline:
        transform = dict(transform)
        if transform.get('type') == 'LoadAnnotations':
            continue
        pipeline_cfg.append(transform)
    return Compose([TRANSFORMS.build(transform) for transform in pipeline_cfg])


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    custom_imports = cfg.get('custom_imports', None)
    if custom_imports:
        imports = custom_imports.get('imports', [])
        if isinstance(imports, str):
            imports = [imports]
        for module in imports:
            __import__(module)

    init_default_scope('mmseg')
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print('CUDA not available; falling back to CPU.')
        device = 'cpu'

    model = init_model(args.config, args.checkpoint, device=device)
    model.eval()
    pipeline = build_inference_pipeline(cfg)

    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f'Image not found: {image_path}')

    data = pipeline({'img_path': str(image_path)})
    # ``PackSegInputs`` returns a single sample; ``test_step`` expects a batch.
    batch = {
        'inputs': [data['inputs']],
        'data_samples': [data['data_samples']],
    }
    with torch.no_grad():
        result = model.test_step(batch)[0]

    pred = result.pred_sem_seg.data.cpu().numpy().astype(np.uint8)
    if pred.ndim == 3:
        pred = pred[0]
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f'Failed to read image: {image_path}')
    height, width = image.shape[:2]
    if pred.shape[:2] != (height, width):
        pred = cv2.resize(
            pred, (width, height), interpolation=cv2.INTER_NEAREST)

    mask = (pred == 1).astype(np.uint8) * 255
    overlay = image.copy()
    overlay[mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    mask_path = out_dir / f'{stem}_mask.png'
    overlay_path = out_dir / f'{stem}_overlay.png'
    cv2.imwrite(str(mask_path), mask)
    cv2.imwrite(str(overlay_path), overlay)

    crack_pixels = int((mask > 0).sum())
    total_pixels = height * width
    ratio = crack_pixels / max(total_pixels, 1) * 100.0
    print(f'image       : {image_path}')
    print(f'config      : {args.config}')
    print(f'checkpoint  : {args.checkpoint}')
    print(f'device      : {device}')
    print(f'image size  : {width}x{height}')
    print(f'crack pixels: {crack_pixels}')
    print(f'crack ratio : {ratio:.3f}%')
    print(f'mask        : {mask_path.resolve()}')
    print(f'overlay     : {overlay_path.resolve()}')


if __name__ == '__main__':
    main()
