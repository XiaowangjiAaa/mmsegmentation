#!/usr/bin/env python
"""Convert an official EfficientViT segmentation checkpoint to MMSeg keys.

The official model stores the segmentation head under ``head.*`` while the
MMSeg port uses ``decode_head.*``.  This script applies the mapping documented
in ``configs/efficientvit/README.md``.

The official Cityscapes checkpoints have 19 output classes.  For a binary
crack model (2 classes) the final classifier is skipped, so the backbone and
the SegHead fusion layers are initialized and ``conv_seg`` stays randomly
initialized.

Example:
    python tools/model_converters/efficientvit_to_mmseg.py \
        --src work_dirs/pretrained/efficientvit_b0_cityscapes.pt \
        --dst work_dirs/pretrained/efficientvit_b0_crack.pth \
        --num-classes 2
"""
import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True, help='official .pt checkpoint')
    parser.add_argument('--dst', required=True, help='output .pth for MMSeg')
    parser.add_argument(
        '--num-classes', type=int, default=2,
        help='target number of classes; 0 keeps the classifier as-is')
    return parser.parse_args()


def load_state_dict(path):
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        checkpoint = checkpoint['state_dict']
    if not isinstance(checkpoint, dict):
        raise TypeError(f'Unsupported checkpoint format: {type(checkpoint)}')
    return checkpoint


def convert(state_dict, num_classes):
    has_final_expand = any(
        key.startswith('head.output_ops.0.op_list.1.')
        for key in state_dict)
    class_prefix = ('head.output_ops.0.op_list.1.'
                    if has_final_expand else 'head.output_ops.0.op_list.0.')
    expand_prefix = 'head.output_ops.0.op_list.0.'

    converted = {}
    skipped_classifier = []
    for key, value in state_dict.items():
        if key.startswith('backbone.'):
            converted[key] = value
        elif key.startswith(('head.input_ops.', 'head.middle.')):
            converted['decode_head.' + key[len('head.'):]] = value
        elif has_final_expand and key.startswith(expand_prefix):
            converted['decode_head.final_expand_conv.' +
                      key[len(expand_prefix):]] = value
        elif key.startswith(class_prefix):
            suffix = key[len(class_prefix):]
            if not suffix.startswith('conv.'):
                continue
            if num_classes and value.shape[0] != num_classes:
                skipped_classifier.append((key, tuple(value.shape)))
                continue
            converted['decode_head.conv_seg.' + suffix[len('conv.'):]] = value
        # Other head buffers (if any) are ignored.

    return converted, has_final_expand, skipped_classifier


def main():
    args = parse_args()
    state_dict = load_state_dict(args.src)
    converted, has_final_expand, skipped = convert(
        state_dict, args.num_classes)

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            state_dict=converted,
            meta=dict(
                source=str(Path(args.src).resolve()),
                num_classes=args.num_classes,
                final_expand=has_final_expand,
                skipped_classifier=skipped)),
        dst)

    print(f'source            : {args.src}')
    print(f'destination       : {dst}')
    print(f'final_expand      : {has_final_expand}')
    print(f'converted tensors : {len(converted)}')
    if skipped:
        print('classifier skipped (class count mismatch):')
        for key, shape in skipped:
            print(f'  {key} {shape}')


if __name__ == '__main__':
    main()
