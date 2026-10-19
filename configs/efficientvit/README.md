# EfficientViT for semantic segmentation

This folder integrates [EfficientViT](https://github.com/mit-han-lab/efficientvit)
(and the segmentation-oriented fork
[CVHub520/efficientvit](https://github.com/CVHub520/efficientvit)) into
MMSegmentation 1.x.

## Models

| Backbone | Strides | Channels | Params (backbone) |
| --- | --- | --- | --- |
| `b0` | 4/8/16/32 | 16/32/64/128 | 0.68M |
| `b1` | 4/8/16/32 | 32/64/128/256 | 4.64M |
| `b2` | 4/8/16/32 | 48/96/192/384 | 14.98M |
| `b3` | 4/8/16/32 | 64/128/256/512 | 39.00M |
| `l1` | 4/8/16/32 | 64/128/256/512 | 38.04M |
| `l2` | 4/8/16/32 | 64/128/256/512 | 49.09M |

Two registered components are provided:

- ``EfficientViT`` (``mmseg/models/backbones/efficientvit``): the encoder.
  It returns four feature maps at strides 4, 8, 16 and 32 and can be combined
  with the usual MMSeg heads (FCN, PSPNet, UPerNet, SegformerHead, ...).
- ``EfficientViTHead`` (``mmseg/models/decode_heads/efficientvit_head.py``):
  the official EfficientViT segmentation head.  It fuses the stride-32,
  stride-16 and stride-8 features with 1x1 convolutions and bicubic
  upsampling, then applies MBConv/FusedMBConv blocks and a classifier.  Use
  ``in_index=(3, 2, 1)`` so that ``in_channels`` is ordered
  ``[C4, C3, C2]`` (stride 32, 16, 8), which is the reference parameter order.

The port is numerically equivalent to the reference implementation: loading
the reference ``state_dict`` after renaming the head parameters reproduces its
outputs (max absolute difference < 1e-5 on CPU).

## Training

Cityscapes (1024x1024) and ADE20K (512x512):

```bash
# single node, 8 GPUs
bash tools/dist_train.sh configs/efficientvit/efficientvit-b1_8xb2-160k_ade20k-512x512.py 8

# single GPU
python tools/train.py configs/efficientvit/efficientvit-b0_8xb2-160k_cityscapes-1024x1024.py
```

The 160k-iteration recipe uses AdamW (lr 1e-3, weight decay 0.01),
1500-iteration warmup and polynomial decay, with CE + Dice loss.  These are
MMSeg-side adaptations; the official training code is not public, so results
can differ slightly from the reference numbers (for example EfficientViT-B0
75.7 mIoU, B1 80.5, B2 82.1, B3 83.0, L1 82.7 and L2 83.2 on Cityscapes
1024x2048 validation).

## Crack segmentation

The repository also ships a binary crack-segmentation setup
(``configs/_base_/datasets/crack.py`` and ``CrackDataset``).  The dataset is
expected as:

```text
dataset/
├── Image/            # crack images (flat or nested), .jpg
└── Mask/             # binary masks, .png, 0 = background, 255 = crack
```

``CrackDataset`` reproduces the deterministic 7:2:1 split of the reference
crack pipeline (sort by stem, shuffle with seed 42) and maps mask value 255 to
the foreground class id 1.  No split files are required.  Masks are decoded as
grayscale (``LoadAnnotations(imdecode_flag='grayscale')``), so RGB-encoded
grayscale PNGs are supported as well.

```bash
# EfficientViT-B0, 512x512, 30k iterations, batch size 3 per GPU
bash tools/dist_train.sh configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py 4
```

The crack configs follow the reference recipe: AdamW, lr 4e-4, weight decay
0.01, gradient clipping 1.0, 1500 warmup iterations, poly schedule and
CE + Dice loss.  Validation reports macro ``mIoU``, ``mDice``, ``mFscore``,
``mPrecision``, ``mRecall`` plus foreground ``IoU/crack``, ``Dice/crack``,
``F1/crack``, ``Precision/crack`` and ``Recall/crack`` via
``CrackIoUMetric``.

## Loading reference checkpoints

The backbone parameter names are identical to the reference implementation.
The official segmentation checkpoints store the head under ``head.*``; rename
them as follows before loading:

```text
backbone.*                          -> backbone.*
head.input_ops.*                    -> decode_head.input_ops.*
head.middle.*                       -> decode_head.middle.*
head.output_ops.0.op_list.0.*       -> decode_head.final_expand_conv.*   (if final_expand is not None)
head.output_ops.0.op_list.1.*       -> decode_head.conv_seg.*            (if final_expand is not None)
head.output_ops.0.op_list.0.conv.*  -> decode_head.conv_seg.*            (if final_expand is None)
```

Then use the converted file as ``load_from`` (or as ``backbone.init_cfg``
after stripping the ``backbone.`` prefix).

The converter is included in the repository:

```bash
python tools/model_converters/efficientvit_to_mmseg.py \
  --src work_dirs/pretrained/efficientvit_b0_cityscapes.pt \
  --dst work_dirs/pretrained/efficientvit_b0_crack.pth \
  --num-classes 2
```

For the official EfficientViT-B0 Cityscapes checkpoint, this one command
downloads and converts it (the 19-class classifier is skipped for a 2-class
crack model):

```bash
python scripts/prepare_efficientvit_b0.py
```
