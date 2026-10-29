# EfficientViT CRA experiment scripts (2026-10-30)

本目录是 Curve-Routed Attention 的批量训练脚本。默认 variants：

```text
baseline, cra
```

## 双卡

```bash
python scripts/efficientvit_cra_20261030/batch_train_cra.py \
  --gpus 0,1 --batch-size 12
```

执行顺序：

```text
GPU0: baseline
GPU1: cra
```

## 只跑 CRA

```bash
python scripts/efficientvit_cra_20261030/batch_train_cra.py \
  --gpus 0 --variants cra --batch-size 12
```

## from-scratch 对照

CRA 没有预训练 attention，建议同时跑：

```bash
python scripts/efficientvit_cra_20261030/batch_train_cra.py \
  --gpus 0,1 --variants baseline,cra --scratch --batch-size 12
```

## 输出

```text
work_dirs/efficientvit_cra_20261030/batch/
├── batch_summary.json
├── logs/
├── baseline/best_mIoU_iter_*.pth
└── cra/best_mIoU_iter_*.pth
```

默认 fp32；CRA 不使用 AMP。
