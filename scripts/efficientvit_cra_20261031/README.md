# EfficientViT CRA v2 experiment scripts (2026-10-31)

默认 variants：

```text
baseline, cra, cra_dense, cra_long, cra_soft
```

## 双卡批量

```bash
python scripts/efficientvit_cra_20261031/batch_train_cra.py \
  --gpus 0,1 --scratch --batch-size 12
```

执行顺序：

```text
GPU0: baseline
GPU1: cra
GPU0 空出后: cra_dense
GPU1 空出后: cra_long
最后: cra_soft
```

## 只跑三个覆盖改进版

```bash
python scripts/efficientvit_cra_20261031/batch_train_cra.py \
  --gpus 0 --variants cra_dense,cra_long,cra_soft --scratch --batch-size 12
```

## 输出

```text
work_dirs/efficientvit_cra_20261031/batch/
├── batch_summary.json
├── logs/
├── baseline/
├── cra/
├── cra_dense/
├── cra_long/
└── cra_soft/
```

默认 fp32；CRA 不使用 AMP。
