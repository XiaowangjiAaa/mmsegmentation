# EfficientViT attention experiment scripts (2026-10-23)

本目录是 2026-10-23 新增的 EfficientViT-B0 attention 实验脚本，详细设计见：

- [docs/20261023_EfficientViT_Attention_Upgrade.md](../../docs/20261023_EfficientViT_Attention_Upgrade.md)

## 脚本

### `run_variants.py`

顺序运行一个或多个变体（单卡）：

```bash
python scripts/efficientvit_attn_20261023/run_variants.py \
  --variants coord,hybrid,window --gpus 0 --batch-size 12
```

### `batch_train_attn.py`

双卡队列：一张卡一个版本，结束一个补一个，带日志、失败摘要和
`batch_summary.json`：

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0,1 --variants coord,hybrid,window --batch-size 12
```

默认 fp32；`hybrid/window` 不建议用 `--amp`。

## 变体

- `baseline`：`configs/crack_batch/efficientvit_b0.py`
- `coord`：`configs/efficientvit_attn_20261023/efficientvit-b0-coord_crack-512x512.py`
- `hybrid`：`configs/efficientvit_attn_20261023/efficientvit-b0-hybrid_crack-512x512.py`
- `window`：`configs/efficientvit_attn_20261023/efficientvit-b0-window_crack-512x512.py`
