# EfficientViT attention experiment scripts (2026-10-23)

本目录是 2026-10-23 新增的 EfficientViT-B0 attention 实验脚本，详细设计见：

- [docs/20261023_EfficientViT_Attention_Upgrade.md](../../docs/20261023_EfficientViT_Attention_Upgrade.md)

## Round 2：全量替换（默认）

把 stage 2/3 的全部 4 个 LiteMLA 替换成新的 token mixer：

- `replace_window`：全部换 8×8 window attention
- `replace_hybrid`：stage 2 window，stage 3 global
- `replace_global`：全部换 global attention

## 脚本

### `run_variants.py`

顺序运行（单卡），默认跑全量替换：

```bash
python scripts/efficientvit_attn_20261023/run_variants.py \
  --gpus 0 --batch-size 12
```

### `batch_train_attn.py`

双卡队列：一张卡一个版本，结束一个补一个，带日志、失败摘要和
`batch_summary.json`：

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0,1 --batch-size 12
```

默认顺序：

```text
GPU0: replace_hybrid
GPU1: replace_window
GPU0 空出后: replace_global
```

默认 fp32；`hybrid/window` 不建议用 `--amp`。

## Round 1：叠加式对照

`coord`、`hybrid`、`window` 仍然保留，但它们是保留 LiteMLA 后额外加
attention，只能作为对照，不是模块替换实验。

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0,1 --variants coord,hybrid,window
```

## 变体列表

```text
baseline
coord
hybrid
window
replace_window
replace_global
replace_hybrid
```
