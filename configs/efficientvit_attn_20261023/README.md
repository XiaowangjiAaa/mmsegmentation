# EfficientViT-B0 attention variants (2026-10-23)

本目录是 2026-10-23 新增的 EfficientViT-B0 attention 实验配置，不改动原
`configs/crack_batch/efficientvit_b0.py`。

## Round 2：全量替换（推荐做正式对比）

把 **stage 2/3 中全部 4 个 LiteMLA** 都替换成新的 token mixer，MBConv 保持
不变。原 B0 只有 stage 2/3 有 LiteMLA，stage 0/1 没有 attention，因此
“替换相似作用的部分”就是替换这 4 个 LiteMLA，而不是只加在一个 stage。

| 配置 | block_type | 参数量 | FLOPs | 说明 |
| --- | --- | ---: | ---: | --- |
| `efficientvit-b0-replace-window_crack-512x512.py` | `replace_window` | 0.5519M | 0.514G | 全部 LiteMLA → 8×8 window attention |
| `efficientvit-b0-replace-hybrid_crack-512x512.py` | `replace_hybrid` | 0.5545M | 0.527G | stage 2 window；stage 3 global |
| `efficientvit-b0-replace-global_crack-512x512.py` | `replace_global` | 0.5558M | 0.780G | 全部 LiteMLA → global attention |

baseline：`0.7045M / 0.564G`。

## Round 1：残差式增强（保留作对照）

| 配置 | block_type | 参数量 | FLOPs | 说明 |
| --- | --- | ---: | ---: | --- |
| `efficientvit-b0-coord_crack-512x512.py` | `coord` | 0.7138M | 0.564G | LiteMLA 后加 CoordAttention |
| `efficientvit-b0-hybrid_crack-512x512.py` | `hybrid` | 0.8283M | 0.611G | stage 3 全局 attention + EdgeGate；stage 2 CoordAttention |
| `efficientvit-b0-window_crack-512x512.py` | `window` | 0.8593M | 0.649G | stage 2/3 window attention；stage 2 Coord；stage 3 EdgeGate |

Round 1 的 40k 结果是：coord 61.65、hybrid 62.10、window 61.74，均未超过
baseline 62.28（IoU/crack）。这是“LiteMLA 保留 + 叠加 attention”的结论，
不能代表“全量替换 LiteMLA”的结论。

详细设计、代码结构、实验计划见：

- [docs/20261023_EfficientViT_Attention_Upgrade.md](../../docs/20261023_EfficientViT_Attention_Upgrade.md)

## 运行

```bash
# 默认跑 Round 2 三个全量替换版本
python scripts/efficientvit_attn_20261023/run_variants.py --gpus 0

# 双卡队列
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0,1 \
  --variants replace_hybrid,replace_window,replace_global \
  --batch-size 12

# 单个变体
python tools/train.py \
  configs/efficientvit_attn_20261023/efficientvit-b0-replace-hybrid_crack-512x512.py \
  --work-dir work_dirs/efficientvit_attn_20261023/replace_hybrid
```

默认 fp32；`--amp` 只建议用于 `coord`，`hybrid/window` 的 attention 分支在
AMP 下可能出现 `grad_norm: inf`。
