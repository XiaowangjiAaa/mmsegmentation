# EfficientViT-B0 attention variants (2026-10-23)

本目录是 2026-10-23 新增的 EfficientViT-B0 attention 实验配置，不改动原
`configs/crack_batch/efficientvit_b0.py`。

| 配置 | block_type | 参数量 | 说明 |
| --- | --- | ---: | --- |
| `efficientvit-b0-coord_crack-512x512.py` | `coord` | 0.7138M | LiteMLA 后加 CoordAttention |
| `efficientvit-b0-hybrid_crack-512x512.py` | `hybrid` | 0.8283M | stage 3 全局 attention + EdgeGate；stage 2 CoordAttention |
| `efficientvit-b0-window_crack-512x512.py` | `window` | 0.8593M | stage 2/3 8×8 window attention；stage 2 Coord；stage 3 EdgeGate |

详细设计、代码结构、实验计划见：

- [docs/20261023_EfficientViT_Attention_Upgrade.md](../../docs/20261023_EfficientViT_Attention_Upgrade.md)

## 运行

```bash
# 三个变体依次跑
python scripts/efficientvit_attn_20261023/run_variants.py --gpus 0

# 单个变体
python tools/train.py \
  configs/efficientvit_attn_20261023/efficientvit-b0-hybrid_crack-512x512.py \
  --work-dir work_dirs/efficientvit_attn_20261023/hybrid
```

默认 fp32；`--amp` 只建议用于 `coord`，`hybrid/window` 的 attention 分支在
AMP 下可能出现 `grad_norm: inf`。
