# EfficientViT-B0 attention variants (2026-10-23)

本目录是 EfficientViT-B0 的 attention / feature-extractor 实验配置，不改动原
`configs/crack_batch/efficientvit_b0.py`。

B0 只有 stage 2/3 有 LiteMLA（各 2 个，共 4 个）；stage 0/1 是纯 MBConv。
因此“替换全部相似作用的部分”就是替换这 4 个 LiteMLA，MBConv 保持不变。

## Round 4：OA-LA（当前主线）

**Orientation-Adaptive Linear Attention**：用 structure tensor 估计局部裂缝方向，
动态加权 0°/45°/90°/135° 四个各向异性 strip 聚合分支；替换全部 4 个 LiteMLA。

| 配置 | block_type | 参数量 | FLOPs | 说明 |
| --- | --- | ---: | ---: | --- |
| `efficientvit-b0-oala_crack-512x512.py` | `oala` | 0.9073M | 0.667G | 方向场 + 4 方向 strip 聚合 |

- 保留 LiteMLA 的 qkv/aggreg/proj，可加载预训练；
- 方向分支 `gamma=0` 初始化，初始等价于原模型；
- 方向场由 Sobel + structure tensor 计算，无可学习参数；
- 4 个方向权重由方向和 coherence 动态生成。

## Round 3：特征提取器改造

| 配置 | block_type | 参数量 | FLOPs | 说明 |
| --- | --- | ---: | ---: | --- |
| `efficientvit-b0-asgla_crack-512x512.py` | `asgla` | 0.7575M | 0.591G | LiteMLA + 水平/垂直 strip 聚合，零初始化 |
| `efficientvit-b0-cdgl_crack-512x512.py` | `cdgl` | 0.6460M | 0.535G | 通道解耦：global LiteMLA + local strip |
| `efficientvit-b0-cdgl-asgla_crack-512x512.py` | `cdgl_asgla` | 0.6857M | 0.555G | CDGL 的 global 分支换成 AS-GLA |
| `efficientvit-b0-foveal_crack-512x512.py` | `foveal` | 0.7114M | 0.477G | 2× 下采样全局 + 全分辨率 strip 局部 |

baseline：`0.7045M / 0.564G`。

说明：

- AS-GLA 保留 LiteMLA 的 qkv/aggreg/proj 参数名，可加载预训练权重；新 strip
  分支零初始化；
- Foveal 也保留 qkv/aggreg/proj，local 分支零初始化；
- CDGL 的 global 分支通道数变小，不能直接加载原 LiteMLA qkv，属于随机初始化；
- 所有新增模块都替换全部 4 个 LiteMLA，不是只改一个 stage。

## Round 2：全量替换成通用 attention（负结果）

| 配置 | block_type | 参数量 | FLOPs | 说明 |
| --- | --- | ---: | ---: | --- |
| `efficientvit-b0-replace-window_crack-512x512.py` | `replace_window` | 0.5519M | 0.514G | 全部 LiteMLA → 8×8 window attention |
| `efficientvit-b0-replace-hybrid_crack-512x512.py` | `replace_hybrid` | 0.5545M | 0.527G | stage 2 window；stage 3 global |
| `efficientvit-b0-replace-global_crack-512x512.py` | `replace_global` | 0.5558M | 0.780G | 全部 LiteMLA → global attention |

40k 结果：

```text
baseline fp32 : 62.22 IoU/crack
replace_window: 61.12 (-1.10)
replace_hybrid: 61.35 (-0.87)
replace_global: 61.34 (-0.88)
```

说明：直接换成通用 window/global attention 会丢掉 LiteMLA 的 multi-scale
aggregation，且新 attention 没有预训练权重，因此效果更差。

## Round 1：残差式增强（负结果）

| 配置 | block_type | 参数量 | FLOPs | 说明 |
| --- | --- | ---: | ---: | --- |
| `efficientvit-b0-coord_crack-512x512.py` | `coord` | 0.7138M | 0.564G | LiteMLA 后加 CoordAttention |
| `efficientvit-b0-hybrid_crack-512x512.py` | `hybrid` | 0.8283M | 0.611G | stage 3 全局 attention + EdgeGate；stage 2 CoordAttention |
| `efficientvit-b0-window_crack-512x512.py` | `window` | 0.8593M | 0.649G | stage 2/3 window attention；stage 2 Coord；stage 3 EdgeGate |

40k 结果：

```text
baseline fp32 : 62.22
coord         : 61.65
hybrid        : 62.10
window        : 61.74
```

详细设计、代码结构、实验结果见：

- [docs/20261023_EfficientViT_Attention_Upgrade.md](../../docs/20261023_EfficientViT_Attention_Upgrade.md)

## 运行

```bash
# 默认跑 Round 4 OA-LA
python scripts/efficientvit_attn_20261023/run_variants.py --gpus 0

# 双卡队列：OA-LA + baseline
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0,1 \
  --variants baseline,oala \
  --batch-size 12

# 单个变体
python tools/train.py \
  configs/efficientvit_attn_20261023/efficientvit-b0-oala_crack-512x512.py \
  --work-dir work_dirs/efficientvit_attn_20261023/oala
```

默认 fp32；OA-LA 保留 qkv/proj/原 5×5 聚合，可加载原 LiteMLA 权重。
CDGL 的 global 分支不能加载原 LiteMLA 权重，正式对比时建议补
from-scratch baseline（`--cfg-options load_from=None`）。
