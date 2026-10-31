# EfficientViT-B0 Curve-Routed Attention (2026-10-30)

## 1. 为什么不是继续换 attention

Round 1–4 的结论：

```text
LiteMLA 保留 + 加 attention       → 61.65–62.10
LiteMLA 换成 window/global        → 61.12–61.35
LiteMLA 换成 AS-GLA/CDGL/Foveal   → 61.51–61.98
LiteMLA 换成 OA-LA                → 62.12
baseline                          → 62.22
```

这些工作的共同点是：

> 仍然在规则 H×W 网格上做 token mixing。

OA-LA 加了方向场，但 token 仍然是网格 token；它只是改变了聚合权重，
没有改变 tokenization。

裂缝是 **1D 曲线结构**，不是 2D 区域。真正的差异应该来自：

> **token 怎么产生、在哪里交互、怎么写回特征图。**

## 2. 顶会工作的可迁移点

| 工作 | 可借鉴 | 不足 |
| --- | --- | --- |
| MixA, ICCV 2025 | layer-wise 动态选择 | 仍是网格 token |
| Inter2Former, ICCV 2025 | token 路由 | 边界路由，仍依赖 dense token |
| MixerCSeg, CVPR 2026 | CNN/Transformer/Mamba 混合 | 直接换 encoder，工程味重 |
| Gazemo, CVPR 2026 Findings | foveal/peripheral | 分辨率分配，不是 tokenization |
| PoM, CVPR 2026 Findings | 线性 token mixer | 仍是 dense grid mixer |
| CASWiT, CVPR 2026 Findings | stage-wise cross-attention | 多尺度融合，不是曲线 |

因此本文不复制上述模块，而是利用它们的共同缺口：

> 现有高效分割 attention 没有 **曲线 tokenization**。

## 3. CRA：Curve-Routed Attention

### 3.1 总体结构

```text
X (C×H×W)
  │
  ├─ 4×4 seed 网格（16 个种子）
  │
  ├─ 每个 seed 预测 K=16 点曲线
  │    base line + learned offsets
  │
  ├─ grid_sample(X, curve coords)
  │    → T (B, M×K, C)
  │
  ├─ curve-aware self-attention
  │    seed embedding + point embedding
  │
  ├─ bilinear splatting 回 H×W
  │
  └─ X + gamma * fuse(local, splat)
```

### 3.2 曲线 proposal

固定 4×4 seed：

```text
seeds = (-0.75, -0.25, 0.25, 0.75)²
```

每个 seed 的基准方向：

```text
0°, 45°, 90°, 135° 循环
```

曲线点：

```text
base = seed + t * direction * curve_length
t ∈ [-0.5, 0.5]
```

偏移：

```text
e = seed_embed + point_embed + global_context
delta = tanh(MLP(e)) * offset_scale
coords = clamp(base + delta, -1, 1)
```

- `curve_length=0.8`；
- `offset_scale=0.15`；
- 曲线覆盖整张图，offset 由分割梯度学习。

### 3.3 曲线采样与 attention

```text
T = grid_sample(X, coords)
T = T + seed_embed + point_embed
T = LayerNorm(T + MHA(T))
T = LayerNorm(T + FFN(T))
```

- `M=16` 条曲线；
- `K=16` 点/曲线；
- 256 个 curve token；
- 4 个 attention heads。

### 3.4 Splatting

每个曲线点通过双线性权重写回 4 个相邻像素：

```text
out[yi, xi] += T * w
norm[yi, xi] += w
out = out / norm
```

避免 grid_sample 的逆操作不可用。

### 3.5 融合

```text
local = depthwise 3×3(X)
splat = Splat(T)
out = X + gamma * Conv1x1(cat(local, splat))
```

- `gamma=0` 初始化；
- 初始等价于 identity residual；
- 替换全部 4 个 LiteMLA；
- MBConv 不变。

## 4. 与已有工作的区别

| 方法 | token 来源 | 交互方式 |
| --- | --- | --- |
| LiteMLA | 网格 | 多尺度线性 attention |
| window/global | 网格 | 窗口/全局 |
| OA-LA | 网格 | 方向加权 |
| Inter2Former | 网格 | 边界 token 路由 |
| **CRA** | **预测曲线上的采样点** | **沿曲线 attention** |

CRA 的贡献不是“又一个 attention”，而是：

1. 曲线 tokenization；
2. 可学习的曲线 proposal；
3. 双线性 splatting；
4. 全量替换 LiteMLA。

## 5. 成本

| 模型 | 参数量 | FLOPs |
| --- | ---: | ---: |
| baseline B0 | 0.7045M | 0.564G |
| CRA | 0.9245M | 0.573G |

- 参数量 +31.3%；
- FLOPs 统计器不包含 grid_sample/scatter，实际会更高；
- 默认 fp32，不建议 AMP。

## 6. 训练

```bash
# baseline + CRA
python scripts/efficientvit_cra_20261030/batch_train_cra.py \
  --gpus 0,1 --batch-size 12

# 只跑 CRA
python scripts/efficientvit_cra_20261030/batch_train_cra.py \
  --gpus 0 --variants cra --batch-size 12

# from-scratch 对照
python scripts/efficientvit_cra_20261030/batch_train_cra.py \
  --gpus 0,1 --variants baseline,cra --scratch --batch-size 12
```

输出：

```text
work_dirs/efficientvit_cra_20261030/batch/
├── batch_summary.json
├── logs/
├── baseline/
└── cra/
```

## 7. 实验设计

| 实验 | 修改 | 目的 |
| --- | --- | --- |
| H0 | baseline | 62.22 |
| C1 | CRA 全量替换 | 验证曲线 tokenization |
| C2 | CRA + from-scratch baseline | 排除预训练混淆 |
| C3 | 固定曲线 vs 可学习 offset | 验证 offset |
| C4 | M/K 消融 | 曲线数量/采样密度 |
| C5 | 去掉 splat，改为 grid token | 隔离曲线 tokenization |
| C6 | 多 seed | 排除噪声 |

判断标准：

```text
C1 > baseline +0.5 → 值得继续
C1 ≈ baseline → 曲线 tokenization 没有收益
C1 < baseline → 当前实现/训练不稳定
```

## 8. 风险

- 曲线 proposal 可能塌缩到少数区域；
- splatting 的覆盖可能不均匀；
- CRA 没有预训练 attention，必须补 from-scratch 对照；
- grid_sample/scatter 的实际速度比 FLOPs 统计更慢；
- 目前没有 coverage/diversity 辅助 loss，v1 先验证曲线 tokenization 是否有效。

## 9. 40k 结果

`baseline,cra` 两组均从 scratch 训练 40k，fp32：

| 模型 | best iter | mIoU | mDice | IoU/crack | F1/crack | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline scratch | 39k | **78.41** | **86.38** | **57.78** | **73.24** | 71.15 | **75.45** |
| CRA scratch | 40k | 77.59 | 85.71 | 56.13 | 71.90 | **73.63** | 70.25 |

相对 from-scratch baseline：

```text
IoU/crack : -1.65
F1/crack  : -1.34
Precision : +2.48
Recall    : -5.20
mIoU      : -0.82
```

对照参考：

```text
baseline pretrained : 62.22
OA-LA pretrained    : 62.12
CRA scratch         : 56.13
baseline scratch    : 57.78
```

### 9.1 分析

- CRA 没有超过 from-scratch baseline，低 1.65 IoU；
- Precision 更高，但 Recall 低 5.20，F1 低 1.34；
- 说明曲线分支不是“没生效”，而是 **覆盖不足、召回下降**；
- `gamma` 已经学到较大值：

```text
stage2: +0.259 / -0.377
stage3: +0.431 / +0.678
```

- 曲线分支确实在参与，但曲线覆盖不到所有裂缝区域；
- 16 条曲线 × 16 点 = 256 个采样点，对整张图的裂缝网络太稀疏；
- 没有 coverage/diversity loss，曲线容易集中在少数区域；
- CRA 没有预训练 attention，from-scratch 40k 也不够充分；
- 当前 CRA v1 不能替代 baseline。

### 9.2 下一步建议

如果继续 CRA，应优先改“覆盖”而不是继续调 attention：

1. **增加曲线密度**：4×4 → 8×8 seed，K=16 → 32；
2. **coverage loss**：用 GT skeleton 监督曲线点覆盖裂缝；
3. **diversity loss**：避免曲线塌缩；
4. **多尺度曲线**：stage2 短曲线、stage3 长曲线；
5. **splat 改为软 splat**：Gaussian/邻域扩散，而不是只在采样点写回；
6. 再跑 from-scratch baseline + CRA 60–80k。

如果 CRA v2 仍然低于 baseline，说明当前曲线 tokenization 不适合 B0/裂缝，
应停止这条路线。
