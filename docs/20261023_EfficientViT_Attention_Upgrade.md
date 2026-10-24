# 2026-10-23 EfficientViT-B0 Attention Upgrade

本文记录在 **EfficientViT-B0 / 裂缝分割** 上新增的三组轻量 attention 变体，包括设计依据、代码改动、参数量、运行方法和实验计划。

- 基线：`configs/crack_batch/efficientvit_b0.py`
- 基线验证结果：裂缝 IoU **62.28%**，F1 **76.76%**，mIoU **80.74%**
- 基线参数量：**0.7045M**
- 新代码目录：`mmseg/models/backbones/efficientvit/attn_plugins/`
- 新配置目录：`configs/efficientvit_attn_20261023/`
- 新脚本目录：`scripts/efficientvit_attn_20261023/`

---

## 1. 为什么要改 attention

当前 B0 的 context block 是：

```text
LiteMLA（ReLU 线性注意力）
    +
MBConv（局部卷积）
```

B0 的 attention 只在 stride 16/32 的 stage 2/3 出现。它的优点是参数量极低，但对裂缝有三个不足：

1. **全局线性注意力容易平均化**：细裂缝、弱边界容易被背景稀释；
2. **缺少方向性/边缘先验**：裂缝是细长、方向性强的结构，普通全局 attention 不擅长；
3. **只在低分辨率 stage 做 token mixing**：缺少局部窗口和高分辨率细节。

因此本方案不是简单“加大 attention”，而是做三组**低成本的残差式增强**：

```text
原有 LiteMLA / MBConv 完全保留
        ↓
新增 attention 分支使用 zero-init 残差门控
        ↓
初始时等价于原 B0，不破坏已加载的官方 checkpoint
```

所有新增模块的 `gamma` 初始为 0：

```python
out = x * (1 + gamma * (gate - 1))   # CoordAttention / EdgeGate
out = x + gamma * attn_branch(x)     # Global / Window attention
```

因此第 0 步时新模块是恒等映射，训练开始后再逐渐学习。

---

## 2. 参考的 2025/2026 顶会工作

### 2.1 MixA（ICCV 2025）

**Mixed Attention / SteLLA**

- 关键层保留 quadratic / softmax attention；
- 非关键层用稳定的 ReLU linear attention；
- 核心观点：不是所有层都需要 full attention，应该把计算放在关键层。

对 B0 的启发：

- stage 3 只有 16×16 = 256 token，full attention 非常便宜；
- stage 2 是 32×32 = 1024 token，继续用 LiteMLA 更合适；
- 因此 `hybrid` 变体只在 stage 3 加一个全局 attention 残差。

链接：

- https://openaccess.thecvf.com/content/ICCV2025/html/Ahmed_MixA_A_Mixed_Attention_approach_with_Stable_Lightweight_Linear_Attention_ICCV_2025_paper.html

### 2.2 MixerCSeg（CVPR 2026，裂缝分割）

**TransMixer + DEGConv + SRF**

- TransMixer 将 token 解耦为 global / local 两条路径；
- DEGConv 使用方向先验 + 1×k / k×1 strip conv 做边缘门控；
- SRF 做多尺度空间细化；
- 2.54M 参数 / 2.05 GFLOPs，专门针对裂缝。

对 B0 的启发：

- 不需要搬整个 Mamba/TransMixer；
- **DEGConv 的 strip-conv 边缘门控**非常适合裂缝；
- 本方案的 `EdgeGate` 是它的轻量简化版，只保留 1×k / k×1 depthwise + sigmoid gate。

链接：

- https://openaccess.thecvf.com/content/CVPR2026/html/Zhao_MixerCSeg_An_Efficient_Mixer_Architecture_for_Crack_Segmentation_via_Decoupled_CVPR_2026_paper.html
- https://github.com/spiderforest/MixerCSeg

### 2.3 PoM（CVPR 2026 Findings）

**Polynomial Mixer**

- 线性时间 token mixing，作为 self-attention 的替代；
- 在多个领域验证了 linear-time 的可行性。

本方案暂未直接使用 PoM，因为 B0 的 LiteMLA 已经是很强的线性 mixer；PoM 更适合作为后续替换实验。

链接：

- https://openaccess.thecvf.com/content/CVPR2026F/html/Picard_PoM_A_Linear-Time_Replacement_for_Attention_with_the_Polynomial_Mixer_CVPRF_2026_paper.html

### 2.4 Inter2Former（ICCV 2025）

**Dynamic Hybrid Attention**

- 边界 token 走 full attention；
- 非边界 token 走 O(N) 的 BSQ linear attention；
- 用 mask/边界信息动态分配计算。

这是“边界感知路由”的代表工作。本方案没有直接实现路由，但 `EdgeGate` 和 stage 3 global attention 是它的轻量近似。后续可以在此基础上做 Sobel/edge 路由。

链接：

- https://openaccess.thecvf.com/content/ICCV2025/html/Huang_Inter2Former_Dynamic_Hybrid_Attention_for_Efficient_High-Precision_Interactive_Segmentation_ICCV_2025_paper.html

### 2.5 其他相关工作

- **HyPiDecoder（ICCV 2025）**：用卷积 FPN 替代部分 deformable attention，说明 decoder 中“轻量卷积 + 少量注意力”更划算。
- **CASWiT（CVPR 2026 Findings）**：stage-wise cross-attention，把低分辨率上下文注入高分辨率特征。
- **Gazemo（CVPR 2026 Findings）**：foveal-peripheral 双分支，高分辨率局部 + 低分辨率全局。

这些工作更适合后续改 SegHead / decoder，不在本次第一版中实现。

---

## 3. 新增代码结构

```text
mmseg/models/backbones/efficientvit/
├── backbone.py                     # 增加 block_type / attn_kwargs
├── layers.py                       # 原 LiteMLA / EfficientViTBlock，未改
└── attn_plugins/                   # 新增
    ├── __init__.py
    ├── modules.py                  # CoordAttention / EdgeGate / GlobalAttention / WindowAttention
    └── blocks.py                   # Coord / Hybrid / Window block 和工厂函数
```

### 3.1 `backbone.py` 的改动

新增参数：

```python
EfficientViT(
    arch='b0',
    block_type='v1',        # 'v1' | 'coord' | 'hybrid' | 'window'
    attn_kwargs=None,       # 传给新 block 的超参
    ...
)
```

- `block_type='v1'`：完全保持原行为；
- `block_type='coord'`：所有 context block 加 CoordAttention；
- `block_type='hybrid'`：stage 3 加 global attention + edge gate，stage 2 加 coord；
- `block_type='window'`：stage 2/3 加 window attention，stage 2 加 coord，stage 3 加 edge。

原来的 B0 配置没有传 `block_type`，默认仍是 `v1`，不会影响现有训练/评估。

### 3.2 新增模块

#### CoordAttention

```text
H 方向平均池化 + W 方向平均池化
→ 拼接
→ 1×1 降维 + BN + Hardswish
→ 拆回 H/W
→ 1×1 升维 + sigmoid
→ 与输入相乘
```

特点：

- 参数量约几千；
- 对细长、条状目标的 channel/spatial 定位更敏感；
- 适合 stage 2/3。

#### EdgeGate

MixerCSeg DEGConv 的轻量版：

```text
1×1 降维
→ 1×k depthwise（水平）
→ k×1 depthwise（垂直）
→ 拼接
→ 1×1 升维 + BN + sigmoid
→ 与输入相乘
```

特点：

- 直接建模裂缝方向；
- 参数量约几万；
- 只加在 stage 3。

#### ResidualGlobalAttention

- 仅在 stage 3（16×16 = 256 token）使用；
- LayerNorm + depthwise local conv + multi-head QKV；
- 标准 softmax attention；
- 分支输出 `x + gamma * proj(attn)`，`gamma=0` 初始化；
- 计算在 fp32 中完成，避免 AMP 溢出。

#### ResidualWindowAttention

- 8×8 非重叠窗口；
- 在窗口内做 multi-head attention；
- 适合 stage 2/3 的局部细节；
- 同样 `gamma=0` 残差。

---

## 4. 三个新版本

### V1：`coord`

```text
stage 2/3:
    LiteMLA → CoordAttention → MBConv
```

配置：

```text
configs/efficientvit_attn_20261023/
  efficientvit-b0-coord_crack-512x512.py
```

实测参数量/FLOPs（512×512，mmengine analysis；部分自定义算子会被低估，只用于相对比较）：

| 模型 | 总参数 | 相对 baseline | FLOPs | 相对 baseline |
| --- | ---: | ---: | ---: | ---: |
| baseline B0 | 0.7045M | — | 0.564G | — |
| V1 coord | 0.7138M | +1.3% | 0.564G | ~0% |
| V2 hybrid | 0.8283M | +17.6% | 0.611G | +8.3% |
| V3 window | 0.8593M | +22.0% | 0.649G | +15.1% |

V1 的增量是 **+0.0093M**，几乎不增加 FLOPs。

预期：

- 裂缝 IoU +0.3 ～ +0.8；
- 几乎不增加训练时间；
- 最安全的第一枪。

### V2：`hybrid`

```text
stage 2:
    LiteMLA → CoordAttention → MBConv

stage 3:
    LiteMLA → ResidualGlobalAttention → MBConv → EdgeGate
```

配置：

```text
configs/efficientvit_attn_20261023/
  efficientvit-b0-hybrid_crack-512x512.py
```

实测参数量：

| 模型 | 总参数 | backbone | FLOPs |
| --- | ---: | ---: | ---: |
| V2 hybrid | 0.8283M | 0.8064M | 0.611G |

增量：**+0.1238M（+17.6%）**，FLOPs **+8.3%**。

预期：

- 裂缝 IoU +1.0 ～ +2.0；
- FLOPs 增加主要来自 stage 3 的 256-token full attention 和 EdgeGate；
- 是“少量计算换明显精度”的主推版本。

### V3：`window`

```text
stage 2/3:
    LiteMLA → ResidualWindowAttention → MBConv
stage 2:
    + CoordAttention
stage 3:
    + EdgeGate
```

配置：

```text
configs/efficientvit_attn_20261023/
  efficientvit-b0-window_crack-512x512.py
```

实测参数量：

| 模型 | 总参数 | backbone | FLOPs |
| --- | ---: | ---: | ---: |
| V3 window | 0.8593M | 0.8374M | 0.649G |

增量：**+0.1548M（+22.0%）**，FLOPs **+15.1%**。

预期：

- 裂缝 IoU +1.0 ～ +2.5；
- 窗口 attention 比 global attention 更保留局部连续性；
- 但参数量和调参复杂度最高。

---

## 5. 运行方法

### 5.1 环境

Windows：

```powershell
conda activate F:\UTS_work_code\mmsegmentation\.venv-mmseg
```

Linux：

```bash
conda activate /data/yinwu/mmsegmentation/.venv-mmseg
```

### 5.2 查看命令

```bash
python scripts/efficientvit_attn_20261023/run_variants.py --dry-run
```

### 5.3 只跑一个变体（推荐先跑 V1）

```bash
python scripts/efficientvit_attn_20261023/run_variants.py \
  --variants coord \
  --gpus 0 \
  --batch-size 12
```

### 5.4 跑三个变体

```bash
python scripts/efficientvit_attn_20261023/run_variants.py \
  --variants coord,hybrid,window \
  --gpus 0 \
  --batch-size 12
```

默认是 **fp32 训练**，因为 hybrid/window 的 attention 在 AMP 下可能出现 `grad_norm: inf`。如果只想用 AMP 跑 `coord`，可以加 `--amp`。

### 5.5 单配置直接训练

```bash
python tools/train.py \
  configs/efficientvit_attn_20261023/efficientvit-b0-hybrid_crack-512x512.py \
  --work-dir work_dirs/efficientvit_attn_20261023/hybrid
```

### 5.6 smoke test

```bash
python scripts/efficientvit_attn_20261023/run_variants.py \
  --variants coord,hybrid,window \
  --max-iters 2 \
  --batch-size 2 \
  --work-root work_dirs/attn_smoke
```

### 5.7 双卡批量队列（推荐正式跑）

`batch_train_attn.py` 的行为和之前的 `batch_train.py` 类似：

- 第一个 GPU 跑队列第一个版本；
- 第二个 GPU 跑第二个版本；
- 任一 GPU 结束，队列下一个版本自动补上；
- 每个版本独立 work_dir、独立日志，最后写 `batch_summary.json`；
- 某个版本失败时自动打印最后 15 行日志。

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0,1 \
  --variants coord,hybrid,window \
  --batch-size 12
```

执行顺序：

```text
GPU0: coord
GPU1: hybrid
GPU0 空出后: window
```

输出：

```text
work_dirs/efficientvit_attn_20261023/batch/
├── batch_summary.json
├── logs/
│   ├── coord_<timestamp>.log
│   ├── hybrid_<timestamp>.log
│   └── window_<timestamp>.log
├── coord/best_mIoU_iter_*.pth
├── hybrid/best_mIoU_iter_*.pth
└── window/best_mIoU_iter_*.pth
```

单卡：

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0 --variants coord,hybrid,window
```

加入 baseline 一起比较（会多跑一个 fp32 baseline）：

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0,1 \
  --variants baseline,coord,hybrid,window
```

smoke test：

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0 \
  --variants coord,hybrid,window \
  --max-iters 2 \
  --batch-size 2 \
  --work-root work_dirs/attn_batch_smoke
```

默认 fp32；`--amp` 会对所有版本开启，不建议给 `hybrid/window` 使用。`--no-stream-logs` 可关闭实时终端输出。

---

## 6. 实验设计

| 实验 | 配置 | 目的 |
| --- | --- | --- |
| baseline | `configs/crack_batch/efficientvit_b0.py` | 62.28 IoU 基线 |
| V1 | coord | 最小改动是否有效 |
| V2 | hybrid | stage 3 全局 attention + 边缘 gate |
| V3 | window | 局部窗口 attention 是否更好 |
| V2+ | hybrid + 更大 attn_dim / 更多 heads | 只在上面的实验有效后再调 |
| V3+ | window + 不同 window_size | 8 / 16 对比 |

统一条件：

- 同一数据划分、同一 CE 0.5 + Dice 1.0、同一 40k schedule；
- 新模块 `gamma=0` 初始化；
- `load_from` 仍加载 `work_dirs/pretrained/efficientvit_b0_crack.pth`；
- 记录 `mIoU`、`IoU/crack`、`F1/crack`、`Precision/crack`、`Recall/crack`；
- 同时记录参数量、FLOPs、FPS（后续补测）。

判断标准：

```text
V1 若 +0.5 IoU 以上 → 保留
V2 若比 V1 再高 +0.8 以上 → 主推
V3 若比 V2 再高 +0.5 以上 → 保留
若没有提升 → 先检查 gamma 是否学习、attention 是否被梯度忽略
```

---

## 7. 注意事项

### 7.1 预训练权重

新模块会增加参数。`load_from` 使用 `strict=False` 语义，缺失的新参数会随机初始化；`gamma=0` 保证初始输出等价于原模型。

### 7.2 AMP

- `coord` 在 AMP 下 smoke test 正常；
- `hybrid` / `window` 的 attention 分支在 AMP 下出现过 `grad_norm: inf`，因此默认 fp32；
- 如果一定要 AMP，建议只对 `coord` 开启，并监控 `grad_norm` / loss 是否 NaN。

### 7.3 显存

B0 很小，batch 12 / 512 在 8GB 卡上通常够用；如果 hybrid/window OOM，先降 batch：

```bash
python scripts/efficientvit_attn_20261023/run_variants.py \
  --variants hybrid,window --batch-size 6
```

### 7.4 这不是“保证 +5 IoU”的方案

三组变体是**可验证的假设**，预期是 +0.5 ～ +2.5 IoU。若目标是 +5 以上，需要同时考虑：

- 更高分辨率特征进入 SegHead；
- B1/B2 宽度；
- 更强的前景 loss / Tversky；
- 更长训练或额外数据。

---

## 8. 如何修改/扩展

### 修改 V2 的 attention 维度

编辑：

```text
configs/efficientvit_attn_20261023/efficientvit-b0-hybrid_crack-512x512.py
```

```python
attn_kwargs=dict(
    global_stage=3,
    coord_stage=2,
    use_edge=True,
    attn_dim=64,      # 改成 32 / 96 / 128
    num_heads=4,      # 改成 2 / 8
    edge_kernel=7)
```

### 修改 V3 的窗口大小

```python
attn_kwargs=dict(
    window_size=8,    # 改成 4 / 16
    ...)
```

### 新增 block 类型

1. 在 `attn_plugins/blocks.py` 新增 `EfficientViTBlockXxx`；
2. 在 `build_efficientvit_block` 增加分支；
3. 在 `attn_plugins/__init__.py` 导出；
4. 新建 config，设置 `block_type='xxx'`。

---

## 9. 回退

不传 `block_type` 时，`EfficientViT` 默认 `v1`，与原实现完全一致：

```python
backbone=dict(type='EfficientViT', arch='b0')
```

因此这些改动不会影响：

- `configs/crack_batch/efficientvit_b0.py`
- `configs/efficientvit/` 下原有配置
- 现有 `best_mIoU` checkpoint 的加载

---

## 10. 首轮 40k 结果（2026-10-24）

三个变体均已完成 40k 训练（验证集 1208 张，512×512，fp32）：

| 模型 | best iter | mIoU | mDice | IoU/crack | Dice/crack | F1/crack | Precision/crack | Recall/crack |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline B0（AMP） | 38k | **80.74** | **88.18** | **62.28** | **76.76** | **76.76** | 76.05 | 77.47 |
| V1 coord | 40k | 80.42 | 87.93 | 61.65 | 76.28 | 76.28 | 76.10 | 76.46 |
| V2 hybrid | 38k | 80.65 | 88.11 | 62.10 | 76.62 | 76.62 | 76.88 | 76.35 |
| V3 window | 34k | 80.46 | 87.97 | 61.74 | 76.35 | 76.35 | 74.76 | 78.01 |

相对 baseline 的裂缝 IoU：

```text
V1 coord : -0.63
V2 hybrid: -0.18
V3 window: -0.54
```

### 结论

- **本轮三个 attention 变体都没有超过 baseline**；
- hybrid 最接近（-0.18 IoU），但仍在验证噪声量级内；
- coord/window 反而更低；
- 新增参数量/FLOPs 没有换来 mIoU 提升，**当前版本不建议直接替换 baseline**。

### 重要对照问题

baseline 是之前用 **AMP** 训练的，而三个变体是 **fp32** 训练，二者不是严格同精度对照。
要下定论，需要补一个 **fp32 baseline** 对照：

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0 --variants baseline --batch-size 12
```

如果 fp32 baseline 本身低于 62.28，那么 hybrid/coord 可能只是与它持平；如果 fp32
baseline 仍接近 62.28，则说明当前 attention 改造确实没有收益。

### 门控是否生效

从 best checkpoint 读取 `gamma`：

```text
coord:
  stage2: 0.126 / 0.135
  stage3: 0.230 / 0.243

hybrid:
  stage2 coord: 0.126 / 0.157
  stage3 global: -0.018 / -0.101
  stage3 edge: 0.582 / 0.905

window:
  stage2 window: -0.152 / -0.137
  stage2 coord: 0.097 / 0.141
  stage3 window: -0.073 / 0.079
  stage3 edge: 0.582 / -1.338
```

说明：

- `gamma` 明显离开 0，**attention 分支确实在工作**，不是“模块没被训练”；
- hybrid 的 edge gate 学得很大（0.58–0.91），但精度没有提升，说明方向/边缘特征
  可能与已有 MBConv 冗余，或在小数据上过拟合；
- global attention 的 gamma 很小甚至为负，说明 stage 3 已有 LiteMLA 的情况下，
  再叠加全局注意力收益有限。

### 下一步建议

1. **先补 fp32 baseline**，确认精度对照；
2. 当前 backbone attention 方向不做直接替换；
3. 如果继续，优先改 **SegHead / 高分辨率特征融合**，而不是继续加深 backbone attention；
4. 可尝试把 DEGConv/EdgeGate 放到 **SegHead 融合之后**，而不是 backbone stage 3；
5. 增加随机种子重复，确认 0.2–0.6 IoU 的差异是否只是验证噪声。

---

## 11. 当前状态

- 代码：已实现并通过 `py_compile`；
- 构建：3 个新配置均能 `MODELS.build()`；
- smoke：coord / hybrid / window 均完成 1–2 iter 前向/反向；
- 正式训练：三个变体均完成 40k；
- 结果：未超过 baseline，当前版本保留作对照，不推荐直接采用；
- 下一步：补 fp32 baseline，再决定是否改 SegHead 或停止该方向。
