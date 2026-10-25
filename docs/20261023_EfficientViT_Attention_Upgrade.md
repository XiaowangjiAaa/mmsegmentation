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
  --variants replace_hybrid,replace_window,replace_global \
  --batch-size 12
```

执行顺序：

```text
GPU0: replace_hybrid
GPU1: replace_window
GPU0 空出后: replace_global
```

输出：

```text
work_dirs/efficientvit_attn_20261023/batch/
├── batch_summary.json
├── logs/
│   ├── replace_hybrid_<timestamp>.log
│   ├── replace_window_<timestamp>.log
│   └── replace_global_<timestamp>.log
├── replace_hybrid/best_mIoU_iter_*.pth
├── replace_window/best_mIoU_iter_*.pth
└── replace_global/best_mIoU_iter_*.pth
```

单卡：

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0 --variants replace_hybrid,replace_window,replace_global
```

加入 baseline 一起比较（会多跑一个 fp32 baseline）：

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0,1 \
  --variants baseline,replace_hybrid,replace_window,replace_global
```

smoke test：

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0 \
  --variants replace_hybrid,replace_window,replace_global \
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
- 构建：Round 1 的 3 个叠加配置 + Round 2 的 3 个全量替换配置均可
  `MODELS.build()`；
- smoke：Round 1 与 Round 2 的配置均完成 1–2 iter 前向/反向；
- 正式训练：Round 1 三个叠加变体、Round 2 三个全量替换变体、fp32 baseline
  均已完成 40k；
- Round 1 结果：未超过 baseline，只作为叠加式对照；
- Round 2 结果：全量替换比 fp32 baseline 低 0.87–1.10 IoU，当前不推荐采用；
- 下一步：补 from-scratch 对照，或转向 SegHead 高分辨率融合 / loss / B1-B2。

---

## 12. Round 2：全量替换 LiteMLA（真正做模块替换）

### 12.1 对 Round 1 的修正

Round 1 的三个版本是“保留 LiteMLA + 额外加 attention”：

- `hybrid` 只在 stage 3 加了 global attention；
- `window` 是在 LiteMLA 后追加 window attention；
- `coord` 是追加一个 gate。

这只能说明“在 LiteMLA 之上叠加 attention 没有收益”，**不能说明用新 attention
替换 LiteMLA 没有收益**。如果目标是验证一个 attention 模块，必须在所有承担
相同作用的位置上替换，而不是只改一个 stage 或只加一个残差分支。

B0 的结构里：

```text
stage 0/1：纯 MBConv，没有 attention
stage 2：2 个 LiteMLA
stage 3：2 个 LiteMLA
```

所以“全部替换相似作用的部分”就是替换 **stage 2/3 的全部 4 个 LiteMLA**；
stage 0/1 没有同类模块，不应为了“全替换”强行塞入高分辨率 attention，否则
是另一类“新增 attention”的实验。

### 12.2 全量替换协议

```text
每个 context block：
    原：LiteMLA → MBConv
    新：TokenMixer → MBConv
```

- 删除 LiteMLA，不再保留其分支；
- MBConv 保持不变，隔离变量；
- 不添加 CoordAttention / EdgeGate 等额外门控；
- 所有 4 个 context block 使用同一种 mixer（hybrid 是 stage 2/3 两种 scope
  的固定组合，但仍然是全部替换）。

### 12.3 三个全量替换版本

| 配置 | block_type | 参数量 | FLOPs | 说明 |
| --- | --- | ---: | ---: | --- |
| `efficientvit-b0-replace-window_crack-512x512.py` | `replace_window` | 0.5519M | 0.514G | 全部 LiteMLA → 8×8 window attention |
| `efficientvit-b0-replace-hybrid_crack-512x512.py` | `replace_hybrid` | 0.5545M | 0.527G | stage 2 window；stage 3 global |
| `efficientvit-b0-replace-global_crack-512x512.py` | `replace_global` | 0.5558M | 0.780G | 全部 LiteMLA → global attention |

baseline：

```text
0.7045M / 0.564G
```

值得注意：

- 全量替换后参数量反而下降到 **0.55M**，因为去掉了 LiteMLA 的 multi-scale
  aggregation/qkv 结构；
- `replace_window` / `replace_hybrid` 的 FLOPs 也略低于 baseline；
- `replace_global` 因为 stage 2 的 32×32 full attention，FLOPs 升到 0.780G。

### 12.4 运行

单卡顺序：

```bash
python scripts/efficientvit_attn_20261023/run_variants.py \
  --gpus 0 \
  --variants replace_hybrid,replace_window,replace_global \
  --batch-size 12
```

双卡队列（默认顺序）：

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0,1 \
  --variants replace_hybrid,replace_window,replace_global \
  --batch-size 12
```

执行：

```text
GPU0: replace_hybrid
GPU1: replace_window
GPU0 空出后: replace_global
```

`batch_train_attn.py` 和 `run_variants.py` 的默认 variants 已切换为这三个
全量替换版本。

### 12.5 40k 结果

baseline 也补跑了 fp32 对照（同一数据、同一 CE+Dice、同一 40k）：

| 模型 | best iter | mIoU | mDice | IoU/crack | Dice/crack | F1/crack | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **baseline B0（fp32）** | 37k | **80.70** | **88.15** | **62.22** | **76.71** | **76.71** | 75.74 | **77.70** |
| replace_window | 39k | 80.15 | 87.73 | 61.12 | 75.87 | 75.87 | 75.74 | 76.00 |
| replace_hybrid | 39k | 80.25 | 87.81 | 61.35 | 76.04 | 76.04 | 74.90 | 77.22 |
| replace_global | 39k | 80.26 | 87.81 | 61.34 | 76.04 | 76.04 | **76.24** | 75.84 |

相对 fp32 baseline 的裂缝 IoU：

```text
replace_window: -1.10
replace_hybrid: -0.87
replace_global: -0.88
```

结论：

- **全量替换 LiteMLA 后反而下降 0.87–1.10 IoU**；
- 三个替换版本都没有超过 fp32 baseline；
- `replace_hybrid` 是替换组里最好的，但仍低 0.87；
- 参数量/FLOPs 虽然更低（window/hybrid），但精度代价更大。

### 12.6 一个重要的初始化混淆

baseline 通过 `load_from` 加载了转换后的官方 checkpoint，其中包含 **LiteMLA 的
预训练权重**；全量替换版本把 LiteMLA 删除后，新 attention 是随机初始化的。

因此：

```text
baseline:      conv + LiteMLA 都来自预训练
replace_*:     conv 来自预训练，新 attention 从零开始
```

这不是严格同初始化的架构对比。要判断“替换结构本身是否更差”，还需要：

```bash
python tools/train.py <config> \
  --work-dir <dir> \
  --cfg-options load_from=None
```

至少补：

```text
baseline from scratch
replace_hybrid from scratch
```

如果 from-scratch 对照下替换仍然更低，才能确认 LiteMLA 的结构/多尺度聚合确实
比 window/global attention 更适合这个 B0 裂缝任务。

### 12.7 当前判断

- **当前不建议采用任何 attention 替换版本**；
- LiteMLA 的多尺度 depthwise aggregation + 线性注意力在 B0 上很强，简单换成
  window/global attention 会丢掉这部分归纳偏置；
- 继续在 backbone 上做 attention 替换的收益不明显；
- 下一步优先级：
  1. from-scratch 对照，确认初始化混淆；
  2. 改 **SegHead 高分辨率融合**（stride 4 特征、skip attention gate）；
  3. 在 SegHead 融合后加 DEGConv/CoordAttention，而不是替换 backbone LiteMLA；
  4. 增强前景 loss（Tversky/class weight）或换 B1/B2 提升容量。

---

## 13. Round 3：面向裂缝的特征提取器改造

Round 2 的结论是“不能把 LiteMLA 换成通用 window/global attention”，不是
“不能改特征提取器”。Round 3 改为在 LiteMLA 框架内设计新模块，并替换全部
4 个 LiteMLA。

### 13.1 四个版本

| 配置 | block_type | 参数量 | FLOPs | 说明 |
| --- | --- | ---: | ---: | --- |
| `efficientvit-b0-asgla_crack-512x512.py` | `asgla` | 0.7575M | 0.591G | LiteMLA + 水平/垂直 strip 聚合 |
| `efficientvit-b0-cdgl_crack-512x512.py` | `cdgl` | 0.6460M | 0.535G | 通道解耦 global/local |
| `efficientvit-b0-cdgl-asgla_crack-512x512.py` | `cdgl_asgla` | 0.6857M | 0.555G | CDGL + AS-GLA global 分支 |
| `efficientvit-b0-foveal_crack-512x512.py` | `foveal` | 0.7114M | 0.477G | 2× 下采样全局 + 全分辨率 strip 局部 |

baseline：`0.7045M / 0.564G`。

### 13.2 AS-GLA

- 继承 LiteMLA，保留 `qkv` / `aggreg` / `proj` 参数名；
- 增加水平 1×7、垂直 7×1 的 depthwise strip 聚合；
- strip 分支用 `gamma_h/gamma_w=0` 初始化，初始等价于原 LiteMLA；
- 可加载原 EfficientViT 预训练注意力权重。

### 13.3 CDGL

- 通道分成 global / local 两组；
- global 组走 LiteMLA（或 AS-GLA）；
- local 组走 1×7 / 7×1 strip conv；
- 1×1 融合；
- **global 分支通道数变小，不能直接加载原 LiteMLA qkv**，属于随机初始化。

### 13.4 CDGL + AS-GLA

- CDGL 的 global 分支使用 AS-GLA；
- 同时保留通道解耦和各向异性 strip；
- 参数量 0.6857M，略低于 baseline。

### 13.5 Foveal

- global 分支在 2× 下采样特征上跑 LiteMLA（peripheral）；
- local 分支在全分辨率上做 strip conv（foveal）；
- `gamma=0` 初始化，先使用全局路径；
- 保留 qkv/aggreg/proj，可加载原预训练注意力。

### 13.6 默认训练

`run_variants.py` / `batch_train_attn.py` 的默认 variants 已切换为：

```text
asgla, cdgl_asgla, cdgl, foveal
```

双卡：

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0,1 --batch-size 12
```

执行：

```text
GPU0: asgla
GPU1: cdgl_asgla
GPU0 空出后: cdgl
GPU1 空出后: foveal
```

### 13.7 当前状态

- 4 个配置均能 `MODELS.build()`；
- 均完成 1–2 iter 前向/反向 smoke；
- 参数量/FLOPs 已测量；
- **尚未进行 40k 正式训练**，因此没有 Round 3 的 mIoU 结果；
- CDGL/CDGL+AS-GLA 的 global 分支随机初始化，正式对比需要补
  from-scratch baseline 或后续做权重转换；
- Boundary-Routed Hybrid Attention（Inter2Former 风格）需要 token gather/
  scatter 和边界路由，暂未纳入本轮，作为下一阶段。

### 13.8 预期

```text
AS-GLA     : +0.5 ~ +1.5 IoU，参数量 +7.5%，FLOPs +4.8%
CDGL       : +0.5 ~ +1.5 IoU，参数量 -8.3%，FLOPs -5.1%
CDGL+AS-GLA: +0.8 ~ +1.8 IoU，参数量 -2.6%，FLOPs -1.6%
Foveal     : +0.5 ~ +1.5 IoU，参数量 +1.0%，FLOPs -15.4%
```

这些是待验证假设，不是保证结果。
