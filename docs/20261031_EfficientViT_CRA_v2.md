# EfficientViT-B0 CRA v2：覆盖改进实验（2026-10-31）

## 1. v1 的结论

CRA v1（16 曲线 × 16 点）40k from scratch：

```text
baseline scratch : 57.78 IoU/crack
CRA v1 scratch   : 56.13 (-1.65)
Precision        : 71.15 → 73.63
Recall           : 75.45 → 70.25
```

问题不是曲线分支没训练，而是：

- 曲线覆盖不足；
- Recall 下降 5.20；
- 没有 coverage/diversity 辅助 loss；
- splat 只在采样点写回，空间覆盖有限。

## 2. v2 的四个版本

| 版本 | 曲线数 | 每曲线点数 | 长度 | Splat | 目的 |
| --- | ---: | ---: | ---: | --- | --- |
| `cra` | 16 | 16 | 0.8 | bilinear | v1 对照 |
| `cra_dense` | 36 | 24 | 0.7 | bilinear | 增加覆盖 |
| `cra_long` | 16 | 32 | 1.2 | bilinear | 覆盖长裂缝 |
| `cra_soft` | 36 | 24 | 0.7 | Gaussian, σ=1.0 | 邻域扩散覆盖 |

参数量：

```text
baseline   : 0.7045M
cra        : 0.9245M
cra_dense  : 0.9353M
cra_long   : 0.9307M
cra_soft   : 0.9353M
```

## 3. 结构变化

### 3.1 Dense

```text
num_curves: 16 → 36
points_per_curve: 16 → 24
```

6×6 seed 网格，覆盖更密。

### 3.2 Long

```text
points_per_curve: 16 → 32
curve_length: 0.8 → 1.2
```

单条曲线更长，适合跨越整张图的连续裂缝。

### 3.3 Soft splat

原 v1：

```text
每个采样点只写回 4 个双线性邻居
```

v2 soft：

```text
每个采样点按 Gaussian 权重写回 (2r+1)² 邻域
σ=1.0, radius=2
```

这样即使曲线稍微偏离裂缝中心，特征也能扩散到附近像素。

## 4. 运行

```bash
# 默认 5 个版本
python scripts/efficientvit_cra_20261031/batch_train_cra.py \
  --gpus 0,1 --scratch --batch-size 12

# 只跑覆盖改进版
python scripts/efficientvit_cra_20261031/batch_train_cra.py \
  --gpus 0 --variants cra_dense,cra_long,cra_soft --scratch --batch-size 12
```

## 5. 预期

- `cra_dense`：Recall 应高于 v1；
- `cra_long`：长裂缝连续性应更好；
- `cra_soft`：Precision 可能略降，但 Recall 应提升；
- 如果三者仍低于 from-scratch baseline，说明曲线 tokenization 本身不适合当前 B0/数据。

## 6. 成本

`grid_sample` / scatter 的实际速度比 FLOPs 统计更慢；dense/soft 版本更明显。
默认 fp32，不建议 AMP。

## 7. 40k 结果

五个版本均完成 40k from scratch，fp32：

| 模型 | best iter | mIoU | mDice | IoU/crack | F1/crack | Precision | Recall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **baseline scratch** | 40k | **78.68** | **86.59** | **58.29** | **73.65** | 71.97 | **75.42** |
| cra | 40k | 77.69 | 85.79 | 56.32 | 72.06 | **73.62** | 70.57 |
| cra_dense | 35k | 77.16 | 85.36 | 55.30 | 71.21 | 72.83 | 69.67 |
| cra_long | 38k | 77.44 | 85.60 | 55.88 | 71.70 | 70.29 | 73.16 |
| cra_soft | 33k | 76.57 | 84.88 | 54.17 | 70.28 | 69.49 | 71.09 |

相对 baseline scratch：

```text
cra        : -1.97
cra_dense  : -2.99
cra_long   : -2.41
cra_soft   : -4.12
```

### 7.1 结论

- **没有任何 CRA v2 版本超过 baseline**；
- v1 CRA 仍是 CRA 系列最好的（56.32）；
- dense / long / soft 都没有把 Recall 拉回 baseline：
  - dense 反而降到 69.67；
  - long 的 Recall 73.16，但 Precision 只有 70.29；
  - soft 最差，54.17 IoU；
- 增加曲线密度、长度、soft splat 都没有解决根本问题。

### 7.2 原因

- CRA 用 **局部 depthwise + 稀疏曲线分支** 替掉 LiteMLA，丢掉了 dense 全局线性上下文；
- 曲线 token 只覆盖少数采样点，splat 无法恢复 LiteMLA 的 dense 特征；
- 更多曲线/更软 splat 会进一步稀释或模糊特征；
- 问题不是“覆盖参数不够”，而是 **曲线 tokenization 本身不适合当前 B0/数据**。

### 7.3 最终判断

CRA v1/v2 都是负结果，建议停止这条路线：

```text
CRA v1 : 56.13
CRA v2 : 54.17–56.32
baseline scratch : 58.29
baseline pretrained : 62.22
```

不再继续调 `num_curves / points / curve_length / splat_sigma`。
