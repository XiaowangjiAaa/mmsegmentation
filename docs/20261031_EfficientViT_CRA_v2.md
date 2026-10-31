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
