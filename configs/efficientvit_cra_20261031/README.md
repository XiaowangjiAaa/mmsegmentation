# EfficientViT-B0 CRA v2 variants (2026-10-31)

本目录是 CRA 的覆盖改进版本。所有版本替换 stage 2/3 的全部 4 个 LiteMLA，
MBConv 不变。

## 版本

| 配置 | block_type | 参数量 | 说明 |
| --- | --- | ---: | --- |
| `efficientvit-b0-cra_crack-512x512.py` | `cra` | 0.9245M | v1：16 曲线 × 16 点，0.8 长度 |
| `efficientvit-b0-cra-dense_crack-512x512.py` | `cra` | 0.9353M | 36 曲线 × 24 点，提高覆盖 |
| `efficientvit-b0-cra-long_crack-512x512.py` | `cra` | 0.9307M | 16 曲线 × 32 点，长曲线 |
| `efficientvit-b0-cra-soft_crack-512x512.py` | `cra` | 0.9353M | dense + Gaussian soft splat |

baseline：`0.7045M / 0.564G`。

40k from-scratch 结果：

```text
baseline scratch : 58.29 IoU/crack
cra              : 56.32
cra_dense        : 55.30
cra_long         : 55.88
cra_soft         : 54.17
```

四个 CRA 版本均低于 baseline。2026-11-02 复核发现旧实现遗漏 seed 坐标且重复添加残差，
现已修复；暂停根据旧结果终止路线的判断。修复后应使用
`configs/efficientvit_recheck_20261102/` 的独立配置与 work_dir 复验，勿覆盖历史目录。

## 设计意图

v1 的问题是 Recall 低，主要因为曲线覆盖不足。v2 分别从三个方向改：

- `cra_dense`：更多 seed / 曲线点，增加空间覆盖；
- `cra_long`：更长曲线 + 更多点，覆盖连续裂缝；
- `cra_soft`：Gaussian soft splat，把曲线特征扩散到邻域，而不是只写回采样点。

## 运行

```bash
# 默认：baseline + 4 个 CRA v2
python scripts/efficientvit_cra_20261031/batch_train_cra.py \
  --gpus 0,1 --scratch --batch-size 12

# 只跑 CRA v2
python scripts/efficientvit_cra_20261031/batch_train_cra.py \
  --gpus 0 --variants cra_dense,cra_long,cra_soft --scratch --batch-size 12
```

默认 fp32；CRA 不建议 AMP。

## 输出

```text
work_dirs/efficientvit_cra_20261031/batch/
├── batch_summary.json
├── logs/
├── baseline/
├── cra/
├── cra_dense/
├── cra_long/
└── cra_soft/
```
