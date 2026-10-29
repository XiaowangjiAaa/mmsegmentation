# EfficientViT-B0 CRA variants (2026-10-30)

本目录是 Curve-Routed Attention（CRA）实验配置。CRA 用曲线 token 替代
stage 2/3 的全部 4 个 LiteMLA；MBConv 不变。

## 配置

| 配置 | block_type | 参数量 | FLOPs | 说明 |
| --- | --- | ---: | ---: | --- |
| `efficientvit-b0-cra_crack-512x512.py` | `cra` | 0.9245M | 0.573G | 4×4 seed + 16 点曲线 + curve attention |

baseline：`0.7045M / 0.564G`。

## CRA 结构

```text
4×4 seed 网格
→ 每个 seed 预测 16 点曲线（base line + learned offsets）
→ grid_sample 沿曲线采样
→ curve-aware self-attention
→ 双线性 splat 回特征图
→ x + gamma * fuse(local, splat)
```

- 曲线分支 `gamma=0` 初始化；
- 曲线 proposal 由全局上下文 + seed/point embedding 预测；
- 曲线覆盖整张图，offset 可学习；
- 替换全部 4 个 LiteMLA。

## 运行

```bash
# baseline + CRA（双卡）
python scripts/efficientvit_cra_20261030/batch_train_cra.py \
  --gpus 0,1 --batch-size 12

# 只跑 CRA
python scripts/efficientvit_cra_20261030/batch_train_cra.py \
  --gpus 0 --variants cra --batch-size 12

# from-scratch 对照（CRA 没有预训练 attention，推荐）
python scripts/efficientvit_cra_20261030/batch_train_cra.py \
  --gpus 0,1 --variants baseline,cra --scratch --batch-size 12
```

默认 fp32；CRA 不建议 AMP。

详细设计、动机、与顶会工作的区别见：

- [docs/20261030_EfficientViT_CRA.md](../../docs/20261030_EfficientViT_CRA.md)
