# EfficientViT attention experiment scripts (2026-10-23)

本目录是 EfficientViT-B0 attention / feature-extractor 实验脚本，详细设计见：

- [docs/20261023_EfficientViT_Attention_Upgrade.md](../../docs/20261023_EfficientViT_Attention_Upgrade.md)

## Round 4：OA-LA（默认）

替换 stage 2/3 的全部 4 个 LiteMLA，MBConv 保持不变：

- `oala`：structure tensor 方向场 + 4 方向 strip 聚合

## Round 3：特征提取器改造（对照）

- `asgla`：LiteMLA + 水平/垂直 strip 聚合
- `cdgl`：通道解耦 global/local
- `cdgl_asgla`：CDGL + AS-GLA global 分支
- `foveal`：2× 下采样全局 + 全分辨率 strip 局部

## 脚本

### `run_variants.py`

顺序运行（单卡），默认跑 OA-LA：

```bash
python scripts/efficientvit_attn_20261023/run_variants.py \
  --gpus 0 --batch-size 12
```

### `batch_train_attn.py`

双卡队列：一张卡一个版本，结束一个补一个，带日志、失败摘要和
`batch_summary.json`：

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0,1 --batch-size 12
```

默认只跑：

```text
GPU0: oala
```

要跑多个版本时显式传 `--variants`，例如：

```bash
python scripts/efficientvit_attn_20261023/batch_train_attn.py \
  --gpus 0,1 --variants baseline,oala --batch-size 12
```

默认 fp32；OA-LA 保留 qkv/proj/原 5×5 聚合，可加载原 LiteMLA 权重。
CDGL global 分支不能加载原 LiteMLA 权重，正式对比建议补 from-scratch baseline。

## Round 1 / 2：负结果对照

- `coord`、`hybrid`、`window`：保留 LiteMLA 的叠加式 attention；
- `replace_window`、`replace_hybrid`、`replace_global`：通用 attention 全量替换。

这两组已完成 40k，均未超过 baseline（62.22 IoU/crack），只保留作对照。

## 变体列表

```text
baseline
coord
hybrid
window
replace_window
replace_global
replace_hybrid
asgla
cdgl
cdgl_asgla
foveal
oala
```
