# EfficientViT-B0 轻量改造实验（2026-11-02）

统一入口：`batch_train.py`。新增机制是针对现有 LiteMLA 的独立适配消融，
**不是 MALA / PolaFormer / RALA / TinyNeXt 的完整复现**；不使用这些论文的名称声称复现指标。
文献依据见 [调研记录](../../docs/20261102_EfficientViT_Lightweight_Research.md)。

## 版本

| 名称 | 修改 | 整网参数 | logits stride |
| --- | --- | ---: | ---: |
| baseline | 原始 LiteMLA / SegHead | 704,498 | 8 |
| magnitude | 零门控、保留 Query 幅值的中心化 KV 协方差增量 | 704,502 | 8 |
| polarity | 零门控同号减异号交互增量 | 704,502 | 8 |
| kv_reweight | 全局 Query 引导的正值 KV token 权重 | 704,502 | 8 |
| late_channel | stage 2 保留 LiteMLA，stage 3 改通道交互 | 704,498 | 8 |
| detail_resize | 仅将主 logits 插值到 stride 4 的对照 | 704,498 | 4 |
| detail | 原 head + 8 通道 stride-4 细节修正 | 704,749 | 4 |
| magnitude_detail | magnitude + 细节修正 | 704,753 | 4 |
| cra_fixed | 修复 seed 网格与重复残差的 CRA | 924,542 | 8 |

参数量由构建后的二分类模型统计，不含优化器。没有完成训练，尚无新精度或实测延迟结论。
细节分支只有 251 个额外参数，但空间计算与插值仍有成本；polarity 的矩阵乘法更多，
late_channel 也不保证实际提速，必须在相同设备上计时。

## 计算说明

所有新 information mixer 都保留原 `qkv/aggreg/proj` 的键和形状，多尺度聚合及 MBConv 不变。
对于 magnitude / polarity / kv_reweight，加载相同主干权重、门控为零时，mixer 初始行为与原 LiteMLA 相同。
末层 channel 的计算行为改变，虽然参数形状可加载，不保证等价。
同 seed 的 scratch 构建仍可能因模型随机数消耗不同而得到不同共享层初值，须报告多 seed。

令正值 Q/K 为 q/k，`A=q(K^T V)`，`s=q sum(K)`，`N=H*W`：

- magnitude：`A/(s+eps) + tanh(gamma)*(A-s*mean(V))/(N*d)`。
  分母 N*d 控制分辨率相关尺度，补偿项保留 q 的幅值；这是受幅值保真工作启发的适配，
  **并非原 MALA 的 beta/gamma 公式**。
- polarity：用原投影的正/负分量计算归一化同号、异号交互之差，零门控叠加到 baseline。
  没有引入完整 PolaFormer 的幂函数与双流设计。
- kv_reweight：`N*softmax(gamma*score)` 给 K 加权，其中 score 来自全局 Query 与各位置 Key。
  gamma=0 时权重均为 1；始终正值。不含 GT 或额外边缘监督，也不声称完成 RALA 秩增强。
- channel：在每个多尺度 head 内做 d×d 的通道交互，没有 N×N 空间矩阵。
- detail：保留 stride-8 主 head，将 logits 双线性插值到 stride-4，叠加零门控细节预测。
  最终回原图的两次插值与 baseline 一次插值有区别，所以提供 detail_resize 单独控制该因素。

注意力累积显式使用 fp32。旧 CRA 的历史 checkpoint 不可用于修复后指标对比。

## 构建与反向检查

在项目环境中运行：

```bash
python scripts/efficientvit_light_20261102/verify.py
```

检查所有配置的真实 CE+Dice loss、输出尺寸和梯度；不需要数据集或预训练文件。
可用 `--size 512` 检查实际裁剪尺寸，`--output work_dirs/light_verify.json` 保存报告。
Windows 若 YAPF 的语法缓存写入阻塞，可仅在验证进程加 `--no-yapf-cache`，不会改依赖包。

## 正式训练

先做两个独立机制及组合，3 个种子（默认 fp32、40k、batch=12）：

```bash
python scripts/efficientvit_light_20261102/batch_train.py --gpus 0,1 --variants baseline,magnitude,detail_resize,detail,magnitude_detail --seeds 42,43,44
```

完整从零训练对照，包含修复后 CRA：

```bash
python scripts/efficientvit_light_20261102/batch_train.py --gpus 0,1 --scratch --variants baseline,magnitude,polarity,kv_reweight,late_channel,detail_resize,detail,magnitude_detail,cra_fixed --seeds 42,43,44
```

小规模执行检查（需要实际数据和 CUDA）：

```bash
python scripts/efficientvit_light_20261102/batch_train.py --gpus 0 --scratch --variants baseline,magnitude --seeds 42 --batch-size 2 --max-iters 2 --work-root work_dirs/light_smoke
```

用 `--dry-run` 只显示命令，用 `--list` 列配置。`--max-iters` 同时调整 warmup、PolyLR、验证及保存间隔。
`--max-iters 80000` 对所有选中版本采用相同 80k schedule。
`--gpus` 是当前可见设备的逻辑索引，尊重父进程 `CUDA_VISIBLE_DEVICES`。
队列一项占用一张卡，空闲卡自动接下一项；退出状态非零会写失败记录；`--stop-on-failure` 停止剩余队列。

CRA 必须加 `--scratch` 且使用 fp32；脚本拒绝不公平的预训练混比与 CRA AMP。
从零训练不会下载预训练文件。预训练模式缺少转换权重时才调用原准备脚本。

## 输出和汇总

```text
work_dirs/efficientvit_light_20261102/
  pretrained/fp32/             # 或 scratch/fp32、pretrained/amp
    batch_summary.json
    seed42/baseline/train.log
    seed42/magnitude/...
    seed43/...
```

不复用已存在的运行目录，重跑请改 `--work-root`。完整日志位于各运行 `train.log`，终端显示队列状态。
summary 在每项结束后原子更新，包含命令、GPU、seed、状态、最佳 mIoU 点的前景指标、均值/样本标准差，
以及与同 seed baseline 的 IoU 差值。一个种子的 std 为 null，不伪装成零方差。

训练后也可重建指标汇总：

```bash
python scripts/efficientvit_light_20261102/summarize.py work_dirs/efficientvit_light_20261102/scratch/fp32
```

写入独立 `metrics_summary.json`，不会覆盖权威进程状态文件。
该重建只说明存在验证指标，可能包括中断实验；结合 `batch_summary.json` 确认是否完整。
不要合并不同初始化、精度或长度的实验作同一对照。

## 回归检查

```bash
python tests/test_scripts/test_efficientvit_light_runner.py
python tests/test_models/test_backbones/test_efficientvit_information.py
python tests/test_models/test_backbones/test_efficientvit_cra.py
```

也可使用 pytest 收集这些 unittest。检查包括预训练功能等价、混合精度零输入、非零分支反向梯度、
末层替换范围、细节插值对照、CRA 四角覆盖与唯一残差、成对汇总和失败队列退出状态。
