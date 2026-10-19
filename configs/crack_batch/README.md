# 批量裂纹训练（双 GPU）

`scripts/batch_train.py` 按队列依次训练 9 个轻量模型。每个 GPU 同一时间只跑
一个算法，先结束的 GPU 自动接上下一个。

## 算法与配置

| 队列 | 配置 | 骨干 | 优化器 |
| --- | --- | --- | --- |
| `convnext_tiny` | [convnext_tiny.py](convnext_tiny.py) | ConvNeXt-T | AdamW 1e-4 + stage-wise LR decay |
| `deeplabv3plus_r18` | [deeplabv3plus_r18.py](deeplabv3plus_r18.py) | ResNet-18 + ASPP | SGD 0.01 |
| `mask2former_r50` | [mask2former_r50.py](mask2former_r50.py) | ResNet-50 + Mask2Former | AdamW 1e-4 |
| `segformer_b0` | [segformer_b0.py](segformer_b0.py) | MIT-B0 | AdamW 6e-5 |
| `segnext_t` | [segnext_t.py](segnext_t.py) | MSCAN-T | AdamW 6e-5 |
| `swin_tiny` | [swin_tiny.py](swin_tiny.py) | Swin-T + UPerNet | AdamW 6e-5 |
| `twins_pcpvt_s` | [twins_pcpvt_s.py](twins_pcpvt_s.py) | PCPVT-S + UPerNet | AdamW 6e-5 |
| `unet_s5` | [unet_s5.py](unet_s5.py) | UNet-S5-D16 + FCN | SGD 0.01 |
| `efficientvit_b0` | [efficientvit_b0.py](efficientvit_b0.py) | EfficientViT-B0 | AdamW 4e-4 |

所有配置统一使用：

- 数据集：`dataset/Image` + `dataset/Mask`，`CrackDataset` 自动 7:2:1 划分；
- 输入：512×512；
- 每卡 batch size：12（脚本默认，可覆盖）；
- 迭代：40,000；
- 验证：每 1,000 iter；
- 指标：`aAcc`、`mIoU`、`mAcc`、`mDice`、`mFscore`、`mPrecision`、
  `mRecall`，以及 `IoU/crack`、`Dice/crack`、`F1/crack`、
  `Precision/crack`、`Recall/crack`；
- checkpoint：只保留 `mIoU` 最优的 `best_mIoU_iter_*.pth`
  （`save_last=False`，不写 `iter_*.pth` / `last.pth`）。

## 损失函数

抽样 500 张 mask，裂缝像素平均占比约 **1.73%**、中位数 **1.20%**，
背景:裂缝约 **57:1**。纯 CrossEntropy 会被背景主导，因此所有算法的主分割
损失统一为：

```text
CrossEntropyLoss  0.5
DiceLoss          1.0
```

具体实现：

- ConvNeXt-T、DeepLabV3+ R18、SegFormer B0、SegNeXt-T、Swin-T、
  Twins PCPVT-S、UNet-S5：`decode_head.loss_decode` 为
  `[CE(0.5), Dice(1.0)]`；
- EfficientViT-B0：同样是 `[CE(0.5), Dice(1.0)]`；
- Mask2Former-R50：使用其解码器原生损失，缩放为
  `loss_cls=1.0`、`loss_mask=0.5`、`loss_dice=1.0`；
- UPerNet / FCN 的辅助头保持 `CrossEntropyLoss=0.4`，作为深监督，
  不参与主损失的统一。

CE 均保留 `avg_non_ignore=True`；Dice 使用 `use_sigmoid=False` 和
`ignore_index=255`，适配 2 类 softmax 输出。

## 预训练权重

除 UNet 外，所有配置都会加载原始 MMSeg 配置中的预训练权重：

| 算法 | 初始化来源 |
| --- | --- |
| ConvNeXt-T | `convnext-tiny_..._in1k` ImageNet 权重（`prefix='backbone.'`） |
| DeepLabV3+ R18 | `open-mmlab://resnet18_v1c` |
| Mask2Former R50 | `torchvision://resnet50` |
| SegFormer B0 | `mit_b0_20220624-7e0fe6dd.pth` |
| SegNeXt-T | `mscan_t_20230227-119e8c9f.pth` |
| Swin-T | `swin_tiny_patch4_window7_224_20220317-1cdeb081.pth` |
| Twins PCPVT-S | `pcpvt_small_20220308-e638c41c.pth` |
| EfficientViT-B0 | 官方 Cityscapes 分割 checkpoint，转换后加载 |
| UNet-S5 | 原始配置没有预训练 URL，从零训练 |

EfficientViT-B0 的分割头是 19 类，裂纹是 2 类。转换脚本会加载 backbone 和
SegHead 融合层，跳过 19 类分类器，`conv_seg` 保持随机初始化：

```bash
python scripts/prepare_efficientvit_b0.py
```

`batch_train.py` 在启动 EfficientViT 前会自动检查并执行该转换；预训练权重
第一次下载后缓存在 `torch.hub`。所有预训练权重第一次加载都需要网络。

## 运行

```powershell
# Windows：使用项目环境，GPU 0 和 GPU 1
.venv-mmseg\python.exe scripts\batch_train.py --gpus 0,1

# 或者双击 / cmd
scripts\batch_train.bat --gpus 0,1
```

```bash
# Linux
./.venv-mmseg/bin/python scripts/batch_train.py --gpus 0,1
```

启动后前两个算法分别在 GPU0 / GPU1 上训练，任一结束就启动队列中的下一个。

## 常用参数

| 参数 | 含义 |
| --- | --- |
| `--gpus 0,1` | 使用的 GPU，默认 `0,1` |
| `--batch-size 12` | 每卡 batch，默认 12；`0` 表示沿用配置 |
| `--no-amp` | 关闭所有算法的 AMP（默认仅 Mask2Former 使用 fp32） |
| `--only convnext_tiny,unet_s5` | 只跑指定算法 |
| `--skip mask2former_r50` | 跳过指定算法 |
| `--stop-on-failure` | 某个算法失败后停止整个队列 |
| `--log-interval N` | 每 N iter 记录一次 loss；默认 50，`1` 为逐 iter |
| `--no-stream-logs` | 不把子进程日志实时打印到终端（默认实时打印） |
| `--dry-run` | 只打印命令，不训练 |
| `--list` | 列出算法和配置 |
| `--work-root work_dirs/crack_batch` | 工作目录根 |
| `--poll 10` | 轮询间隔秒数 |

示例：

```powershell
# 先只验证两个模型
.venv-mmseg\python.exe scripts\batch_train.py --gpus 0,1 --only unet_s5,efficientvit_b0

# 显存取不下 12 时，改成每卡 2
.venv-mmseg\python.exe scripts\batch_train.py --gpus 0,1 --batch-size 2
```

## 输出

```text
work_dirs/crack_batch/
├── batch_summary.json              # 每个 run 的状态、耗时、日志、返回码
├── logs/
│   ├── convnext_tiny_20261019_110000.log
│   └── ...
├── convnext_tiny/
│   ├── best_mIoU_iter_XXXX.pth     # 只保留这一个 checkpoint
│   └── <timestamp>/                # mmseg 的日志 / scalars
├── deeplabv3plus_r18/
│   └── best_mIoU_iter_XXXX.pth
└── ...
```

训练日志写到 `work_dirs/crack_batch/logs/`；mmseg 自己的日志和 scalars 在
各 run 的 `<timestamp>/` 下。最优 checkpoint 直接位于
`work_dirs/crack_batch/<name>/best_mIoU_iter_*.pth`。

## 日志内容

每次训练会保存三份日志：

1. **调度器日志** `work_dirs/crack_batch/logs/<name>_<timestamp>.log`
   - 子进程的完整 stdout/stderr；
   - 包含 mmseg 打印的每一条 loss/学习率/验证结果；
   - 进程异常时也在这里看到完整报错。
   - 默认会把子进程日志实时打印到终端，每行带算法前缀，例如：

     ```text
     [convnext_tiny] 10/19 13:23:43 - mmengine - INFO - Iter(train) ...
     ```

     两个 GPU 的日志会交错显示，用前缀区分。只想看调度事件时加
     `--no-stream-logs`；也可以另开终端：

     ```bash
     tail -f work_dirs/crack_batch/logs/*.log
     ```
   - 某个算法失败时，脚本会自动把该日志的最后 15 行打印到终端。
2. **MMSeg 日志** `<name>/<timestamp>/<timestamp>.log`
   - 训练日志，默认每 50 iter 一行：

     ```text
     Iter(train) [50/40000]  lr: ...  eta: ...  time: ...  data_time: ...
     memory: ...  grad_norm: ...  loss: ...
     decode.loss_ce: ...  decode.loss_dice: ...  decode.acc_seg: ...
     ```

   - 验证日志包含 `aAcc`、`mIoU`、`mAcc`、`mDice`、`mFscore`、
     `mPrecision`、`mRecall`、`IoU/crack`、`Dice/crack`、`F1/crack`、
     `Precision/crack`、`Recall/crack`，以及逐类表格。
3. **结构化 scalars** `<name>/<timestamp>/vis_data/scalars.json`
   - 一行一个 JSON，训练和验证指标分开记录；
   - 适合直接画 loss / mIoU 曲线或做后处理。

`vis_data/config.py` 是本次运行的完整解析配置；`batch_summary.json` 记录
每个算法的状态、返回码、日志路径和耗时。

默认 `LoggerHook` 间隔是 50 iter；需要逐 iter 记录时：

```powershell
.venv-mmseg\python.exe scripts\batch_train.py --gpus 0,1 --log-interval 1
```

TensorBoard event 文件默认不生成（使用 `LocalVisBackend`）；如果需要在
TensorBoard 中看曲线，可以再加 `TensorboardVisBackend`。

## 已知训练问题

- **Mask2Former-R50**：fp16 AMP 下 mask CE 会出现 `NaN`，随后 Hungarian
  assigner 报 `ValueError: cost matrix is infeasible`。`batch_train.py`
  会自动对 `mask2former_r50` 不传 `--amp`，即用 fp32 训练；其他算法仍用
  AMP。单独运行 `tools/train.py` 时也不要加 `--amp`。
- **UNet-S5**：backbone 要求输入宽高都能被 16 整除。验证/测试的图会按
  `Resize(keep_ratio=True)` 变成类似 `512×910`，因此 `test_pipeline` 里
  已加 `Pad(size_divisor=16)`，会 pad 到 `512×912`；`seg` 的 padding 值
  是 255，指标会自动忽略。其他模型不受影响。

> 注意：`convnext_tiny` 和 `mask2former_r50` 依赖 `mmpretrain` / `mmdet`，
> 一键环境脚本已经安装；手动安装时执行
> `pip install mmpretrain==1.2.0 mmdet==3.3.0`。

> 注意：512×512、batch 12 对 Mask2Former-R50、Swin-T 等模型显存要求较高，
> 8GB 显卡很可能 OOM。调度脚本会把失败记录下来并继续下一个算法；需要重跑
> 时用 `--only` 和较小的 `--batch-size`。
