# EfficientViT + MMSegmentation：语义分割与裂纹分割

本仓库基于 **MMSegmentation 1.2.2**，将
[EfficientViT](https://github.com/mit-han-lab/efficientvit)
（以及分割分支 [CVHub520/efficientvit](https://github.com/CVHub520/efficientvit)）
的模型代码完整移植到 MMSeg 中，并补充了 Cityscapes、ADE20K 和二分类裂纹分割的配置与训练脚本。

主要改动：

- 纯 PyTorch 实现 EfficientViT 骨干（B0–B3、L0–L2），不依赖外部 `efficientvit` 包；
- 实现官方 `SegHead`，可复现官方分割模型的输出；
- 新增 `EfficientViT`、`EfficientViTHead`、`CrackDataset` 三个注册组件；
- 新增 14 个实验配置，以及 Windows / Linux 一键环境脚本；
- 参数名与官方 checkpoint 一一对应，可按规则转换后加载。

---

## 1. 环境安装

### 1.1 一键安装（推荐）

Windows（PowerShell）：

```powershell
# CUDA 11.8，默认安装到 .\.venv-mmseg
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1

# 纯 CPU
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Cpu

# 重建环境 + 安装测试依赖
powershell -ExecutionPolicy Bypass -File .\scripts\setup_windows.ps1 -Force -WithTests
```

也可以直接双击 `scripts\setup_windows.bat`。

Linux：

```bash
# CUDA 11.8
bash scripts/setup_linux.sh

# 纯 CPU
bash scripts/setup_linux.sh --cpu

# 重建环境 + 安装测试依赖
bash scripts/setup_linux.sh --force --with-tests
```

脚本会创建 `./.venv-mmseg`，安装以下固定版本：

| 组件 | 版本 |
| --- | --- |
| Python | 3.10 |
| PyTorch / torchvision | 2.1.0 / 0.16.0（cu118 或 cpu） |
| MMCV | 2.1.0（预编译 wheel） |
| MMEngine | 0.10.7 |
| MMDetection / MMPretrain | 3.3.0 / 1.2.0（ConvNeXt、Mask2Former） |
| setuptools | `<81`（`torch.utils.cpp_extension` 仍依赖 `pkg_resources`） |
| MMSegmentation | 本仓库，editable 安装 |
| NumPy | 1.26.4 |
| OpenCV | 4.8.1.78 |

安装完成后自动执行 `tools/verify_env.py`。更多参数见
[scripts/README.md](scripts/README.md)。

环境位于仓库根目录的 `./.venv-mmseg`，**使用前需要激活**，见下一节。

### 1.2 激活环境

Windows（PowerShell / Anaconda Prompt）：

```powershell
conda activate .\.venv-mmseg

# 或使用绝对路径
conda activate F:\UTS_work_code\mmsegmentation\.venv-mmseg
```

Linux：

```bash
conda activate ./.venv-mmseg
```

激活成功后，命令行提示符前会出现 `(.venv-mmseg)`。可以用下面的命令确认
当前 Python 确实来自该环境：

```bash
python -c "import sys; print(sys.executable)"
```

如果提示 `conda: 无法将“conda”项识别为 ...`，或提示
`Your shell has not been properly configured to use 'conda activate'`：

```powershell
# Windows PowerShell
conda init powershell
# 关闭并重新打开终端后，再执行 conda activate
```

```bash
# Linux bash
conda init bash
source ~/.bashrc
```

如果安装时没有找到 conda、脚本回退到了 Python `venv`，激活方式不同：

```powershell
# Windows PowerShell
.\.venv-mmseg\Scripts\Activate.ps1

# Windows cmd
.\.venv-mmseg\Scripts\activate.bat
```

```bash
# Linux
source ./.venv-mmseg/bin/activate
```

不想激活也可以直接使用环境里的解释器：

```powershell
# Windows
.\.venv-mmseg\python.exe tools\verify_env.py
.\.venv-mmseg\python.exe scripts\batch_train.py --gpus 0,1
```

```bash
# Linux
./.venv-mmseg/bin/python tools/verify_env.py
./.venv-mmseg/bin/python scripts/batch_train.py --gpus 0,1
```

### 1.3 手动安装

手动安装下面使用名为 `mmseg` 的 conda 环境；一键脚本使用的是
`./.venv-mmseg`，两者选一种即可。

```bash
conda create -n mmseg python=3.10 -y
conda activate mmseg

pip install --upgrade pip wheel "setuptools<81"
pip install numpy==1.26.4 opencv-python==4.8.1.78
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install mmengine==0.10.7
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1.0/index.html
pip install mmdet==3.3.0 mmpretrain==1.2.0
pip install -r requirements/runtime.txt
pip install -e . --no-deps --no-build-isolation
```

> `setuptools>=81` 已移除 `pkg_resources`，而 PyTorch 的
> `torch.utils.cpp_extension` 仍会导入它，所以这里固定 `setuptools<81`。
> 如果环境已经是新版 setuptools，执行
> `pip install "setuptools<81"` 即可。

CPU 环境把 PyTorch 和 MMCV 的索引换成：

```text
https://download.pytorch.org/whl/cpu
https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html
```

### 1.4 验证环境

```bash
python tools/verify_env.py
```

正常输出示例：

```text
python     : 3.10.22
numpy      : 1.26.4
torch      : 2.1.0+cu118
mmcv       : 2.1.0
mmengine   : 0.10.7
mmseg      : 1.2.2 (F:\UTS_work_code\mmsegmentation\mmseg)
cuda avail : True
cuda       : 11.8 | NVIDIA GeForce RTX 4060 Laptop GPU
EfficientViT forward : OK (1, 2, 8, 8)
config build         : OK configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py
dataset train      : 4230 samples
dataset val        : 1208 samples
dataset test       : 604 samples

ENVIRONMENT_OK
```

没有 `dataset/` 时会自动跳过数据集检查；也可以加 `--skip-config` 或
`--skip-dataset`。

---

## 2. 数据准备

裂纹分割数据按下面的结构放置：

```text
dataset/
├── Image/            # 裂纹图像，.jpg
└── Mask/             # 二值 mask，.png
```

要求：

- `Image/` 与 `Mask/` 按主文件名（stem）配对，例如
  `Image/1946_Crack_1.jpg` ↔ `Mask/1946_Crack_1.png`；
- mask 为灰度二值图，`0` 表示背景，`1` 或 `255` 表示裂纹；
- 图像和 mask 尺寸一致；支持平铺或子目录结构。

`CrackDataset` 会自动完成：

1. 按 stem 配对，校验缺失/多余文件；
2. 使用固定 seed=42 做 7:2:1 划分，**无需提前生成 split 文件**；
3. 读取 mask 时强制转灰度，并把 `255` 映射为前景类 `1`，避免被当成
   `ignore_index=255`；
4. 训练时使用 `RandomResize` + `RandomCrop(512×512)` + 翻转 + 光度增强。

当前 `dataset/` 下 6042 对样本的划分结果为：

```text
train: 4230    val: 1208    test: 604
```

如果需要使用自己的显式划分，可以给 `CrackDataset` 传 `ann_file`，每行一个
图像 stem（不带扩展名），此时 `split` 参数不再生效。

---

## 3. 训练

### 3.1 裂纹分割

单卡：

```bash
python tools/train.py \
  configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py \
  --work-dir work_dirs/crack_b0
```

4 卡（配置中的 3 是**每卡** batch size，总 batch 为 12）：

```bash
bash tools/dist_train.sh \
  configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py 4
```

恢复训练 / 混合精度：

```bash
python tools/train.py configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py \
  --work-dir work_dirs/crack_b0 --resume

python tools/train.py configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py \
  --work-dir work_dirs/crack_b0 --amp
```

可用配置：

| 配置 | 骨干 | 输入 | 轮数 |
| --- | --- | --- | --- |
| [efficientvit-b0_4xb3-30k_crack-512x512.py](configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py) | B0 | 512×512 | 30k |
| [efficientvit-b2_4xb3-30k_crack-512x512.py](configs/efficientvit/efficientvit-b2_4xb3-30k_crack-512x512.py) | B2 | 512×512 | 30k |
| [efficientvit-l1_4xb3-30k_crack-512x512.py](configs/efficientvit/efficientvit-l1_4xb3-30k_crack-512x512.py) | L1 | 512×512 | 30k |

训练配置沿用参考实现的裂纹 recipe：AdamW（lr 4e-4，weight decay 0.01）、
1500 iter warmup、poly 衰减、梯度裁剪 1.0、CE + Dice loss。验证指标为
`mIoU` 和 `mDice`，并按 `mIoU` 保存最优 checkpoint。

### 3.2 Cityscapes / ADE20K

```bash
# Cityscapes，8 卡，1024×1024，160k
bash tools/dist_train.sh \
  configs/efficientvit/efficientvit-b1_8xb2-160k_cityscapes-1024x1024.py 8

# ADE20K，8 卡，512×512，160k
bash tools/dist_train.sh \
  configs/efficientvit/efficientvit-b1_8xb2-160k_ade20k-512x512.py 8
```

配置列表：

| 数据集 | 配置 |
| --- | --- |
| Cityscapes | `efficientvit-{b0,b1,b2}_8xb2-160k_cityscapes-1024x1024.py` |
| Cityscapes | `efficientvit-{b3,l1,l2}_8xb1-160k_cityscapes-1024x1024.py` |
| ADE20K | `efficientvit-{b1,b2,b3,l1,l2}_8xb2-160k_ade20k-512x512.py` |

Cityscapes / ADE20K 使用 AdamW（lr 1e-3，weight decay 0.01）、1500 iter
warmup、poly 160k、CE + Dice loss。学习率和 batch size 是按 MMSeg 习惯做的
适配，官方训练代码未公开，因此不能保证逐点复现论文指标。

显存不足时可降低每卡 batch：

```bash
python tools/train.py configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py \
  --cfg-options train_dataloader.batch_size=1
```

### 3.3 双卡批量训练 9 个算法

[scripts/batch_train.py](scripts/batch_train.py) 按队列训练 9 个轻量模型，
每个 GPU 同一时间只跑一个；任一结束，下一个自动顶上。

```powershell
# Windows，GPU 0 + GPU 1
.venv-mmseg\python.exe scripts\batch_train.py --gpus 0,1

# 或
scripts\batch_train.bat --gpus 0,1
```

```bash
# Linux
./.venv-mmseg/bin/python scripts/batch_train.py --gpus 0,1
```

队列：

```text
convnext_tiny → deeplabv3plus_r18 → mask2former_r50 → segformer_b0
→ segnext_t → swin_tiny → twins_pcpvt_s → unet_s5 → efficientvit_b0
```

所有配置在 [configs/crack_batch](configs/crack_batch)，统一使用：
CrackDataset、512×512、每卡 batch 12、40k iter、每 1000 iter 验证、
只保留按 `mIoU` 最优的 `best_mIoU_iter_*.pth`（不保存 `last.pth` 和
`iter_*.pth`）。

预训练方面：除 UNet（原始配置没有预训练 URL）外，其余算法都会加载原始
MMSeg 配置中的 ImageNet / 骨干权重。EfficientViT-B0 使用官方 Cityscapes
分割 checkpoint，首次运行 `batch_train.py` 会自动下载并按键名规则转换
（19 类分类器跳过，`conv_seg` 随机初始化）；也可手动执行：

```bash
python scripts/prepare_efficientvit_b0.py
```

损失函数方面，针对裂缝像素占比低（抽样均值约 1.73%，背景:裂缝约 57:1），
9 个算法的主分割损失统一为 **CE 0.5 + Dice 1.0**；Mask2Former 按其原生
mask 损失缩放为 `loss_cls=1.0`、`loss_mask=0.5`、`loss_dice=1.0`，
UPerNet/FCN 辅助头保留 CE 0.4。这样既保留稳定的像素级梯度，又让 Dice
直接优化前景重叠。

常用参数：

```bash
# 只跑其中两个
.venv-mmseg\python.exe scripts\batch_train.py --gpus 0,1 --only unet_s5,efficientvit_b0

# 显存不足时改为每卡 2
.venv-mmseg\python.exe scripts\batch_train.py --gpus 0,1 --batch-size 2

# 先看命令
.venv-mmseg\python.exe scripts\batch_train.py --gpus 0,1 --dry-run
```

输出在 `work_dirs/crack_batch/`：`logs/` 为每个算法的完整日志和 loss，
`<name>/<timestamp>/vis_data/scalars.json` 为逐条结构化指标，
`batch_summary.json` 为状态汇总，各算法子目录下保存最优 checkpoint。
默认每 50 iter 记录一次 loss；`--log-interval 1` 可逐 iter 记录。

批量脚本默认把每个算法的训练日志实时打印到终端（每行带
`[算法名]` 前缀），同时写入日志文件；不想在终端看时加
`--no-stream-logs`。也可以另开终端：

```bash
tail -f work_dirs/crack_batch/logs/*.log
```

某个算法失败时，脚本会自动打印该日志最后 15 行到终端。
详细说明见 [configs/crack_batch/README.md](configs/crack_batch/README.md)。

---

## 4. 测试与推理

### 4.1 评估

```bash
python tools/test.py \
  configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py \
  work_dirs/crack_b0/best_mIoU_iter_30000.pth \
  --work-dir work_dirs/crack_b0/eval
```

多卡评估：

```bash
bash tools/dist_test.sh \
  configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py \
  work_dirs/crack_b0/best_mIoU_iter_30000.pth 4
```

### 4.2 单张图像可视化

```bash
python demo/image_demo.py \
  dataset/Image/1946_Crack_1.jpg \
  configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py \
  work_dirs/crack_b0/best_mIoU_iter_30000.pth \
  --out-file work_dirs/crack_b0/pred.png \
  --device cuda:0
```

`image_demo.py` 输出的是叠加可视化结果；批量评估和指标请使用
`tools/test.py`。

### 4.3 裂纹评估指标

裂纹配置使用 [CrackIoUMetric](mmseg/evaluation/metrics/crack_metric.py)，
评估结果同时包含宏平均和裂缝前景标量：

| 指标 key | 含义 |
| --- | --- |
| `aAcc` | 像素总准确率 |
| `mIoU` / `mDice` / `mFscore` | background + crack 的宏平均 IoU / Dice / F1 |
| `mPrecision` / `mRecall` | background + crack 的宏平均 Precision / Recall |
| `mAcc` | 类别平均准确率 |
| `IoU/crack` | 裂缝前景 IoU |
| `Dice/crack` | 裂缝前景 Dice |
| `F1/crack` | 裂缝前景 F1 |
| `Precision/crack` | 裂缝前景 Precision |
| `Recall/crack` | 裂缝前景 Recall |

日志还会打印逐类表格（background / crack 的 IoU、Acc、Dice、Fscore、
Precision、Recall）。注意：`mIoU`、`mDice` 等是 **两个类别的宏平均**，
背景占比大时会偏高，裂缝质量应重点看 `IoU/crack`、`Dice/crack`、
`F1/crack`。

默认最优模型按 `mIoU` 保存；如果希望按裂缝前景 IoU 保存：

```bash
python tools/train.py \
  configs/efficientvit/efficientvit-b0_4xb3-30k_crack-512x512.py \
  --work-dir work_dirs/crack_b0 \
  --cfg-options default_hooks.checkpoint.save_best=IoU/crack
```

---

## 5. 本次新增/修改内容

### 5.1 算法实现

| 文件 | 内容 |
| --- | --- |
| [mmseg/models/backbones/efficientvit/layers.py](mmseg/models/backbones/efficientvit/layers.py) | `LiteMLA`、`MBConv`、`FusedMBConv`、`DSConv`、`ResBlock`、`ResidualBlock`、`OpSequential` 等基础模块 |
| [mmseg/models/backbones/efficientvit/backbone.py](mmseg/models/backbones/efficientvit/backbone.py) | `EfficientViT` 骨干，支持 `b0`–`b3`、`l0`–`l2` |
| [mmseg/models/decode_heads/efficientvit_head.py](mmseg/models/decode_heads/efficientvit_head.py) | `EfficientViTHead`，官方 `SegHead` 的 MMSeg 实现 |
| [mmseg/models/backbones/__init__.py](mmseg/models/backbones/__init__.py) | 注册 `EfficientViT` |
| [mmseg/models/decode_heads/__init__.py](mmseg/models/decode_heads/__init__.py) | 注册 `EfficientViTHead` |

骨干输出 4 个特征图，stride 分别为 4、8、16、32，通道数为
`width_list[1:]`。`EfficientViTHead` 默认：

```python
in_index=(3, 2, 1)      # stage4, stage3, stage2
strides=(32, 16, 8)
in_channels=[C4, C3, C2]
```

这与官方 `SegHead` 的参数顺序一致。若要用 FCN / UPerNet 等标准头，则改为
`out_indices=(0,1,2,3)`、`in_index=[0,1,2,3]`、
`in_channels=[C1,C2,C3,C4]`。

### 5.2 配置

| 目录 | 内容 |
| --- | --- |
| [configs/_base_/models/efficientvit-b0.py](configs/_base_/models/efficientvit-b0.py) … `l2.py` | 6 个模型基础配置（Cityscapes 头参数） |
| [configs/_base_/datasets/crack.py](configs/_base_/datasets/crack.py) | 裂纹数据 pipeline |
| [configs/efficientvit](configs/efficientvit) | 14 个实验配置 + 训练 schedule |
| [configs/efficientvit/README.md](configs/efficientvit/README.md) | EfficientViT 集成细节、参数表、checkpoint 键名映射 |
| [configs/crack_batch](configs/crack_batch) | 9 个轻量算法的裂纹批量训练配置（40k、batch 12、仅保存最优） |

### 5.3 数据集与 transform

- [mmseg/datasets/crack.py](mmseg/datasets/crack.py)：`CrackDataset`，
  自动 7:2:1 划分、`255 → 1`、灰度 mask 读取；
- [mmseg/datasets/__init__.py](mmseg/datasets/__init__.py)：注册
  `CrackDataset`；
- [mmseg/datasets/transforms/loading.py](mmseg/datasets/transforms/loading.py)：
  `LoadAnnotations` 新增可选 `imdecode_flag` 参数（默认行为不变），
  裂纹配置使用 `imdecode_flag='grayscale'` 读取 RGB 编码的灰度 mask。

### 5.4 工具、脚本与测试

| 文件 | 内容 |
| --- | --- |
| [scripts/setup_windows.ps1](scripts/setup_windows.ps1) / [.bat](scripts/setup_windows.bat) | Windows 一键环境 |
| [scripts/setup_linux.sh](scripts/setup_linux.sh) | Linux 一键环境 |
| [scripts/batch_train.py](scripts/batch_train.py) / [.bat](scripts/batch_train.bat) | 双卡批量训练 9 个算法 |
| [tools/verify_env.py](tools/verify_env.py) | 环境、模型、配置、数据集自检 |
| [mmseg/evaluation/metrics/crack_metric.py](mmseg/evaluation/metrics/crack_metric.py) | `CrackIoUMetric`：宏平均 + 裂缝前景指标 |
| [tests/test_models/test_backbones/test_efficientvit.py](tests/test_models/test_backbones/test_efficientvit.py) | 骨干输出形状/stride 测试 |
| [tests/test_models/test_heads/test_efficientvit_head.py](tests/test_models/test_heads/test_efficientvit_head.py) | 分割头前向、loss 测试 |
| [tests/test_evaluation/test_metrics/test_crack_metric.py](tests/test_evaluation/test_metrics/test_crack_metric.py) | 裂纹指标数值测试 |

运行单元测试：

```bash
pytest tests/test_models/test_backbones/test_efficientvit.py \
       tests/test_models/test_heads/test_efficientvit_head.py \
       tests/test_evaluation/test_metrics/test_crack_metric.py -q
```

---

## 6. 与官方实现的一致性

在同一输入、加载官方 `state_dict` 后对比（CPU / float32）：

| 模型 | 最大绝对误差 |
| --- | --- |
| EfficientViT-B0 骨干 | `0.0` |
| EfficientViT-B0 完整分割模型 | `≈ 7.45e-9` |
| EfficientViT-L1 完整分割模型 | `≈ 6.6e-7` |

骨干参数名与官方完全一致；官方分割 checkpoint 的 head 参数需要按
[configs/efficientvit/README.md](configs/efficientvit/README.md) 中的规则重命名：

```text
backbone.*                          -> backbone.*
head.input_ops.*                    -> decode_head.input_ops.*
head.middle.*                       -> decode_head.middle.*
head.output_ops.0.op_list.0.*       -> decode_head.final_expand_conv.*   (final_expand 非空)
head.output_ops.0.op_list.1.*       -> decode_head.conv_seg.*            (final_expand 非空)
head.output_ops.0.op_list.0.conv.*  -> decode_head.conv_seg.*            (final_expand 为 None)
```

转换后可写入 `load_from`，或去掉 `backbone.` 前缀后作为
`backbone.init_cfg=dict(type='Pretrained', checkpoint=...)`。

---

## 7. 常见问题

**MMCV 版本不兼容**
MMSeg 1.2.2 要求 `mmcv>=2.0.0rc4,<2.2.0`。推荐固定 `mmcv==2.1.0`，并使用
预编译 wheel，避免在 Windows 上源码编译。

**NumPy 2.x 报错**
MMCV 2.1.0 的预编译 wheel 基于 NumPy 1.x。请保持
`numpy==1.26.4`；`tools/verify_env.py` 会检查这一点。

**conda 创建环境失败 / Terms of Service**
新版 conda 可能要求先接受 ToS：

```bash
conda tos accept --override-channels --channel defaults
```

或安装 Python 3.10/3.11 后让脚本走 venv 分支。

**Windows 无法运行 ps1**
使用 `powershell -ExecutionPolicy Bypass -File ...`，或直接双击
`scripts\setup_windows.bat`。

**Windows 中文环境报 GBK / UnicodeDecodeError**
这是 `mmengine.collect_env` 读取 MSVC 信息时的编码问题。运行前设置
`$env:PYTHONUTF8='1'` 即可；`scripts/batch_train.py` 已自动设置该变量。

**Linux 报 `libGL.so.1` 找不到**

```bash
sudo apt-get install -y libgl1 libglib2.0-0
```

**显存不足**
降低 `train_dataloader.batch_size`，或减小 `data_preprocessor.size`；
L1/L2 在 512×512 下建议每卡 batch ≤ 2。批量训练里 Mask2Former-R50、
Swin-T 等模型在 batch 12 / 512×512 下也可能 OOM，可用
`scripts/batch_train.py --batch-size N` 重跑。

**想用其他分割头**
骨干本身与 MMSeg 的 FCN、PSPNet、UPerNet、SegformerHead 等兼容，按 5.1 节
设置 `in_channels` / `in_index` 即可；`EfficientViTHead` 仅用于复现官方
EfficientViT 分割结构。

---

## 8. 目录结构（主要部分）

```text
mmsegmentation/
├── configs/
│   ├── _base_/models/efficientvit-*.py
│   ├── _base_/datasets/crack.py
│   └── efficientvit/                 # 14 个实验配置
├── mmseg/
│   ├── datasets/crack.py
│   ├── models/backbones/efficientvit/
│   └── models/decode_heads/efficientvit_head.py
├── scripts/                          # Windows/Linux 一键环境
├── tools/verify_env.py
├── tests/test_models/...             # EfficientViT 单测
└── dataset/
    ├── Image/
    └── Mask/
```

---

## 9. 许可证与引用

本仓库基于 [MMSegmentation](https://github.com/open-mmlab/mmsegmentation)
（Apache-2.0）修改；EfficientViT 模型代码来自
[mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit)
和 [CVHub520/efficientvit](https://github.com/CVHub520/efficientvit)
（Apache-2.0）。详见 [LICENSE](LICENSE)。

如果使用 EfficientViT，请引用：

```bibtex
@inproceedings{cai2023efficientvit,
  title={EfficientViT: Multi-Scale Linear Attention for High-Resolution Dense Prediction},
  author={Cai, Han and Li, Junyan and Hu, Muyan and Gan, Chuang and Han, Song},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  pages={17302--17313},
  year={2023}
}
```

MMSegmentation 引用：

```bibtex
@misc{mmseg2020,
  title={{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark},
  author={MMSegmentation Contributors},
  howpublished={\url{https://github.com/open-mmlab/mmsegmentation}},
  year={2020}
}
```
