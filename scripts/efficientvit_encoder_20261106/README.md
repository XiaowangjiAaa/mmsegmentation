# EfficientViT 编码器结构模块（2026-11-06）

这轮修改发生在 backbone 内部。默认只运行 baseline 与三个独立主干改进，
保留原始 stride-8 分割头；另提供组合配置用于后续验证。
这些是论文启发的研究原型，尚无本项目训练精度、部署速度或已确认原创性结论。

## 版本与参数

|版本|修改位置|分割头|整网参数|
|---|---|---|---:|
|baseline|原始 B0|原始|704,498|
|downsample|stage 1/2 入口，即 stride 4→8、8→16|原始|709,588|
|query|stage 2/3 共四个 LiteMLA|原始|713,718|
|context_local|stage 2/3 共四个 block 内部|原始|711,458|
|downsample_query|前两个机制联合|原始|718,808|
|subpixel|原始 B0|原版 subpixel|708,453|
|downsample_query_subpixel|联合主干|原版 subpixel|722,763|
|context_local_subpixel|上下文引导局部提取|原版 subpixel|715,413|

独立主干版本比原始整网增加约 0.72%、1.31%、0.99% 参数。
参数少不代表算子开销或延迟低；所有软路由分支都会计算，不能称为计算稀疏化。

## 1. 结构感知下采样

`StructureDownsample` 继承原 MBConv，保留原有卷积、归一化及参数名。
输入先经 C→8 的窄投影，再按 2×2 区域拆成四个位置（奇数边界 replicate pad）：

```text
M = mean_phase(X_p)
R_p = X_p - M
E = Conv1x1(flatten_channels_and_ordered_phases(R))
G = sigmoid(Conv1x1(concat(M, mean_phase(abs(R)))))
Y = original_stride2_MBConv(X) + tanh(scale) * G * E
```

残差不能直接求和，否则四相位相消；这里为不同位置保留不同投影权重。
scale 初值 0.01，保证开始时接近原算子且新增分支第一步有梯度。
窄投影与后续压缩是有损的，不宣称完整保存输入细节或抗混叠。
默认不改 stem、stride 2→4 和 stride 16→32。

## 2. 结构条件 LiteMLA

`StructureQueryLiteMLA` 保留原 qkv/aggreg/proj、多尺度分支以及 ReLU 归一化读取。
从当前 stage 特征生成窄结构描述，而不是从 decoder 回传特征：

```text
D = Conv1x1_C_to_8(X)
R = D - replicate_border_average3x3(D)
Q'_scale,head = Q_scale,head + tanh(scale) * W_structure(R)
K' = K; V' = V
Y = original_multiscale_linear_attention(Q', K, V)
```

更新作用于每个尺度和 head 的 Query 向量，发生在 ReLU 特征映射之前；
不是给整个 Q 乘一个会在归一化分子分母约掉的标量。
K/V 构建与语义全局状态保留原定义，不重复引入此前收益不足的残差全局记忆写入。
仍然存在背景混淆、过平滑或 ReLU 抑制结构更新的可能，必须通过训练和特征诊断验证。

## 3. 上下文引导局部提取

`ContextGuidedLocalBlock` 保留原 LiteMLA 与 MBConv，在 block 内增加窄局部修正：

```text
C = original_LiteMLA(X)
Z = X + C
L = reduce(Z); S = reduce(C)                    # 共享窄投影
w = softmax(router(concat(L, S)), expert_axis)  # 每个位置一组权重
E = w_h * DW1x5(L) + w_v * DW5x1(L) + w_l * DW3x3(L)
Y = Z + original_MBConv(Z) + tanh(scale) * expand(E)
```

这里的 C 是 LiteMLA 生成的上下文增量，路由同时依赖局部特征和全局读取结果。
三个专家分别处理水平、垂直和二维邻域；3×3 专家也能学习斜向模式，但没有显式斜向分支。
MBConv 未被替换，当前原型属于上下文条件局部增强，未宣称比重构整个 MBConv 更高效。
残差只由各层各加一次，避免历史 CRA 中内外重复 identity 的问题。

## 论文联系与研究定位

- [RALA，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fan_Breaking_the_Low-Rank_Dilemma_of_Linear_Attention_CVPR_2025_paper.html)：借鉴 KV/输出多样性的问题意识；本轮并非 RALA 复现，也未证明当前 LiteMLA 的有效秩是主瓶颈。
- [MALA，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Rectifying_Magnitude_Neglect_in_Linear_Attention_ICCV_2025_paper.html)：提醒关注归一化的信息损失；本轮没有重复幅值修正，Query 更新为结构向量。
- [OverLoCK，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/papers/Lou_OverLoCK_An_Overview-first-Look-Closely-next_ConvNet_with_Context-Mixing_Dynamic_Kernels_CVPR_2025_paper.pdf)：借鉴上下文指导局部提取；这里只实现窄通道位置路由，不复现其网络与动态核。
- 本项目 subpixel：把结构条件交互前移到 encoder；不是把原 decoder 模块直接复制四遍。

以上引用解释设计来源，不构成新颖性证明。与相关下采样、条件卷积和条件注意力工作的区别仍需进一步检索与验证。

## 运行

在仓库根目录执行。继承已有 dataset、预训练、CE 0.5 + Dice 1.0、AdamW、40k 协议。
默认 fp32、batch size 12、seeds 42/43/44、每 1000 iter 验证。

```bash
# 全部 8 个配置的真实分割 CE+Dice 前后向检查（随机数据，非准确率）
python scripts/efficientvit_encoder_20261106/verify.py --size 512

# 列出配置 / 预览默认训练任务
python scripts/efficientvit_encoder_20261106/batch_train.py --list
python scripts/efficientvit_encoder_20261106/batch_train.py --dry-run

# 第一阶段：固定原始 head，比较三个独立主干机制，共 12 次训练
python scripts/efficientvit_encoder_20261106/batch_train.py --gpus 0,1

# 一次运行全部 8 配置，共 24 次训练，包含联合机制与 subpixel 对照
# 与上面的命令二选一；相同 work-root 不允许覆盖既有运行。
python scripts/efficientvit_encoder_20261106/batch_train.py --gpus 0,1 --variants all

# 第一阶段结束后，单独补联合方案与 subpixel 组合；可复用同轮已有对照
python scripts/efficientvit_encoder_20261106/batch_train.py --gpus 0,1 --variants downsample_query,subpixel,downsample_query_subpixel,context_local_subpixel --work-root work_dirs/efficientvit_encoder_followup_20261106

# 从零训练对照：所有选中版本统一关闭预训练
python scripts/efficientvit_encoder_20261106/batch_train.py --gpus 0,1 --scratch

# 重算同轮同 seed 指标；不要将缺少对照的 followup 单独作为成对结论
python scripts/efficientvit_encoder_20261106/summarize.py work_dirs/efficientvit_encoder_20261106/pretrained/fp32
```

Windows 若 YAPF 缓存写入阻塞，verify.py 可加 `--no-yapf-cache`，仅当前进程禁用缓存。
训练要求数据集和可用 CUDA；本次没有启动正式训练。
复用共享 runner，按初始化/精度/seed/版本隔离目录，不覆盖历史结果。
汇总保留权威 batch_summary.json，指标重算写 metrics_summary.json；进程完成情况须另核对。
有同 seed 对照时会输出相对 baseline/subpixel 的差值，不能用旧轮次分数充当本轮对照。

## 预训练兼容与检查

原主干与原 head 的参数名和形状保留。`load_from` 继续使用已转换的 Cityscapes 权重；
新增 `structure_*` 参数随机初始化，出现对应 missing keys 属预期，不能使用旧 checkpoint resume 新结构。
转换权重原本就跳过了 Cityscapes 的 19 类分类器，因此二分类 `conv_seg` 也随机初始化；
subpixel 配置额外的 coarse_projection 同样是新增参数。
不应忽略上述预期之外的参数缺失或任何尺寸不匹配。关闭所有新增 scale 后，使用相同原权重的算子与原函数一致。
当前原型限制为 B 系列；L 系列选择新结构会明确报错。

本次验证：8 个配置均通过 64×64 和 512×512 的真实 CE+Dice 前后向；
37 项本轮与已有 CRA/信息交互/结构头/subpixel/runner 回归检查通过。
实际转换的预训练 checkpoint 的 228 个参数/缓冲键均成功加载到全部 8 个配置，
没有 unexpected keys 或尺寸不匹配，仅上述预期新增参数及二分类器缺失。

新增 unittest 检查：四相位分解/重建与非抵消、奇数/单像素尺寸、只改 Query 不改 K/V、
零 scale 兼容、上下文确实改变路由、新增参数 fp32/CPU bf16 非零有限梯度、
原主干参数键兼容、stage 接入位置、四尺度输出，以及训练脚本与固定 head 协议。

尚无正式训练、独立测试集、拓扑指标、FLOPs 或目标硬件延迟结果。
后续评估应先判断三个独立机制是否优于同轮 baseline，再判断与 subpixel 是否互补。
