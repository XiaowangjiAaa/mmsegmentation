# LiteMLA 核心聚合修改（2026-11-12）

本轮直接替换 EfficientViT-B0 四个 LiteMLA 的记忆构建/读取公式：stride16 两处、stride32 两处。
原 Q/K/V、原 5×5 多尺度聚合及输出投影的参数名/形状保留；周边 MBConv、原始下采样和原分割头固定。
没有附加输出修正分支，没有与此前 adaptive 下采样或 subpixel 叠加。
两类方案均为待验证研究原型，尚无正式训练、新颖性或精度/延迟收益结论。

## 1. 结构关系驱动的全局记忆

原版每个 head 用一个压缩 KV 状态。本轮使用两个状态，并让同一位置的路由同时参与写入与读取。
路由在同一 LiteMLA 的 heads/多尺度之间共享，语义 Q/K/V 仍分别计算。

```text
D = Conv1x1_C_to_8(X)
M = replicate_border_mean3x3(D)
structure features = concat(D, D-M, abs(D-M))
a = softmax(Conv1x1_24_to_2(features), bank_axis)

A_r = sum_j a_jr * ReLU(K_j) V_j^T
z_r = sum_j a_jr * ReLU(K_j)
Y_i = sum_r a_ir ReLU(Q_i)^T A_r /
      (sum_r a_ir ReLU(Q_i)^T z_r + eps)
```

它对应的非负关系核为：
`score_ij = ReLU(Q_i)^T ReLU(K_j) * dot(a_i, a_j)`。
因此改动位于哪些位置共享全局信息的规则，而非单独提高某个 Key 权重。
整个读取只有一次联合归一化，不是分别归一化两个状态再相减/相加。
两个状态不预设为裂缝/背景，也不使用占用率或分割辅助监督；路由可能退化，必须诊断。

对照：

- bank_content：特征为 concat(D, M, D*M)，投影宽度/参数与 structure 完全相同，用于检查结构差异表征的必要性。
- bank_uniform：a 固定为 (0.5,0.5)，无新参数。所有位置的路由相同时，读取数学上还原原版全局线性注意力（浮点/eps 差异除外）。

保持关于 token 数的线性复杂度，但两个状态增大统计与读取计算。增加状态只扩大表达可能性，
不保证实际有效秩提高。此前 decoder partition 使用多记忆并未成功，因此不能仅凭“多个记忆”认定创新或收益。
[RALA，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Fan_Breaking_the_Low-Rank_Dilemma_of_Linear_Attention_CVPR_2025_paper.html)
为关注 KV/输出表达能力提供参考；本模块不是 RALA 复现，其结构关系核是否有效需要独立证据。

## 2. 局部与全局补集共同归一化

使用相同 ReLU(Q)/ReLU(K)/V，同时计算有效 3×3 邻域和其余所有位置。
先计算全图压缩统计，再减去邻域统计构成全局补集，避免重复计入局部位置。

设 Nl/Dl 是局部的加权值/注意力质量，Ng/Dg 是全局补集的加权值/质量，nl/ng 是有效位置数：

```text
lambda_i,head = softplus(Conv1x1_24_to_packed_heads(structure features))
Y_i = (Nl/nl + lambda*Ng/ng) / (Dl/nl + lambda*Dg/ng + eps)
```

按位置数平衡两类统计，避免全局项仅凭位置数量压过局部项；实际权重仍由 Q/K 的匹配质量参与决定。
共同归一化改变注意力分配，而不是在原输出后加局部分支。这里不采用局部 softmax，局部与全局都用同一 ReLU 关系核。
边界只统计有效邻居；全局补集为空时强制其统计为零，退化为局部读取。

对照：

- joint_fixed：lambda=1，不增加参数，检查学习的位置/head 条件是否必要。
- postfusion：与 joint 完全相同的路由参数，但分别归一化局部和全局，再按 lambda/(1+lambda) 混合。
  全局补集为空时只用局部输出。它用于判断共同归一化是否优于普通输出融合。

当 lambda=ng/nl，joint 公式数学上还原原版全图读取；这也是归一化/去重检查之一。
若存在 Q/K 全零，输出为零并保持有限。统计累积、局部/全局相减及除法使用 float32。
相减仍可能存在浮点误差，global mass 仅对数值负误差钳制到零；不能宣称任意输入下严格代数精确。

## 版本与预算

|版本|作用|整网参数|
|---|---|---:|
|baseline|原 B0|704,498|
|bank_structure|结构关系双记忆|707,770|
|bank_content|等参数内容路由对照|707,770|
|bank_uniform|原注意力等价控制|704,498|
|joint|学习局部/全局共同归一化|708,770|
|joint_fixed|固定 lambda 的共同归一化|704,498|
|postfusion|等参数分别归一化输出融合|708,770|

bank 主版本比基线多约 0.46%，joint 多约 0.61%。
代码不构造空间 N×N 矩阵；单元测试仅在小网格显式计算该矩阵作为数学 oracle。
bank 增加状态，joint/postfusion 通过 unfold 展开局部 K/V，均可能增加激活、访存和延迟。
目前没有实际部署成本结果，不能以参数少或 O(N) 宣称低延迟。

## 初始化与预训练

本轮默认 scratch。CPU 构建模型时，新增权重在隔离 RNG 上下文中初始化，原 attention 参数从原 block 拷贝。
同一 seed 下已有 backbone/head 参数和构建后的 CPU RNG 状态与 baseline 对齐；新路由保持随机、非零初始化。
这有助于减少比较中的初始化差异，不保证 CUDA 训练跨运行逐位复现。

原转换权重 `work_dirs/pretrained/efficientvit_b0_crack.pth` 的 228 个参数/缓冲键在全部版本均可直接加载，
无需转换或丢弃已有权重；新增 routing_* 随机初始化，二分类 conv_seg 依旧随机初始化。
预训练参数兼容不等于改变后的注意力函数等价；旧训练 checkpoint 不可用于新模式直接 resume。

## 训练与汇总

固定同一数据、原始 head、CE 0.5 + Dice 1.0、AdamW lr=0.0004、batch=12、fp32、
40k、warmup=1500、每 1000 iter 验证、seeds 42/43/44。默认全部 7 配置，共 21 次训练。

```bash
# 随机数据上的真实分割损失/反向检查，不是精度
python scripts/efficientvit_core_20261112/verify.py --size 512

# 列出版本 / 预览全部 21 个任务
python scripts/efficientvit_core_20261112/batch_train.py --list
python scripts/efficientvit_core_20261112/batch_train.py --dry-run

# 主实验：全部版本，从零训练
python scripts/efficientvit_core_20261112/batch_train.py --gpus 0,1

# 可选预训练独立协议，沿用原转换权重
python scripts/efficientvit_core_20261112/batch_train.py --gpus 0,1 --pretrained

# 重算指标，保留 batch_summary.json 的进程状态，另写 metrics_summary.json
python scripts/efficientvit_core_20261112/summarize.py work_dirs/efficientvit_core_20261112/scratch/fp32
```

初始化/精度/seed/版本隔离目录，禁止覆盖既有运行。共享 runner 新增 bank_content/bank_uniform/
joint_fixed/postfusion 作为同 seed 参考，历史默认行为不变。
bank_structure 比较 baseline、bank_content 与 bank_uniform；joint 比较 baseline、joint_fixed 与 postfusion。
不将旧轮次 adaptive 下采样分数作为本轮 baseline；先判断核心独立收益，再决定是否组合。
直接 tools/train.py 执行配置会保留预训练 load_from，直接 scratch 必须加 `--cfg-options load_from=None`。
若 40k 未充分收敛，延长实验应统一 baseline 和关键对照的迭代与 scheduler。

## 训练后诊断

```bash
python scripts/efficientvit_core_20261112/diagnose.py configs/efficientvit_core_20261112/bank_structure.py --checkpoint PATH_TO_BEST_CHECKPOINT --num-samples 8 --device cuda:0 --output work_dirs/core_bank_diagnostics.json
python scripts/efficientvit_core_20261112/diagnose.py configs/efficientvit_core_20261112/joint.py --checkpoint PATH_TO_BEST_CHECKPOINT --num-samples 8 --device cuda:0 --output work_dirs/core_joint_diagnostics.json
```

严格加载对应 checkpoint。bank 记录路由熵、空间变化、状态占用、两个记忆的相似性；
joint 记录 lambda、实际全局质量占比及空补集比例；postfusion 的占比按其实际输出混合公式记录。
前向诊断使用图像特征，不使用 GT。统计只是内部激活，不能单独证明准确率、裂缝分组或因果贡献。
`--synthetic` 用随机模型/64×64 输入检查脚本，输出显式标记；后两个 stage 在这种小输入下会遇到空全局补集，属边界检查，不代表真实验证图像情况。
Windows 验证/诊断可加 `--no-yapf-cache`，仅禁用当前进程可选缓存。

## 已完成验证

- 70 项本轮与既有 core/编码器/decoder/runner 检查通过。
- 全部 7 配置通过 64×64、512×512 的真实 CE+Dice 前后向；两类主方案四个模块的所有新路由参数
  在 512×512 分割损失下有有限非零梯度。
- 验证双记忆与显式结构关系核一致；均匀/单状态还原原注意力；局部/全局统计与显式掩码一致。
- 验证边界、空补集、全零核、共同与分别归一化的区别、原注意力恢复条件、fp32/CPU bf16 梯度。
- 验证四处替换、原权重键、原下采样/head/训练协议、同种子已有参数及 CPU RNG 初始化对齐。
- 实际转换权重全部 228 键可加载；诊断脚本完成随机模型检查。
- 没有启动正式训练；没有本轮精度、实际部署速度或已确认原创性结论。
