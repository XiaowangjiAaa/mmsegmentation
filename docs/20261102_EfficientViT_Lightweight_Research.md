# EfficientViT-B0：2025–2026 轻量 ViT 调研与实验建议

日期：2026-11-02。面向当前裂缝二分类项目；优先考虑整网约 0.7M 参数的预算。
以下区分论文机制、面向本项目的推断、已完成代码修复；不把论文公开数据集收益当作本项目收益。

后续实现：已新增 9 个可运行对照（含 baseline、插值对照与修复后 CRA），见
[统一脚本与各版本公式](../scripts/efficientvit_light_20261102/README.md)。
新增 magnitude / polarity / kv_reweight 是受论文启发的适配消融，不是原论文完整复现。

## 1. 已确认的实现与记录问题

### CRA 两个修复

1. `_sample_curves()` 创建了 `seed_grid`，但计算坐标时未使用。修复为
   `coords = seed_grid + base_offsets + learned_offsets`。
   原始 dense 增加的是集中在中部的曲线，而非设计中的全图网格覆盖。
2. `CRAModule` 返回 `x + gamma*fused`，外层又使用带 identity 的 `ResidualBlock`，
   导致 context 为 `2*x + gamma*fused`。现在内部仅返回 `gamma*fused`，外层负责唯一的 shortcut。

针对性检查覆盖 16/36 条曲线、不同长度的全图四角采样、context 零门控恒等、
非零门控时 offset/attention/fusion 的有限且非零梯度。
历史权重张量形状未变，但采样位置和残差语义改变；旧 checkpoint 不能直接用于修复后指标对比，须重新训练。
修复后的 `CRAModule` 单独调用返回残差增量，调用者如需完整输出应使用外层 context block。

### OA-LA 对照更正

原始 `scalars.json` 按最佳 mIoU 选点：

| 运行 | 裂缝 IoU | F1 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| 20261024 fp32 baseline | 62.22 | 76.71 | 75.74 | 77.70 |
| 20261027 同批次 baseline | 61.94 | 76.50 | 74.87 | 78.20 |
| 20261027 OA-LA | 62.12 | 76.63 | 76.25 | 77.02 |

OA-LA 对同批次基线是 +0.18，而非 -0.10；两次基线差 0.28，不能据此证明稳定收益。
同批次也不等于相同训练随机种子。数据集 `seed=42` 只固定数据划分，不能替代 `randomness.seed`。

## 2. 值得借鉴的主会论文

### MALA，ICCV 2025：优先级最高的 attention 实验

[Rectifying Magnitude Neglect in Linear Attention](https://openaccess.thecvf.com/content/ICCV2025/html/Fan_Rectifying_Magnitude_Neglect_in_Linear_Attention_ICCV_2025_paper.html)
研究归一化线性注意力丢失 query 幅值的问题，用缩放和偏移修正分布，保持关于 token 数的线性复杂度。
[作者代码](https://github.com/qhfan/MALA)。

项目建议：保留 LiteMLA 的 `qkv/aggreg/proj`、多尺度分支和 MBConv，只替换聚合后的 attention 计算，
对 stage 2/3 全部 4 个 LiteMLA 生效。先独立验证 MALA 思路，暂不叠加方向门控。
这是针对尚未测试的归一化问题，区别于已有 window/global 替换及 strip 增强。
需要按原公式处理尺度与数值稳定，不能将“给 Q 乘一个标量”当作修复：相同标量会在分子分母约掉。
作为工程迁移不应宣称原创；论文完整网络的精度和成本不能直接移用到 B0。

### PolaFormer，ICLR 2025；NaLaFormer，ICML 2026

[PolaFormer 正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7d2b770c3ccd35b41c9453ef6f8765a3-Abstract-Conference.html)
恢复同号和异号 query/key 交互，并通过可学习幂函数增强区分性。
[NaLaFormer 论文](https://arxiv.org/abs/2506.21137)进一步研究 norm×direction 分解；
其 ICML 2026 接收信息见[作者仓库公告](https://github.com/ZacharyMeng/PolaFormer)。

项目建议：将 polarity 和 magnitude 拆成独立消融，保留原多尺度聚合。
弱裂缝与背景纹理能否从保留符号信息中受益是待验证假设，不能把 ReLU 丢弃负值直接等同于丢失裂缝。
正负分路可能增加中间张量、矩阵乘法及显存，须测实际延迟，不能仅以 O(N) 判定轻量。
排在 MALA 之后，避免一次改动两个问题。

### RALA，CVPR 2025：KV 内容加权与输出多样性

[Breaking the Low-Rank Dilemma of Linear Attention](https://openaccess.thecvf.com/content/CVPR2025/html/Fan_Breaking_the_Low-Rank_Dilemma_of_Linear_Attention_CVPR_2025_paper.html)
从 KV buffer 与输出特征两处提高有效秩，使用 token 权重与逐 token 通道调制。

项目建议：先测现有 LiteMLA 的奇异值谱、有效秩，再决定是否迁移。
可探索由 stride-4 细节预测引导 KV token 权重，避免大面积背景占据上下文。
这属于本项目提出的适配假设，尚未实现，也未证明新颖性。
推理权重必须来自图像特征，GT 只能用于训练监督；纹理边缘不是裂缝本身。
初始权重设为均匀，保留原上下文，再分别消融内容加权、细节监督和输出调制。

### TinyNeXt，ICCV 2025：最接近当前参数规模

[An Efficient Hybrid Vision Transformer for TinyML Applications](https://openaccess.thecvf.com/content/ICCV2025/papers/Zeng_An_Efficient_Hybrid_Vision_Transformer_for_TinyML_Applications_ICCV_2025_paper.pdf)
结合 Lean Single-Head Self-Attention 与分阶段空间/通道建模，关注 reshape、copy、norm 的内存开销。
[作者仓库](https://github.com/yuffeenn/TinyNeXt)提供 1.0M 分类模型及分割适配；分割整网参数应另计。

项目建议：先尝试仅在最后 stage 使用低成本通道交互，stage 2 保留空间 LiteMLA，做计算预算重分配。
它验证的是“哪一层需要什么交互”，区别于此前四个 LiteMLA 全部换为通用 softmax attention。
不盲目增加早期下采样：分类效率策略可能损害裂缝分割的细线细节。
该路线可能改变预训练参数对应关系，既要报告部署配方，也要补 from-scratch 对照。

### Edge-RecViT，CVPR 2026：只借鉴计算分配

[正式论文](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Edge-RecViT_Efficient_Vision_Transformer_via_Semantic-Refined_Dynamic_Recursion_CVPR_2026_paper.html)
用边缘感知 token 排序和共享参数递归，将计算分给结构信息丰富的 token。
其 Base 模型仍约 23M，不能直接当作本项目的轻量替代。

项目建议：全图保留 LiteMLA，只对不确定的细节区域做额外修正。
先做规则网格上的软门控版本，再考虑真正稀疏 gather/scatter。
软门控若仍计算所有分支，就不节省对应 FLOPs；硬路由还需证明实际硬件提速，且有漏掉弱裂缝的风险。

### LiDeRe，CVPR 2026：借鉴 readout，不替换为大骨干

[作者论文与代码](https://github.com/timojl/lidere)研究冻结视觉骨干上的轻量密集预测 readout。
可借鉴其细粒度特征恢复思路，但“可训练参数少”不等于“推理整网轻量”。
项目建议：保留 B0 与现有 stride-8 主 head，额外从 stride-4 特征做窄通道细节修正。
先做双线性上采样 + depthwise + 1×1 的便宜对照，再判断是否需要内容自适应上采样。
避免将整个 32 通道 decoder 搬到 stride 4，后者使相应空间算子的计算约变为 4 倍。

## 3. 建议执行顺序

| 次序 | 实验 | 要回答的问题 | 预算约束（目标，非实测） |
| --- | --- | --- | --- |
| 0 | 修复后 CRA 与 scratch baseline | 历史负结果是否受实现偏差影响 | 先原始 16×16，暂不扩展 dense/soft |
| 1 | LiteMLA 幅值保真，借鉴 MALA | 改归一化是否比加空间分支有效 | 尽量不增加参数、不构造 N×N 矩阵 |
| 2 | stride-4 窄细节分支 | 当前限制是否在细线恢复与输出分辨率 | 先将整网控制在约 0.8M，测延迟 |
| 3 | polarity-aware LiteMLA | 符号信息是否带来独立收益 | 不与第 1 项一次性合并 |
| 4 | TinyNeXt 式分阶段交互 | 能否用更少成本保持或提高精度 | 同精度下比较真实延迟 |
| 5 | 细节引导的 KV/区域修正 | 能否做出有证据支持的任务适配机制 | 先有独立消融再组合 |

更看好的研究主线：**保留 dense 全局线性上下文，修正 attention 信息损失，用低成本高分辨率细节补偿输出。**
这是一组待验证设计判断，不承诺任何 IoU 增益。
不优先继续扫 strip 长度、增加更多平行方向分支、全量替换为曲线或引入完整 Mamba 骨干。

## 4. 第一组可运行对照

新增配置明确固定训练 seed、关闭预训练，并使用独立 work_dir：

```bash
python tools/train.py configs/efficientvit_recheck_20261102/baseline_scratch.py
python tools/train.py configs/efficientvit_recheck_20261102/cra_fixed_scratch.py
```

至少再运行两个种子，例如 43、44；覆盖 `randomness.seed` 时同时更换 `--work-dir`，
不要覆盖 seed42 或原 20261030/31 目录。不从旧 CRA checkpoint resume。
本次未启动完整训练；上述命令是待执行实验。

本地验证：两个 unittest 通过（覆盖三种曲线布局、集成残差与反向梯度）；
两个配置均完成整网构建与 64×64 输入的前向/反向。
实测整网参数：baseline 704,498，CRA fixed 924,542；未测完整输入的延迟与训练精度。
本地未安装 pytest，使用 unittest 执行新增检查。Windows 环境的 YAPF 语法缓存写入阻塞，
验证进程临时禁用了该可选磁盘缓存，没有修改依赖包。

统一数据划分、精度、batch、增强、CE/Dice、40k schedule 和 checkpoint 选择规则。
先保持现有按 mIoU 选点以便历史对照，同时报告该点的 IoU/crack、F1、Precision、Recall。
每个 seed 成对比较，报告差值及均值/标准差，最终选定配置后再评估独立 test split。
如果 scratch baseline 与 CRA 在 40k 仍上升，另做双方同长度的 80k，对齐 scheduler，不能只延长改进模型。

计时至少包含整网 batch=1 的中位数与 P95 延迟、峰值显存、参数量；统一设备、精度、预热和同步。
512×512 训练裁剪与保持宽高比的验证输入不是同一形状，延迟应同时报告标准形状和实际验证形状。
FLOPs 统计可能漏算 grid_sample/scatter 等操作，不能代替实测。
