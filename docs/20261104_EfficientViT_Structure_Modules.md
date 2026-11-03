# EfficientViT：从细节实验转向结构交互模块（2026-11-04）

本轮目标是设计可独立替换的机制模块；参数搜索和消融只用于后续验证。
完整实现定义、辅助监督和运行命令见
[结构模块说明](../scripts/efficientvit_structure_20261104/README.md)。

## 已有实验依据

来源：本地 `efficientvit_light_20261102/pretrained/fp32/`，5 个版本各 3 个 seed，均完成 40k。
按最佳 mIoU checkpoint 读取同点的前景指标，下面为裂缝 IoU 均值与样本标准差：

| 版本 | IoU/crack | 相对 baseline（百分点） |
| --- | ---: | ---: |
| baseline | 61.98 ± 0.15 | — |
| magnitude | 61.97 ± 0.17 | -0.01 |
| detail_resize | 63.75 ± 0.08 | +1.78 |
| detail | 64.22 ± 0.18 | +2.24 |
| magnitude_detail | 64.08 ± 0.36 | +2.11 |

detail 相对 detail_resize 的三个成对差值为 +0.19、+0.64、+0.56。
这支持继续研究细节参与信息交互，但不证明背景干扰或下采样混叠已经被确认为主要瓶颈。
detail_resize 本身也重新训练，不能将其收益全部归为同权重下的纯后处理变化。

## 新模块及论文启发

1. **StructurePartitionMemory**：细节和语义共同组织两个互补全局 KV 状态，高分辨率 Query
   独立归一化读取，并通过内容相关的对比读取融合。借鉴
   [RALA 的 KV 信息表达问题](https://arxiv.org/abs/2411.07635)及
   [PolaFormer 对交互信息损失的分析](https://proceedings.iclr.cc/paper_files/paper/2025/hash/7d2b770c3ccd35b41c9453ef6f8765a3-Abstract-Conference.html)。
   与旧单状态 kv_reweight 的区别是状态分区及条件读取；普通双路相减已有
   [Differential Transformer](https://proceedings.iclr.cc/paper_files/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html)
   等相关工作，不能仅以此声称创新。
2. **ReciprocalConnectivity**：预测双向邻域支持，结合语义一致性形成对称连接，并以 self-loop
   做两步归一化传播。借鉴
   [Spatial-Mamba 的结构感知状态融合](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b7216f4a324864e1f592c18de4d83d10-Abstract-Conference.html)，
   这里采用学习到的相互确认连接，而非固定方向卷积或完整 Mamba 扫描。
3. **SubpixelStructureMemory**：保留 2×2 子位置及零和结构残差，让四种位置分别读取共享的
   粗尺度全局状态，再重建高分辨率特征。借鉴
   [Alias-Free ViT 的采样敏感性问题](https://papers.nips.cc/paper_files/paper/2025/hash/0234a34d7c581d82dd94dfa6517841de-Abstract-Conference.html)。
   可逆降采样已有 [Wave-ViT](https://arxiv.org/abs/2207.04978) 等工作，本实现的候选机制在于
   子位置残差如何改变全局读取，不宣称简单重排本身新颖，也不宣称严格 alias-free。

三个模块均为待验证研究原型，尚未完成最接近工作的新颖性全面核查。
它们保留预训练骨干和主 head，在 decoder 的特征交互处独立运行，不将三个模块叠加。
partition 和 connectivity 分别有训练期路由、邻接辅助监督；推理均不使用 GT。

## 本地验证

- 6 个配置（3 个对照 + 3 个新模块）均通过 64×64 与 512×512 真实分割损失和反向梯度检查。
- partition / connectivity / subpixel 的整网参数分别为 709,254 / 708,845 / 708,453。
- 回归检查覆盖压缩读取、独立归一化、路由梯度、连接对称和边界、常量传播、子位置重建、
  bfloat16、ignore 标签、旧预训练键兼容及辅助损失。
- 新训练入口默认使用三个 seed，并与 baseline、detail_resize、detail 一起对照。
- 尚未启动正式 40k 训练；本文件不包含三个新模块的精度或延迟结论。
