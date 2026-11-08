# 语义引导的结构保留下采样（2026-11-08）

在 backbone 的 stride 8→16 处，用公共分量、带符号结构分量及首次 LiteMLA 上下文
共同生成下一阶段特征，替换原下采样算子。默认实验固定原分割头。

配置：baseline、semantic，以及 fixed/local/pooled 机制对照；subpixel 与
semantic_subpixel 用于后续互补性验证。

[机制、训练和验证说明](../../scripts/efficientvit_transition_20261108/README.md)。

注意：**批训练脚本默认从零训练**，会覆盖 `load_from=None`；直接使用 tools/train.py
执行配置则保留配置中的预训练路径。要直接从零训练，显式增加
`--cfg-options load_from=None`。正式训练尚未启动。
