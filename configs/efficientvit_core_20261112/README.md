# LiteMLA 核心聚合实验（2026-11-12）

直接替换 B0 的四个 LiteMLA 核心读取规则，保留原始下采样、Q/K/V、多尺度投影、
周边 MBConv 和原分割头。主方案为 bank_structure、joint；其余为机制对照。

[机制、预算和命令](../../scripts/efficientvit_core_20261112/README.md)。
批训练默认从零训练；直接 tools/train.py 使用配置中的预训练路径，直接从零训练须加
`--cfg-options load_from=None`。本轮尚无正式训练或精度结果。
