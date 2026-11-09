# 局部结构引导的可学习下采样（2026-11-10）

只替换 B0 的 stride 8→16 MBConv 内部深度卷积空间聚合。
保留其通道扩展/投影、后续四个原始 LiteMLA 及原始分割头。

主版本：structure。对照：baseline、等参数量 content、static、uniform。
[机制、验证与运行说明](../../scripts/efficientvit_adaptive_20261110/README.md)。

批训练脚本默认 scratch，并覆盖 load_from=None；直接使用 tools/train.py 执行配置
会保留其中预训练路径。从零直接训练时增加 `--cfg-options load_from=None`。
本轮尚未启动正式训练，没有准确率结论。
