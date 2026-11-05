# EfficientViT 编码器结构实验（2026-11-06）

本轮改变 backbone 特征提取过程，默认保留原始 EfficientViTHead。
机制、预算、预训练兼容性、运行命令见
[统一说明](../../scripts/efficientvit_encoder_20261106/README.md)。

默认独立实验：baseline / downsample / query / context_local。
组合实验：downsample_query，以及原版 subpixel、downsample_query_subpixel、context_local_subpixel。
所有配置使用相同数据、CE+Dice、训练预算，未启动正式训练。
