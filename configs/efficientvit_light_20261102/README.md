# EfficientViT-B0 2026-11-02

9 个轻量对照配置；统一运行方法、预算、机制定义和初始化限制见
[训练脚本说明](../../scripts/efficientvit_light_20261102/README.md)。

正式实验优先通过该批量入口运行。直接调用 `tools/train.py` 时请显式设置独立
`--work-dir`；`cra_fixed.py` 默认关闭预训练，需要与同为 `load_from=None` 的 baseline 比较。
