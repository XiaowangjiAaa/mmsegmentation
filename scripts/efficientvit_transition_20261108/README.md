# 语义引导的结构保留下采样（2026-11-08）

本轮核心改进发生在 EfficientViT-B0 主干的 stride 8→16：
替换原 stride-2 MBConv，并将紧接着的首次 LiteMLA 移入新模块，先生成语义上下文，
再聚合未被压缩掉的结构分量。新输出直接传入后续主干。

这是待验证的研究原型；尚无正式训练、精度提升、拓扑质量、平移鲁棒性或部署速度结论。
结构分解本身不是创新证明；设计贡献是否成立仍需相关工作比较与机制证据。

## 信息流与接入

```text
输入 backbone stride-8 特征 X（B0: 32 通道）
  ├─ M：四相位均值 → 1×1/BN/Hardswish → 原首次 LiteMLA + 单次残差 → C（64 通道）
  └─ Dh / Dv / Dd：带符号位置差异 → 共享 32→8 投影 ─────────────┐
                                                                ↓
                 M 的窄投影、结构幅值、C 的窄投影 → 逐位置/通道组软门控
                                                                ↓
                 concat(C, gh*Dh, gv*Dv, gd*Dd) → 1×1/BN → Y（64 通道）
                                                                ↓
                 原首次 block 的局部 MBConv → 剩余原始 block / stage
```

设 a/b/c/d 对应一个 2×2 单元的左上/右上/左下/右下：

```text
M  = (a+b+c+d)/4
Dh = (a-b+c-d)/4
Dv = (a+b-c-d)/4
Dd = (a-b-c+d)/4
```

四个分量在学习投影之前可以精确重建 replicate-pad 后的输入；之后的窄投影、
门控与通道融合是有损压缩，不能宣称完整保留细节或严格抗混叠。

语义模式的门控输入为 concat(Ps(M), abs(Ps(Dh/Dv/Dd)), Pc(C))：
`g=0.05+0.95*sigmoid(router(...))`。幅值用于决定权重，实际传递的是带正负号的结构分量。
每个方向有 8 个权重通道，不构造空间 N×N 注意力，不做硬路由或 GT 引导。
0.05 是固定软门控下限，不做该常数的参数扫描。

具体接入：`backbone.stages.2.op_list.0` 为新转换模块；
`stages.2.op_list.1.context_module` 为 Identity（该 LiteMLA 已移入转换模块），
该 block 的 local_module 保留。总计仍为四个 LiteMLA，每个前向仅执行一次。
原 stride-2 MBConv 已从新配置计算图中移除，不存在旧下采样结果加结构旁路。
默认原架构和上一轮编码器配置的图与参数布局保持原有行为。

## 配置与预算

|版本|作用|分割头|整网参数|
|---|---|---|---:|
|baseline|原始 B0|原始|704,498|
|semantic|完整语义条件结构聚合|原始|700,362|
|fixed|结构权重固定为 0.525|原始|698,866|
|local|门控仅看局部均值与结构幅值，移除 C 条件|原始|699,658|
|pooled|仅均值语义路径，移除三个结构分量|原始|697,074|
|subpixel|原始 B0|原版 subpixel|708,453|
|semantic_subpixel|完整新主干模块|原版 subpixel|704,317|

fixed/local/pooled 是用于检验结构和语义条件必要性的对照，不作为独立模块创新。
完整 semantic 比原始整网少 4,136 参数，约 -0.59%；对照版本并非严格参数相等，需结合成本解释。
参数减少不代表延迟减少；当前没有 FLOPs 和目标硬件延迟数据。
实现支持 B 系列，当前训练配置与预训练迁移只针对 B0；拒绝与上一轮结构旁路或非 v1 block 同时启用。

## 训练：主协议从零训练

新模块无法完整复用原下采样权重，因此本轮 runner **默认 scratch**，用于与原始
baseline 比较架构本身。共享旧 runner 默认仍为 pretrained，本次只增加显式的
互斥 `--scratch/--pretrained` 选择，不改变历史默认行为。

所有版本保持相同数据划分、CE 0.5 + Dice 1.0、AdamW lr=0.0004、batch=12、
fp32、40k（warmup 1500）、每 1000 iter 验证、seeds 42/43/44。

```bash
# 构建全部 7 配置，随机数据上的真实 CE+Dice 前后向检查
python scripts/efficientvit_transition_20261108/verify.py --size 512

# 查看默认 15 次任务；dry-run 不启动训练、不生成预训练文件
python scripts/efficientvit_transition_20261108/batch_train.py --dry-run

# 主实验：baseline + semantic + 三个机制对照，共 15 次，从零训练
python scripts/efficientvit_transition_20261108/batch_train.py --gpus 0,1

# 包含 subpixel 互补性实验的全部 7 配置，共 21 次，从零训练
# 与上一条二选一；相同运行目录不允许覆盖。
python scripts/efficientvit_transition_20261108/batch_train.py --gpus 0,1 --variants all

# 可选预训练部署配方，独立输出到 pretrained/fp32/，不混入 scratch 汇总
python scripts/efficientvit_transition_20261108/batch_train.py --gpus 0,1 --variants all --pretrained

# 重算指标；保留权威进程状态 batch_summary.json，写入 metrics_summary.json
python scripts/efficientvit_transition_20261108/summarize.py work_dirs/efficientvit_transition_20261108/scratch/fp32
```

输出按初始化/精度/seed/版本隔离。汇总增加了对 fixed/local/pooled 的同 seed 差值，
原 baseline/subpixel 及历史字段保留。原始 head 的 semantic 应比较 baseline 及三个机制对照；
semantic_subpixel 应比较同轮、同初始化的 subpixel。不同初始化、旧轮次分数不可直接配对。
若 scratch 40k 尚未充分收敛，应双方统一延长并同步 scheduler，不能只延长改进模型。

## 预训练迁移

`--pretrained` 正式执行前会运行 prepare_pretrained.py：

1. 读取 `work_dirs/pretrained/efficientvit_b0_crack.pth`，源文件保持不变。
2. 原 stage2 第一块的 LiteMLA 共 9 个参数/缓冲键迁移到新转换模块。
3. 丢弃已被替换的 stride-2 MBConv 的 10 个键；其他旧参数继续复用。
4. 检查键名、形状及预期 missing keys，写独立
   `work_dirs/pretrained/efficientvit_b0_semantic_transition.pth`，保存源 SHA256 与迁移记录。

这是兼容参数初始化，不是函数等价转换。新均值投影、结构投影、门控和融合随机初始化；
Cityscapes 19 类分类器原本已被转换脚本跳过，二分类 conv_seg 同样随机初始化。
subpixel 的新增 head 参数也随机初始化。旧 checkpoint 不可直接 resume 新结构。
baseline 可加载 228 个原键，新模块配置加载 218 个兼容键；不同覆盖率是预训练结果的解释限制。

手动准备：

```bash
python scripts/efficientvit_transition_20261108/prepare_pretrained.py
```

Windows 验证或手动迁移若 YAPF 缓存写入阻塞，可给 verify.py / prepare_pretrained.py /
diagnose.py 加 `--no-yapf-cache`，仅禁用当前进程的可选缓存，不改依赖文件。
直接 `tools/train.py CONFIG` 会保留配置的预训练路径；直接从零训练必须显式传
`--cfg-options load_from=None`。

## 训练后机制诊断

```bash
python scripts/efficientvit_transition_20261108/diagnose.py configs/efficientvit_transition_20261108/semantic.py --checkpoint PATH_TO_BEST_CHECKPOINT --num-samples 8 --device cuda:0 --output work_dirs/semantic_transition_diagnostics.json
```

严格加载训练 checkpoint，在真实验证图像上记录三方向门控、带符号结构幅值、
置零上下文条件时的门控变化，以及同一融合算子内置零结构项造成的输出变化。
GT 不进入特征计算。这些是内部激活诊断，不是全验证准确率或机制因果贡献。
`--synthetic` 是无 checkpoint 的随机数据/随机模型检查，输出明确标记，不能视作训练证据。

## 已完成验证与限制

- 48 项本轮及既有编码器、结构头、subpixel、CRA 和 runner 检查通过。
- 全部 7 配置通过 64×64 与 512×512 的真实 CE+Dice 前后向；没有启动正式训练。
- 分解覆盖偶数、奇数、单像素尺寸与重建；验证水平/垂直/对角模式、符号传递、
  上下文确实改变门控、所有控制模式下 fp32/CPU bf16 参数梯度有限且非零。
- 检查原 MBConv 已移除、首次 LiteMLA 未重复执行、四尺度输出正确、旧配置默认行为及预算隔离。
- 实际转换权重在全部配置中的缺失键符合预期，没有 unexpected keys 或尺寸不匹配。
- 诊断脚本完成随机模型 synthetic 检查，尚无已训练新模块的激活或精度结果。

本轮验证用于确认实现与信息流，不能据此认定论文创新成立或模型精度提升。
