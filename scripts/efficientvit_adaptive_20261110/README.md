# 局部结构引导的可学习下采样（2026-11-10）

本轮在主干 stride 8→16 的 MBConv 内部替换空间聚合规则，研究如何根据邻域结构
选择应传入下一阶段的信息。后续原始 LiteMLA、其执行顺序及原始分割头固定。
这是尚待训练验证的研究原型；动态聚合本身不构成新颖性证明。

## 主干信息流

```text
stride-8 特征（B0: 32 通道）
    ↓ 原 inverted_conv：1×1 扩展 + 原有激活（128 通道）
    ├─ 128→8 描述 → 中心/邻域差异 → 共享小 MLP → 4 组 × 9 邻域权重
    │                                           ↓
    └─ 3×3、stride-2 原始扩展特征邻域 → 分组加权聚合（128 通道）
                                                   ↓
                   原 depth_conv 的归一化/激活 → 原 point_conv 投影（64 通道）
                                                   ↓
                       原有后续 LiteMLA + MBConv blocks
```

接入位置：`backbone.stages.2.op_list.0.main.depth_conv.conv`。
该位置由原 nn.Conv2d 替换成 LocalStructureAggregation；没有原深度卷积结果的附加旁路，
不修改其他 stage，不移入/重复计算首次 LiteMLA，不从 decoder 获取反馈，不叠加 subpixel。
`structure_downsample_stages` 仅控制历史 structure 模式；新 adaptive 模式固定只在 stage2 入口生效。

## 聚合机制

设 V 是原 MBConv 扩展后的特征，Z=Conv1x1_128_to_8(V)，在间隔为 2 的规则中心 i
提取 3×3 邻域 j。structure 版使用：

```text
delta_ij = Z_j - Z_i
f_ij = concat(Z_i, delta_ij, abs(delta_ij))
score_igj = MLP_shared(f_ij)_g + learned_position_bias_gj
a_igj = softmax_over_valid_neighbors(score_igj)
Y_ig = sum_j a_igj * V_jg
```

- 四组通道拥有独立的空间权重；同组内通道共享权重。值来自原扩展特征，不使用描述投影替代值。
- MLP 在所有位置和偏移间共享，输入 24 通道、隐藏 8 通道、Hardswish、输出 4 个组分数。
- 输出层不加跨邻域共享的组 bias，因为该常量会在 softmax 中约掉；另有可学习的 4×9 位置偏置。
- 无采样坐标偏移、硬选择或全局条件；GT 仅用于原 CE+Dice 监督，不进入路由。
- 边界 padding 位置严格掩蔽后再归一化。偶数/奇数/单像素输入均输出 ceil(H/2)×ceil(W/2)。
- softmax 与加权求和使用 float32，结果转回输入 dtype；没有空间 N×N 注意力。

空间权重为非负归一化权重，因此聚合具有凸组合约束；原深度卷积可学习带符号空间滤波。
该约束是否影响细边缘提取是待验证问题，不能预先保证结构保真、抗混叠或平移等变。

## 版本与参数

|版本|邻域权重依据|整网参数|
|---|---|---:|
|baseline|原 stride-2 depthwise 卷积|704,498|
|structure|中心、带符号差异及差异幅值|704,510|
|content|中心、邻域内容及二者逐通道乘积|704,510|
|static|仅学习每组九个位置偏置|703,254|
|uniform|有效邻域均匀权重|703,218|

structure/content 的路由网络宽度、参数量与其余模型一致；content 提供内容交互对照，
结构版没有独占更大参数预算。两者仅在路由输入表征上不同，不保证某个表征一定更有效。
static 用于检查输入相关性是否必要，uniform 用于检查可学习空间选择是否必要。
uniform 与忽略 padding 的 3×3 stride-2 平均池化等价；static 的零偏置初始化也从均匀权重开始。
这些对照用于验证机制，不作为额外模块创新。

完整 structure 比原始整网多 12 个参数（约 +0.0017%）。但 unfold 会显式展开局部 patch，
float32 聚合和路由均产生额外访存。参数近似不变及线性复杂度不能代表延迟/显存近似不变；
当前未测实际部署延迟和峰值内存，不宣称加速。

## 训练

本轮默认所有版本从零训练，固定原始分割头。五配置 × seeds 42/43/44 =15 次。
数据、CE 0.5 + Dice 1.0、AdamW lr=0.0004、batch=12、fp32、40k、warmup=1500、
每 1000 iter 验证均沿用现有协议。

```bash
# 全部配置真实 CE+Dice 前后向（随机数据，不是精度）
python scripts/efficientvit_adaptive_20261110/verify.py --size 512

# 列出配置 / 预览默认任务
python scripts/efficientvit_adaptive_20261110/batch_train.py --list
python scripts/efficientvit_adaptive_20261110/batch_train.py --dry-run

# 全部五配置、三个 seed，从零训练
python scripts/efficientvit_adaptive_20261110/batch_train.py --gpus 0,1

# 可选预训练独立协议；自动准备迁移权重，不混入 scratch 汇总
python scripts/efficientvit_adaptive_20261110/batch_train.py --gpus 0,1 --pretrained

# 重算同轮指标，保留权威 batch_summary.json，另写 metrics_summary.json
python scripts/efficientvit_adaptive_20261110/summarize.py work_dirs/efficientvit_adaptive_20261110/scratch/fp32
```

输出按初始化/精度/seed/版本隔离，拒绝覆盖既有运行目录。共享 runner 保留旧轮次默认行为，
增加 content/static/uniform 同 seed 参考差值；结构版必须同时比较原始 baseline 和各机制对照。
不可将旧轮 pretrained 的数字用于本轮 scratch 配对。

40k 从零训练可能未完全收敛。若验证更长训练预算，应统一延长 baseline、主版本和关键对照，
同时同步 scheduler，不只延长改进版本。

直接 `tools/train.py CONFIG` 保留配置的 load_from。直接从零训练须显式加
`--cfg-options load_from=None`。正式训练依赖数据与 CUDA，本次未启动。

## 预训练兼容

`--pretrained` 会先运行 prepare_pretrained.py，从原转换的
`work_dirs/pretrained/efficientvit_b0_crack.pth` 丢弃已替换空间核的 weight/bias 两个键，
写入独立 `efficientvit_b0_adaptive_downsample.pth`，原文件保持不变。
该过程不移动任何注意力参数；所有 LiteMLA、扩展、输出投影及 head 的兼容参数保持原名/原值。
新路由参数随机初始化，二分类 conv_seg 仍按原配方随机初始化。

迁移包含源 SHA256、丢弃键与缺失键检查。实测 baseline 加载 228 键，新配置加载 226 键，
没有 unexpected keys 或尺寸不匹配。新空间算子与原卷积不等价，旧 checkpoint 不可直接 resume。
可手动执行 `python scripts/efficientvit_adaptive_20261110/prepare_pretrained.py`。

## 训练后路由诊断

```bash
python scripts/efficientvit_adaptive_20261110/diagnose.py configs/efficientvit_adaptive_20261110/structure.py --checkpoint PATH_TO_BEST_CHECKPOINT --num-samples 8 --device cuda:0 --output work_dirs/adaptive_routing_diagnostics.json
```

严格加载训练 checkpoint，在真实验证图像中记录权重归一化误差、中心权重、熵、组间差异、
有效完整邻域内的空间变化，以及同一特征下相对均匀聚合的数值变化。
诊断只衡量内部信息流，不是准确率或因果贡献；不能用激活更强直接认定模块更好。
`--synthetic` 使用随机模型/随机输入，输出显式标记，不能当作训练结果。
Windows 的 verify/prepare_pretrained/diagnose 可加 `--no-yapf-cache` 避免可选缓存写入阻塞。

## 已完成检查

- 59 项新机制与现有 backbone/decoder/runner 回归检查通过。
- 五配置通过 64×64 与 512×512 的真实 CE+Dice 前后向；完整 structure 的所有路由参数
  在 512×512 分割损失下均有有限非零梯度。
- 验证 padding 掩蔽、权重归一化、常数保持、均匀池化等价、通道组选择不同邻居、
  内容相关性、fp32/CPU bf16 梯度、奇数与单像素输入、四尺度输出与唯一接入位置。
- 验证原四个 LiteMLA 保留、原兼容参数迁移、head/数据/训练协议一致、参考差值与默认 scratch 隔离。
- 诊断脚本完成随机模型 synthetic 检查。没有正式训练精度、部署速度或新颖性结论。
