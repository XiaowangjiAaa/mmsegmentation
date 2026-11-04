# Subpixel 机制改进（2026-11-05）

本轮独立实现三个方向，并提供前两个方向的统一版本。它们是待验证研究原型，
不是参数扫描；尚未正式训练，不声称已获得精度、平移鲁棒性或部署速度收益。

## 版本与预算

所有新版本使用同一个 B0、原预训练融合 head、16 通道模块宽度和 stride-4 输出，
主损失统一 CE 0.5 + Dice 1.0，没有新增辅助监督。正式实验不要混用旧轮次的对照分数。

| 配置 | 机制 | 整网参数 |
| --- | --- | ---: |
| baseline | 原始 B0 | 704,498 |
| detail_resize | 两段插值对照 | 704,498 |
| detail | 已有细节分支 | 704,749 |
| subpixel | 原始子位置条件读取 | 708,453 |
| write | 残差参与 Key/Value 写入结构记忆 | 709,287 |
| contrast | 共享语义与零均值位置修正分开重建 | 708,453 |
| write_contrast | 结构写入与分离重建的统一版本 | 709,287 |
| stable | 滤波下采样与完整保留的细节残差成对重建 | 708,453 |

参数量不等于延迟。write 增加状态写入与读取运算；其他版本虽无新增参数，也有不同算子成本。

## 1. write：让结构进入全局记忆

原版只用子位置残差 R 改变 Query，Key/Value 全部来自粗语义 S。
现在保留语义状态，增加结构状态：

```text
K_r = ELU(Wk_r R) + 1
V_r = Wv_r R                         # 保留正负值、无 bias
w_r = 0.02 + 0.98 sigmoid(g_write(S, |R|))
A_r = sum(w_r K_r V_r^T)
z_r = sum(w_r K_r)
T_r = (ELU(Wq_r R)+1) A_r / ((ELU(Wq_r R)+1) z_r)
```

结构 Key 随残差变化，因此不会像“共享 Key × 零和残差 Value”那样必然抵消。
语义条件权重与读取置信度是可学习软权重，不使用 GT；不能未经检查就称其为裂缝掩码。
四个位置共享投影，写入一份小型 d×d 状态，不生成空间 N×N 矩阵。
输出为原语义读取 + 置信度加权结构读取 + 直接残差。

## 2. contrast：明确公共语义与位置差异的职责

用共享 Query 读取语义 G；用子位置 Query 得到四份 T_p，取
`Delta_p = T_p - mean_phase(T)`，再重建 `P_p = G + R_p + Delta_p`。
因此在四相张量上，`mean_phase(P) = G`（浮点误差范围内）。

它用角色约束避免位置修正分支重复搬运公共语义，不保证自动提高表达能力或精度。
write_contrast 进一步将语义/结构读取的合量中心化，令结构记忆参与位置差异重建。
奇数尺寸仍沿用原版 replicate padding 并裁回，零均值约束针对裁剪前的四相单元。

## 3. stable：保留细节的滤波采样与重建

本版本移除固定四相 Query 分组，先构造
`L = binomial_3x3_filter_then_stride2(D)`，再保留 `H = D - bilinear_up(L)`。
Query 在完整细节网格上由 `up(Wq(S)) + Wr(H)` 形成，K/V 来自粗语义与 L。
输出是每个高分辨率位置的上下文读取 + H。

滤波在降采样之前执行；被滤去的变化仍在 H 中保留，而非直接抹平细线。
`up(L)+H` 可重建当前投影后的 D，支持奇数和单像素尺寸。
内存网格仍有 stride-2 采样，因此没有严格平移等变保证；stable 仅是候选版本名称。
需要在真实图像平移、细裂缝和误报纹理上检验其效果，不能把固定低通滤波本身认定为创新。

## 运行

```bash
# 8 个配置的真实分割损失、输出和梯度检查，不需要数据或权重
python scripts/efficientvit_subpixel_20261105/verify.py --size 512

# 显示配置与命令
python scripts/efficientvit_subpixel_20261105/batch_train.py --list
python scripts/efficientvit_subpixel_20261105/batch_train.py --variants all --dry-run

# 默认 8 个配置、3 个 seed、40k、fp32；空闲 GPU 自动接下一项
python scripts/efficientvit_subpixel_20261105/batch_train.py --gpus 0,1 --variants all --seeds 42,43,44

# 单独跑旧 subpixel 和四个新版本
python scripts/efficientvit_subpixel_20261105/batch_train.py --gpus 0,1 --variants subpixel,write,contrast,write_contrast,stable --seeds 42,43,44

# 有真实数据/CUDA 时的两步执行检查
python scripts/efficientvit_subpixel_20261105/batch_train.py --gpus 0 --variants write,contrast,write_contrast,stable --scratch --seeds 42 --batch-size 2 --max-iters 2 --work-root work_dirs/subpixel_v2_smoke
```

正式比较时建议包含同轮 detail 和 subpixel。可统一加 --scratch；它会对所有选中版本关闭预训练。
不会覆盖已有运行目录；重跑需新 --work-root。默认输出位于
`work_dirs/efficientvit_subpixel_20261105/pretrained/fp32/seed*/variant/`。
Windows 验证时若 YAPF 缓存写入阻塞，可加 `--no-yapf-cache`，不会修改依赖。

## 汇总与机制诊断

```bash
python scripts/efficientvit_subpixel_20261105/summarize.py work_dirs/efficientvit_subpixel_20261105/pretrained/fp32

python scripts/efficientvit_subpixel_20261105/diagnose.py configs/efficientvit_subpixel_20261105/write_contrast.py --checkpoint PATH_TO_ITS_BEST_CHECKPOINT --num-samples 8 --device cuda:0 --output work_dirs/write_contrast_diagnostics.json
```

训练汇总现在同时记录相对 baseline、detail_resize、detail、subpixel 的成对 IoU 差值，
保留原 baseline 字段以兼容旧脚本。缺少对应 seed 的参考时不构造该配对。
重建指标文件不能替代 batch_summary 中的进程完成状态。

diagnose 用验证集图像与训练时预处理，记录四相语义读出差异、结构读取 RMS、
contrast 的均值误差、中心化重建误差及修正 logits/主 logits 的 RMS 比例。
它不是分割准确率，也不能单独证明因果贡献。原 subpixel 使用同权重数值等价的分量适配器。
`--synthetic` 可做无数据的随机模型脚本检查，输出会明确标记，不能当作训练结果。
诊断没有 GT 参与模型特征或记忆计算，也不缓存样本标签。

## 检查范围与限制

- 所有配置通过 64×64 与 512×512 的真实 CE+Dice 前后向检查。
- 新测试覆盖旧权重/函数兼容、共享 Key 下的残差抵消、结构 Key 避免必然抵消、
  无结构时零写入、四相零均值重建、滤波与残差重建、棋盘高频保留、CPU bfloat16 及分支梯度。
- 原始 subpixel checkpoint 的真实验证样本诊断已完成脚本检查，不代表新模块已训练。
- 没有正式 40k 训练或新模块精度、平移鲁棒性与延迟结果。
- 正式论证还需排除投影、局部残差及公共上下文的作用；不能只凭新版本名称认定机制成功。
