# EfficientViT 结构模块实验（2026-11-04）

本轮实现三个独立机制：结构分区全局记忆、互相确认的连接传播、子位置残差条件化上下文重建。
它们是待验证的研究原型；尚未完成正式训练或新颖性全面核查，不声称复现某篇论文。

## 模型与位置

原 B0 骨干、LiteMLA、预训练 head 融合层和主分类器保留。
新模块位于 decoder：读取 stride-8 主语义特征与 stride-4 细节特征，投影到 16 通道，
进行对应结构交互，再生成 stride-4 的 logits 修正。三个机制分别运行，不互相叠加。

| 配置 | 含义 | 整网参数 | logits stride |
| --- | --- | ---: | ---: |
| baseline | 原 B0 | 704,498 | 8 |
| detail_resize | 两段插值对照 | 704,498 | 4 |
| detail | 上轮 8 通道细节分支 | 704,749 | 4 |
| partition | 细节/语义引导的两种互补 KV 记忆 | 709,254 | 4 |
| connectivity | 双向确认连接与两步共享传播 | 708,845 | 4 |
| subpixel | 四个子位置分别查询共享全局状态 | 708,453 | 4 |

新增整网参数不足 0.7%，但不据此推导延迟或显存。所有输入位置均保留，没有硬 token 裁剪。

## 1. partition：结构分区记忆

浅层细节的平均值、绝对响应最大值与深层语义共同预测路由 logits a。
`r = 0.02 + 0.96*sigmoid(a)`，构造互补权重 r 和 1-r。
分别累积 `S_s = sum(r*k*v^T)` 与 `S_b = sum((1-r)*k*v^T)`，以及各自的 key 质量项 z。
高分辨率 Query 来自上采样语义与细节，独立读取两种记忆：`Y_s=q*S_s/(q*z_s)`、`Y_b=q*S_b/(q*z_b)`。
输出包含全局读取与 query-specific 对比：`D + Y_global + tanh(gate(D,X))*(Y_s-Y_b)`。

与旧 kv_reweight 的区别是保留两种状态及两个归一化项，而不是加权后合入同一个状态。
路由由训练时的前景占据比例辅助监督，避免仅凭无监督分区把两个记忆任意解释为结构/背景。
该监督使用当前样本 GT，仅在 loss 内构造；推理只输入特征。
使用正值 ELU+1 kernel 和 fp32 累积，没有高分辨率 N×N 注意力矩阵。

## 2. connectivity：互相确认的连接传播

在 stride-4 网格上，由细节和语义预测每个位置指向 8 邻域的连接 logits。
边权同时依赖 i→j、j→i 的支持，以及深层语义相似性；因此未归一化边权对称。
无效图像外邻居的权重为零，不用循环位移，避免左右/上下边界串接。

保留 self-loop，使用 `h'=(h+sum(w_ij*h_j))/(1+sum(w_ij))`，执行两步共享传播，
再加回细节路径。它学习的是位置之间的连接可信度，不是固定方向卷积。
辅助目标为：两个相邻 cell 都触及前景时的邻接支持；padding/ignore cell 和图像外边排除。
这是网格级结构监督，不是精确骨架连通性的保证；近邻纹理误连仍需实测。
该实现为规则网格上的稠密软传播，不声称获得稀疏算子加速。

## 3. subpixel：子位置结构重建

对投影后的 stride-4 细节做 pixel-unshuffle，保留每个 2×2 单元的四个位置。
将其分解为共享均值与四个零和残差。分解/合并在该投影特征上可重建；
奇数尺寸采用右/下 replicate padding，重建后裁回原尺寸。

粗语义与均值生成一份共享的低分辨率 K/V 状态；四个子位置分别用
`Q_phase = Wq(semantic) + Wr(residual_phase)` 读取它。
读取结果加回各自残差后 pixel-shuffle，还原到 stride 4。
它改变了子位置如何读取上下文，不是只将四个位置平均后做插值。
不声称原图无损编码、整个网络严格平移等变，或简单小波分解本身构成创新。

## 初始化与训练目标

三个模块共享同样的窄投影、输出分辨率及主路径。
修正系数从 0.01 初始化，使路由/传播/读取第一步就能获得梯度，避免双零门控的停滞。
它们初始输出不是严格与 baseline 等价；将 structure_scale 设零时等于同权重主 head 的两段插值。
旧预训练参数键/形状保留，新模块参数随机初始化，不能把不同架构的相同 seed 当作完全相同初值。

主损失仍为 CE 0.5 + Dice 1.0；partition 额外增加 0.1×前景占据辅助损失，
connectivity 额外增加 0.1×邻接支持辅助损失；subpixel 没有辅助损失。
辅助 BCE 对正负质量分别归一化，支持无前景和全 ignore 输入。
因此本轮比较的是完整模块及其必要训练目标，不能把总 loss 数值直接横向比较。
如进一步论证收益来源，需要将辅助监督的影响单独说明，不能都归因于结构交互。

## 验证与训练

```bash
# 不需要数据集/预训练文件：6 个配置的真实 loss、前后向与参数量
python scripts/efficientvit_structure_20261104/verify.py --size 512

# 列出配置或只显示命令
python scripts/efficientvit_structure_20261104/batch_train.py --list
python scripts/efficientvit_structure_20261104/batch_train.py --variants all --dry-run

# 默认即 baseline、detail_resize、detail 和三个独立模块；3 seeds、40k、fp32
python scripts/efficientvit_structure_20261104/batch_train.py --gpus 0,1 --seeds 42,43,44

# 若要从零训练，所有选中版本统一关闭预训练
python scripts/efficientvit_structure_20261104/batch_train.py --gpus 0,1 --variants all --scratch

# 有真实数据/CUDA 时的两步执行检查，使用独立根目录
python scripts/efficientvit_structure_20261104/batch_train.py --gpus 0 --variants partition,connectivity,subpixel --scratch --seeds 42 --batch-size 2 --max-iters 2 --work-root work_dirs/structure_smoke
```

Windows 若 YAPF 可选语法缓存写入阻塞，仅验证脚本可加 `--no-yapf-cache`；不修改依赖包。
正式训练依赖实际数据与可用 CUDA；本次只完成构建、真实分割 loss 和梯度检查，没有启动 40k。

## 结果

新入口复用已有 GPU 队列、失败退出、目录防覆盖与 scheduler 长度对齐逻辑。
默认目录为 `work_dirs/efficientvit_structure_20261104/pretrained/fp32/seed*/variant/`。
每项结束后更新 `batch_summary.json`，包含最佳 mIoU 点的前景指标、多 seed 均值/SD、
相对同 seed baseline 的 IoU 差值。只比较相同初始化、精度和迭代长度的运行。

```bash
python scripts/efficientvit_structure_20261104/summarize.py work_dirs/efficientvit_structure_20261104/pretrained/fp32
```

重建写入独立 metrics_summary.json，可能包含中断运行，不能替代 batch_summary 的完成状态。
根据上一轮结论，三个新模块都应同时比较 detail_resize 与 detail，不能只比较旧 baseline。

回归测试包括独立归一化对显式 attention 的数值等价、路由梯度、连接对称与边界、
常量传播、奇数尺寸四相重建、CPU bfloat16、ignore 处理、预训练键兼容和真实辅助损失。
