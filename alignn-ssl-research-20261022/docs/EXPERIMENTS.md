# 正式实验协议

通过所选formal配置的 `self_supervised` 开关。false时复用该配置的公共、model和downstream参数，
没有预训练阶段，原子输入92维、hidden256、4+4层、300轮，主要优化参数与下游一致。
生成的有效监督配置会剥离pretrain和Z字段。下文三组对比仅适用于true。

## 三组配置

| 文件 | mask_kind | dim | z_dim | embedding_features |
|---|---|---:|---:|---:|
| `formal_learnable.json` | learnable | 256 | 64 | 64 |
| `formal_zero.json` | zero | 256 | 64 | 64 |
| `formal_zero_direct.json` | zero | 256 | 256 | 64 |

三份配置仅增加self_supervised流程开关，其余训练参数与原正式配置一致。前两组只改变 `mask_kind`；`zero_direct` 与 `zero` 的配置只改变 `z_dim`。实际后者会改变投影层、原子重构头和下游输入映射的结构及参数量，属于瓶颈设计对照。

## 公共设置

| 设置 | 值 |
|---|---|
| 随机种子 | 123 |
| 图 | k-nearest，cutoff=8 Å，cutoff_extra=3 Å，max_neighbors=12 |
| 输入特征 | 原子 CGCNN 92，距离 RBF 80，键角 RBF 40 |
| 每个主干 | 4 ALIGNN 层 + 4 GCN 层 |
| 预训练 / 下游 | 各 300 轮 |
| 有效批次 / 微批 | 64 / 32 |
| 优化器 | AdamW，分组 weight decay，bias/BN/norm 不衰减 |
| 最大学习率 | 0.001 |
| weight decay | 0.00001 |
| 学习率计划 | OneCycle，pct_start=0.3 |
| 梯度裁剪阈值 | null，不限制范数，但检查非有限梯度 |
| 掩码比例 | 原子与物理键均为 0.15 |
| 预训练任务权重 | 行号 CE、列号 CE、距离 CE 均为 1 |
| 断点频率 | 每 1 个优化器步 |
| 图缓存 | LMDB |
| 实际加载 worker | 0，启动器显式覆盖配置中的 4 |

## 比较口径

`learnable` 与 `zero` 用相同数据和配置比较可学习与固定零掩码。单次随机种子的差异不能据此宣称统计显著。

`zero` 与 `zero_direct` 比较经过 64 维瓶颈与 256 维直传的完整设计。压缩维度同时影响网络参数量，应一并说明。

与原版 ALIGNN 单主干监督基线比较时，需满足相同 ID、划分、结构和标签。自监督方案多了预训练预算、冻结编码器以及第二个主干，因此是完整方案比较，不能把结果全部归因于 mask。

正式比较应使用各自最佳验证模型在同一测试集上的预测。训练未完成时只能明确标注阶段验证比较，不能将验证 MAE 与另一组测试 MAE 当作同一指标相减。

## 本包的状态

代码包是方法实现及配置快照，不包含实验结果，也不宣称三个组都已训练完成。`tests/fixtures/zero_test_config.json` 为原有自动测试夹具，其小规模设置不是正式训练配置，测试会进一步覆盖为极小模型。
