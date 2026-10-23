# 自监督开关与原版 ALIGNN 监督对照

## 只改一个参数

打开所选组的配置，例如 `configs/formal_learnable.json`，修改顶层字段（保留其他训练参数）：

```json
"self_supervised": true
```

改成 `false` 就关闭自监督：

```json
"self_supervised": false
```

必须使用 JSON 布尔值 `true`/`false`，不要写成字符串 `"false"`。开关只在进程启动时读取，不会在训练过程中切换。

统一入口仍然是：

```bash
bash run_formal_wsl.sh --mode learnable --check
bash run_formal_wsl.sh --mode learnable
```

也可在兼容环境中用 `python run_formal_display.py --single-process-loader --mode learnable`。不要绕过统一入口直接调用底层 `run_formal.py` 或 `run.py`；这两个底层工具保留原有显式阶段接口，不负责此开关。

`--mode learnable/zero/zero_direct` 分别读取对应的 `formal_learnable.json`、`formal_zero.json`、`formal_zero_direct.json`。每份配置都包含 `self_supervised`，默认均为true。

不指定mode时默认both，要求learnable和zero的开关一致。两者都为true则依次运行两组自监督；两者都为false且有效监督参数相同，则只训练一次原版ALIGNN。若只修改了一份配置，明确指定相应mode即可。开关或监督参数冲突会在启动前报错，不会悄悄忽略配置。

## 两种行为

| 开关 | 启动后的行为 |
|---|---|
| true | 原有掩码预训练，再冻结编码器进行下游训练；`--mode` 选择自监督组别 |
| false | 只执行一次原版 ALIGNN 随机初始化监督训练，无预训练、无冻结编码器、无 Z 投影 |

关闭时直接调用包内原始 `ALIGNNAtomWise.forward()`，原子输入92维、hidden256、4 ALIGNN层+4 GCN层，图池化后输出形成能。不是简单地取消掩码，也不是把随机初始化的编码器冻结后继续串联第二套网络。

监督训练直接读取所选formal配置的公共参数、`model`及`downstream`，忽略`pretrain`和`model.z_dim`。默认保持300轮、有效batch64、microbatch32、AdamW、峰值lr=0.001、weight decay=1e-5、OneCycle升温比例0.3、seed123、每1个优化器步保存。实际加载worker=0。

不再维护独立的开关文件或监督配置文件。运行结果目录内仍会保存实际生效的监督配置快照，用于审计及恢复；不要把结果快照当作启动配置编辑。

## 结果位置与恢复

关闭后的默认输出是当前代码目录下的：

```text
runs/supervised_10000/supervised/
```

其中包含 `training_state.pt`、`best.pt`、`history.json`、`test_predictions.csv`、`test_metrics.json`。测试使用最低验证MAE对应模型。

该目录独立于旧自监督实验和旧原版监督基线，程序拒绝把纯监督结果写入含learnable/zero组的目录。

首次运行与异常恢复使用同一条命令。已有断点时自动恢复。主动暂停可执行：

```bash
bash run_formal_wsl.sh --mode learnable --pause
```

保持开关为false，等待PAUSED后，确认要恢复时删除自己发出的根目录 `runs/supervised_10000/STOP_REQUESTED`，再运行同一启动命令。不要删除权重或协议文件。

本机入口默认沿用原固定10,000材料数据。提交版默认从 `data/data10000` 读取，也可指定：

```bash
bash run_formal_wsl.sh --mode learnable --data /path/to/data10000 --out runs/supervised_10000
```

## 与历史原版结果比较的边界

关闭后的网络是原版ALIGNN，数据划分、主要优化设置与本次对比保持一致。数据读取、轮次顺序生成、恢复和记录使用本项目的训练框架，不是调用历史基线的整套 `alignn/train.py`。

因此可用来排查“同一框架下，去掉预训练和冻结是否仍表现较差”。它不承诺逐轮复现历史原版训练曲线：原版DataLoader的打乱实现与这里按 `seed + epoch` 生成顺序不同。

## 已完成的调用核验

`audit_alignn_call.py` 对相同原子图、线图、原子特征、距离、角度和相同权重，比较当前无掩码Backbone与原版完整forward。

2026-10-22在CPU上检查4个真实训练材料，正式hidden256、4+4层：

- train/eval输出最大绝对差为0。
- 已连接参数的梯度最大绝对差为0，双方未连接参数集合一致。
- 一步AdamW更新后的权重最大绝对差为0。
- 原版监督模型参数为4,026,753，全部参数允许训练。

原版末层角/边状态中有未进入最终读出的支路，相应末层归一化参数两边都可能没有梯度；核验检查这种情况也一致，没有将其误判为封装错误。

这支持“已检查的无掩码主干调用一致”，不证明预训练掩码目标合理或完整300轮训练必然一致。自监督效果较差还可能与冻结、64维瓶颈、重构目标和能量任务的适配有关，需要上述监督对照继续区分。

另一个源码细节：当前上游实际使用 `LayerNorm`。某些变量名叫bn、注释写BatchNorm，不代表这里真的使用BatchNorm，不能用批次运行统计漂移解释这些层。

## 复核命令

```bash
CUDA_VISIBLE_DEVICES='' python audit_alignn_call.py --formal-model --out outputs/call_audit.json
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -v
```

第一条默认使用合成材料；加 `--data /path/to/data10000` 后只读取前4个训练材料，不读取测试标签。数值核验不会启动正式训练。

本次流程适配位于 `ssl_alignn/` 外，原核心源码及 `run_formal.py` 保持原样。三组配置仅新增开关，交给旧自监督入口前会剥离该流程字段，因此本机已有自监督断点的有效训练配置与源码签名不变。真正的模型、优化器等参数修改仍会被恢复校验拒绝。开关默认保留true，尚未自动启动新的300轮监督实验。
