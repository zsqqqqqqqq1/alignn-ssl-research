# 正式运行、暂停与恢复

所有示例从仓库根目录执行，使用已配置好的 Linux/WSL Python 环境。

使用所选 `configs/formal_*.json` 的 `self_supervised` 开关。本页原有分组操作对应true。
关闭时只运行原版监督模型，`--mode`选择使用哪一份配置，默认输出变为 `runs/supervised_10000`。
只修改可学习组时使用 `--mode learnable`；默认both要求两份配置的开关一致，关闭后还要求有效监督参数一致。
完整说明见[开关手册](SELF_SUPERVISED_SWITCH.md)。

## 1. 检查与准备

```bash
python tools/check_environment.py --require-cuda
bash run_formal_wsl.sh --check
python tools/verify_dataset.py --data /path/to/data10000
bash run_formal_wsl.sh --prepare-only --data /path/to/data10000 --out runs/formal_10000
```

`--check` 只检查配置、导入和 CUDA，不验证数据文件，也不开始训练。
`--prepare-only` 复制并校验固定数据、保存配置协议，但不开始训练。

## 2. 按需启动组别

```bash
# 当前重点：可学习向量，Z64
bash run_formal_wsl.sh --mode learnable --data /path/to/data10000 --out runs/formal_10000

# 稍后运行：零向量，Z64
bash run_formal_wsl.sh --mode zero --data /path/to/data10000 --out runs/formal_10000

# 单独手动启动：零向量，Z256 直传
bash run_formal_wsl.sh --mode zero_direct --data /path/to/data10000 --out runs/formal_10000
```

上面是分别执行的命令，不要同时启动它们写入同一输出目录。也可以一次顺序执行前两组：

```bash
bash run_formal_wsl.sh --mode both --data /path/to/data10000 --out runs/formal_10000
```

`both` 仅指 `learnable` 和 `zero`。每一组先预训练 300 轮，再冻结最优预训练模型做下游 300 轮。`zero_direct` 不会自动跟随启动。

## 3. 实时显示

正式入口包含每批次进度条与每轮 Train/Val 表。预训练显示总损失、三个交叉熵及准确率；下游显示形成能 MAE。每阶段 `console.log` 同时保存输出。

只看显示效果，可运行下列演示，它使用合成数值，不加载正式数据和模型：

```bash
bash run_formal_wsl.sh --demo
```

## 4. 正常暂停

可在训练终端按一次 Ctrl+C，等待出现 `PAUSED`。也可以在同一环境的另一个终端发出请求：

```bash
bash run_formal_wsl.sh --out runs/formal_10000 --pause
```

后者创建根目录 `STOP_REQUESTED`，并通知当前阶段。程序会在安全边界保存后暂停。如果正在验证，需要等待验证结束。不要在正在保存时强制关闭 WSL。

## 5. 同一份分享版实验恢复

异常中断后，重新执行原来的启动命令，程序根据已有完整断点继续，不从第 1 轮开始。已完成阶段会跳过。

主动 `--pause` 后，确认当前进程已经停止，并且你确实准备恢复，再仅删除根目录的暂停请求：

```bash
rm -- runs/formal_10000/STOP_REQUESTED
bash run_formal_wsl.sh --mode learnable --data /path/to/data10000 --out runs/formal_10000
```

如果是 Ctrl+C 或断电中断，根目录可能没有这个文件，此时直接续训。不要删除 `training_state.pt`、`protocol.json`、配置或数据哈希文件。

恢复要求模型源码、配置、数据、PyTorch 版本和设备等协议一致。报 `Protocol changed` 或 `Resume refused` 时，先检查差异，不要绕过校验。已经完成的原始本机实验必须继续用原训练目录，不能用本分享版接管。

## 6. 输出文件

```text
runs/formal_10000/
  protocol.json             三组配置与程序协议
  data10000/                固定数据副本和按需生成的缓存
  loader_runtime.jsonl      实际单进程加载记录
  learnable/
    pretrain/
      training_state.pt     最近提交的完整恢复状态
      best.pt               验证重构总损失最小的模型
      history.json          已完成轮次记录
      run_summary.json
      TRAINING_COMPLETE
    downstream/
      training_state.pt
      best.pt               验证 MAE 最小的模型
      test_predictions.csv  训练完成后最佳模型的测试预测
      test_metrics.json
      TRAINING_COMPLETE
  zero/                     同上
  zero_direct/              显式运行后才生成
```

`best.pt` 用于选模/下游；`training_state.pt` 用于最近进度恢复，二者不能互换。

两组 Z64 均完成后，程序校验协议并生成 `comparison.json`、`paired_test_errors.csv` 等对比结果。只有 `learnable` 完成时，其独立测试结果也已在该组下游目录内，不需要等待零向量。

## 7. 导出原子表示（可选）

```bash
python run.py export --config configs/formal_learnable.json \
  --data runs/formal_10000/data10000 \
  --pretrained runs/formal_10000/learnable/pretrain/best.pt \
  --out outputs/learnable_z --device cuda
```

每个材料输出 `N_atoms × 64` 的 Z，同时保留元素、原子索引和结构哈希。使用直传配置及对应预训练模型则输出 256 维。导出过程没有额外随机投影。
