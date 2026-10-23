# 固定数据集与复用方法

## 数据内容

任务是每原子形成能预测，单位 eV/atom。正式实验使用同一份 10,000 材料：

| 子集 | 数量 | 用途 |
|---|---:|---|
| train | 8,000 | 下游监督训练 |
| val | 1,000 | 下游选模 |
| test | 1,000 | 完成训练后的最终测试 |
| pretrain_train | 7,200 | 从 train 内部划分，预训练 |
| pretrain_val | 800 | 从 train 内部划分，预训练选模 |

`pretrain_train` 与 `pretrain_val` 的并集恰为 `train`，不包含监督 val/test。预训练批次不读取 `targets.json`。数据准备阶段为保证实验身份会校验标签文件，这与把形成能标签输入预训练模型不同。

分享包保留 `data/splits/manifest_10000_seed123.json`，包括固定 ID、顺序和每个结构的 SHA-256；`dataset_identity.json` 记录 manifest 和 targets 的字节哈希。不附结构文件、形成能标签、图缓存或断点。

材料 ID 是 `POSCAR-JVASP-....vasp` 形式。仅从在线数据库重新获取同名材料，并不能保证与历史实验使用的结构和标签完全相同。使用原始数据快照或经过校验的 prepared 数据；不一致时程序会报错。

## 方式 A：沿用已有 prepared 数据

需要以下目录结构：

```text
data10000/
  manifest.json
  targets.json
  structures/
    POSCAR-JVASP-....vasp
```

```bash
python tools/verify_dataset.py --data /path/to/data10000
bash run_formal_wsl.sh --mode learnable --data /path/to/data10000 --out runs/formal_10000
```

启动器首次运行会把必要数据复制到输出目录的 `data10000/`，核验结构哈希并保持原文件内容。不会复制源目录中的图缓存和权重。后续继续使用输出目录内的数据，`--data` 只提供首次准备的数据来源。

## 方式 B：从原始 POSCAR 目录恢复

原始目录必须含 `id_prop.csv` 和结构文件。CSV 没有表头，每行两列：

```text
POSCAR-JVASP-25390.vasp,-0.00094
```

```bash
python tools/prepare_fixed_data.py --source /path/to/JARVIS_dataset --out data/data10000
python tools/verify_dataset.py --data data/data10000
```

工具严格按包中的已有划分选取材料，不调用随机采样。它先检查标签及全部结构，再创建临时目录、复制并复核，最后将完整目录提交到目标位置。遇到中断留下的 `.preparing` 目录时会停止，请先检查该目录，再决定是否清理或保留。

本方法需要原始快照，分享包本身无法凭空恢复结构或形成能标签。数据可单独传给导师；上传公开仓库前应单独确认所用数据来源及其分发条件。

## 与原版 ALIGNN 保持相同输入

原版监督训练也必须使用这份 `train/val/test` 的 ID 列表及对应标签，不能让上游重新随机划分。若对照程序使用 `id_train/id_val/id_test` 键，按如下对应关系导出，保留列表顺序：

| 本包 manifest 键 | 原版 split 键 |
|---|---|
| splits.train | id_train |
| splits.val | id_val |
| splits.test | id_test |

本包提供了原版单主干监督分支，设置 所选 `configs/formal_*.json` 的 `self_supervised=false`，按同一套划分运行即可。
该分支直接调用上游模型forward，训练调度沿用本项目框架，并未包含上游整套 `alignn/train.py`。
不能将自监督JSON改名后直接交给另一个上游训练器，两者配置schema不同。

## 新数据实验

`run.py prepare --source ... --split ... --out ... --size ...` 是底层通用采样工具，需要原始监督划分覆盖整个输入 CSV。它会建立新的实验子集，不等价于恢复本次 10,000 材料。新数据应使用新的输出目录并单独报告，不能与当前固定基线直接混比。
