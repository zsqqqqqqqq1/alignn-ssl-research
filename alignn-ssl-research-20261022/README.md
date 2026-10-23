# ALIGNN Self-Supervised Research

基于 NIST ALIGNN 的晶体原子与距离掩码预训练，以及冻结编码器的形成能预测。

## 自监督开关

在所选 `configs/formal_*.json` 中设置 `"self_supervised": true` 或 `false`。
`true` 保持自监督流程；`false` 直接训练随机初始化的原版单主干 ALIGNN，
沿用该配置的 model 与 downstream 参数，结果写入独立的 `runs/supervised_10000`。
例如修改 formal_learnable.json 后，执行 `bash run_formal_wsl.sh --mode learnable`。详见[开关说明与调用核验](docs/SELF_SUPERVISED_SWITCH.md)。

默认仍为true；未自动启动监督对照。


这是 **2026-10-22 的独立代码分享版**，适合导师审阅和作为 GitHub 仓库内容。它整理自实际训练代码，保留当前 hidden=256 的三组配置；旧的 hidden=64 实验、历史中文学习版、数据集和训练权重不在包内。

> 分享版更改了路径发现和启动入口，源码指纹随之变化。现有本地实验仍应使用原训练目录恢复；不要用本包接管旧断点。使用本包启动的新实验可以在本包内正常续训。

## 实验内容

| 组名 | 掩码向量 | 预训练 hidden | 传给下游的 Z | 下游 hidden |
|---|---|---:|---:|---:|
| `learnable` | 可学习向量 | 256 | 64 | 256 |
| `zero` | 固定零向量 | 256 | 64 | 256 |
| `zero_direct` | 固定零向量 | 256 | 256 | 256 |

- 原子输入直接使用 JARVIS/ALIGNN 的 CGCNN 92 维特征。原子及物理键对的掩码比例均为约 15%。
- 遮蔽距离的 80 维 RBF 表示，保留角度及图拓扑。预训练重构周期表行号、列号和距离离散标签。
- `learnable` 与 `zero` 的 256→64 投影在预训练中学习，然后与编码器一起冻结。
- 下游包含一个新训练的 ALIGNN 主干，并继续输入距离和角度。该方案与原版单主干监督训练的结构、训练预算不同。
- 每组预训练 300 轮、下游 300 轮。预训练按验证重构损失选模，下游按验证 MAE 选模，完成后用最佳下游模型评估测试集。
- 每个优化器更新后保存完整断点，包含模型、优化器、学习率调度和随机状态。终端显示批次进度和轮末指标表。

## 建议阅读顺序

1. [环境安装](docs/ENVIRONMENT.md)：实际训练环境版本及安装方法。
2. [数据准备](docs/DATA.md)：沿用固定的 10,000 个材料和原划分。
3. [启动、暂停与恢复](docs/RUNNING.md)：首次训练和中断恢复命令。
4. [模型与代码导读](docs/CODE_GUIDE.md)：函数职责、数据流、维度和选模方式。
5. [正式实验协议](docs/EXPERIMENTS.md)：分组、配置与对比边界。
6. [GitHub 上传说明](docs/GITHUB.md)：应该提交的文件及不应上传的生成物。
7. [来源、修改与验证范围](docs/RELEASE_NOTES.md)。

## 快速开始

以下命令在 **Linux / WSL 的 Bash 终端**运行，先按环境文档激活 Python 环境。

```bash
cd alignn-ssl-research-20261022
python tools/check_environment.py --require-cuda

# 选项 A：已有与本实验一致的 prepared data10000 文件夹
python tools/verify_dataset.py --data /path/to/data10000
bash run_formal_wsl.sh --mode learnable --data /path/to/data10000 --out runs/formal_10000
```

只有原始 POSCAR 与 `id_prop.csv` 时，先按固定清单恢复数据，**不重新随机抽样**：

```bash
python tools/prepare_fixed_data.py --source /path/to/JARVIS_dataset --out data/data10000
bash run_formal_wsl.sh --check
bash run_formal_wsl.sh --mode learnable
```

上面的 `/path/to/...` 是占位路径，必须替换为实际位置。项目路径含空格时加引号。

`--mode both` 依次运行 `learnable` 和 `zero`，不会自动启动 `zero_direct`。所有正式阶段均为 300 轮。完整命令见 [运行手册](docs/RUNNING.md)。

## 仓库结构

```text
configs/                 三组配置，均内置 self_supervised 开关
ssl_alignn/              实际自监督模型、数据、训练器与断点实现
supervised_model.py       直接调用原版 ALIGNN forward
supervised_runner.py      纯监督训练、验证及断点恢复
supervised_run.py         纯监督入口，单独结果目录
audit_alignn_call.py      原版与当前无掩码主干的数值核验
run_formal.py            自监督实验调度、协议校验、阶段恢复
run_formal_display.py    简洁终端显示入口
run_formal_wsl.sh        Linux/WSL 启动器，默认使用单进程数据加载
run.py                  显式路径的底层 prepare/pretrain/downstream/export 工具
dataset_utils.py         固定数据身份校验
tools/                  环境检查、固定数据重建、数据校验工具
tests/                  原有测试及分享版新增的数据迁移测试
data/splits/            材料 ID 划分、结构哈希与数据身份，不含结构和标签
vendor/alignn/          保持原样的上游运行时子集及其许可证
provenance/             源码哈希、上游来源和可迁移改动记录
docs/                   使用和方法说明
```

## 核验

解压后可先核对文件是否完整，这一步不需要训练依赖：

```bash
python tools/verify_package.py
```

不安装训练依赖，也能运行固定数据迁移测试：

```bash
python -m unittest discover -s tests -p test_portable_data.py -v
```

在兼容的训练环境中执行完整测试。测试使用合成小材料和临时目录；以下命令禁用 GPU：

```bash
CUDA_VISIBLE_DEVICES='' MPLBACKEND=Agg python -m unittest discover -s tests -v
```

本次分享版的已执行检查与尚未执行项见 [验证记录](docs/VALIDATION.md)。不要把“包含测试文件”理解为“已在全新机器完成所有测试”。

## 上游与授权说明

使用的是 [NIST / usnistgov ALIGNN](https://github.com/usnistgov/alignn)，不是 Alibaba 项目。包中保留所用上游源文件及 [NIST 原始许可声明](vendor/alignn/LICENSE.rst)。本研究新增代码尚未指定对外开源许可证，见 [LICENSE.md](LICENSE.md)。上传 GitHub 不会自动赋予新增代码 MIT 等开源许可。
