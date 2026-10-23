# 环境配置

## 已查询的实际训练环境

2026-10-22 从正在使用的 WSL Ubuntu 环境只读查询，未修改环境：

| 组件 | 实际版本 |
|---|---|
| Python | 3.10.0 |
| PyTorch 运行时 | 2.2.1+cu121 |
| PyTorch CUDA runtime | 12.1 |
| DGL | 2.4.0+cu121 |
| NumPy | 1.26.4 |
| JARVIS tools | 2026.6.12 |
| Pydantic / pydantic-settings | 2.13.4 / 2.14.2 |
| LMDB | 2.2.1 |

其余直接依赖锁定在根目录 `requirements.txt`。这是实际环境的版本记录，不代表已在新环境完成安装验收。

## 新机器安装示例

下面面向 Linux x86_64 或 WSL2，并要求主机已安装支持 CUDA 的 NVIDIA 驱动。使用独立环境，不要在正在训练的环境中升级依赖。

```bash
conda create -n alignn_ssl python=3.10 -y
conda activate alignn_ssl
python -m pip install 'numpy==1.26.4'
python -m pip install 'torch==2.2.1' --index-url https://download.pytorch.org/whl/cu121
python -m pip install 'dgl==2.4.0+cu121' -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html
python -m pip install -r requirements.txt
python tools/check_environment.py --require-cuda
python -m pip check
```

PyTorch 的版本与 CUDA 安装源参考 [官方旧版本安装页](https://pytorch.org/get-started/previous-versions/)；DGL 的对应 wheel 来自 [官方 Torch 2.2 / CUDA 12.1 索引](https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html)。2026-10-22 已核对该索引包含 Python 3.10 的 DGL 2.4.0+cu121 wheel。

Python 3.10 的更新补丁版通常更适合新环境，但它与原训练的 3.10.0 不完全相同。跨环境不承诺逐位一致。先执行测试和小样本验收，再开始正式实验。

本包通过 `ssl_alignn/__init__.py` 加载 `vendor/alignn/alignn`，不需要另行 `pip install alignn`。上游 `setup.py` 声明过 `dgl<=1.1.1`，而实际训练环境使用 DGL 2.4.0+cu121；因此不直接套用该声明，也不将不同 DGL 版本混装。

## 使用现有环境

在已经能够训练的环境中，先激活该环境，然后检查本包的导入路径：

```bash
conda activate my_alignn
python tools/check_environment.py --require-cuda
```

`alignn_path` 应指向本包的 `vendor/alignn/alignn/__init__.py`。启动脚本默认使用当前环境的 `python`；如需指定解释器：

```bash
ALIGNN_PYTHON=/path/to/environment/bin/python bash run_formal_wsl.sh --check
```

## 常见问题

- 终端前缀为 `PS` 时，当前是 PowerShell；先执行 `wsl -d Ubuntu`，再运行本文的 Bash 命令。
- PowerShell 中的 `D:\...` 与 WSL 中的 `/mnt/d/...` 属于不同路径语法。
- `CUDA unavailable`：确认处于目标环境，再检查驱动、PyTorch 和 DGL 的 CUDA 版本。`--check` 不会开始训练。
- 本次原训练曾遇到数据加载子进程崩溃。分享版正式启动脚本默认启用单进程加载，实际 workers=0，覆盖行为记录到 `loader_runtime.jsonl`。
- 配置中的 workers=4 保留自原实验。实际单进程运行不改变模型、batch、样本顺序和优化器，但读取速度可能改变。
