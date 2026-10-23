"""训练的外层显示适配器；不参与模型、损失、随机采样或断点保存。

为什么单独放在项目根目录？已有正式断点锁定了 ssl_alignn/*.py 和
run_formal.py 的源码指纹。只在运行期间替换这两个模块的打印入口及进度条，
就可以改善显示，同时保留原有训练源码、配置和断点校验。
"""
import builtins
from contextlib import contextmanager
import json
import re
import sys
import time

from tqdm import tqdm


class CompactBar:
    """保持训练器需要的接口，只展示 loss；每轮结束后清除临时进度条。"""

    def __init__(self, owner, total, description, initial=0, enabled=True):
        self.owner = owner
        self.description = description
        self.started = time.monotonic()
        self.closed = False
        # 正式入口把输出复制进 console.log。进度条只走终端，避免日志积累
        # 成千上万条带回车符的刷新记录；每轮指标仍由原来的 Tee 写入日志。
        stream = sys.stderr
        while hasattr(stream, "terminal"):
            stream = stream.terminal
        label = re.sub(r"^(pretrain|downstream) ", "", description)
        self.bar = tqdm(
            total=total, initial=initial, desc=label, file=stream,
            leave=False, dynamic_ncols=True, mininterval=1.0,
            disable=not enabled or not stream.isatty(),
            bar_format="{desc}: {percentage:3.0f}%|{bar:12}| "
                       "{n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
        )

    def set_postfix(self, refresh=False, **values):
        # lr 在每轮汇总中展示；step/saved 已可从断点和 history.json 查阅。
        # 去掉这些重复字段只改变画面，不改变每一步保存断点的频率。
        if "loss" in values:
            self.bar.set_postfix(loss=values["loss"], refresh=refresh)

    def update(self, n=1):
        self.bar.update(n)

    def close(self):
        if not self.closed:
            self.closed = True
            self.owner.elapsed[self.description] = time.monotonic() - self.started
            self.bar.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class TerminalDisplay:
    """把训练器已经算好的指标排成表；原始精度仍保存在 history.json。"""

    def __init__(self):
        self.elapsed = {}
        self.epochs = {}

    def progress_bar(self, total, description, initial=0, enabled=True):
        return CompactBar(self, total, description, initial, enabled)

    def print(self, *args, **kwargs):
        # 仅识别两个已知的 JSON 输出：启动配置、每轮汇总。
        # 报错、暂停提示、恢复位置和其他输出一律原样保留。
        value = None
        if len(args) == 1 and isinstance(args[0], str):
            try:
                value = json.loads(args[0])
            except (ValueError, TypeError):
                pass
        if isinstance(value, dict):
            if set(value) in ({"learnable", "zero"}, {"learnable", "zero", "zero_direct"}):
                return builtins.print(self.settings(value), **kwargs)
            if {"epoch", "step", "train", "val", "lr"} <= value.keys():
                return builtins.print(self.epoch_table(value), **kwargs)
        builtins.print(*args, **kwargs)

    def settings(self, configs):
        cfg = configs["learnable"]
        self.epochs = {s: cfg[s]["epochs"] for s in ("pretrain", "downstream")}
        rows = [f"Model: hidden={cfg['model']['dim']} | "
                f"embedding={cfg['model'].get('embedding_features', cfg['model']['dim'])} | "
                f"ALIGNN/GCN={cfg['model']['alignn_layers']}/{cfg['model']['gcn_layers']} | "
                f"checkpoint every {cfg['checkpoint_every_steps']} step(s)"]
        rows.append("Groups: " + " | ".join(
            f"{name}: Z={c['model'].get('z_dim', c['model']['dim'])}"
            for name, c in configs.items()))
        for stage in ("pretrain", "downstream"):
            c = cfg[stage]
            rows.append(f"{stage}: {c['epochs']} epochs | batch={c['batch_size']} "
                        f"(micro={c.get('microbatch_size', c['batch_size'])}) | max_lr={c['lr']:g}")
        if "zero_direct" in configs:
            rows.append("both = learnable + zero (Z=64); zero_direct (Z=256) runs only when selected.")
        return "\n".join(rows)

    def epoch_table(self, record):
        pretrain = "period_loss" in record["train"]
        stage = "pretrain" if pretrain else "downstream"
        epoch = record["epoch"]
        total = self.epochs.get(stage)
        counter = f"{epoch}/{total}" if total else str(epoch)
        rows = [f"Epoch {counter} | {stage} | lr={record['lr']:.3e}"]
        if pretrain:
            rows.append(f"{'Split':<6}" + " ".join(f"{s:>8}" for s in
                        ("Total", "RowCE", "ColCE", "DistCE")) + " " +
                        " ".join(f"{s:>6}" for s in ("Row%", "Col%", "Dist%")) +
                        f" {'Time(s)':>9}")
        else:
            # 下游 L1 loss 与 MAE 是同一个指标，避免重复两列相同的数。
            rows.append(f"{'Split':<6}{'MAE(eV/atom)':>13} {'Time(s)':>9}")
        for split in ("train", "val"):
            m = record[split]
            label = "Train" if split == "train" else "Val"
            prefix = f"{stage} {label} {epoch}/"
            durations = [v for k, v in self.elapsed.items() if k.startswith(prefix)]
            seconds = f"{sum(durations):.1f}" if durations else "--"
            if pretrain:
                losses = " ".join(f"{m[k]:8.4f}" for k in
                                  ("loss", "period_loss", "group_loss", "distance_loss"))
                accuracies = " ".join(f"{100*m[k]:6.1f}" for k in
                                      ("period_accuracy", "group_accuracy", "distance_accuracy"))
                rows.append(f"{label:<6}{losses} {accuracies} {seconds:>9}")
            else:
                rows.append(f"{label:<6}{m['mae_eV_per_atom']:13.6f} {seconds:>9}")
        # 只保存当前轮的计时；断点中的累计耗时仍完全由原训练器管理。
        self.elapsed.clear()
        return "\n".join(rows) + "\n"

    @contextmanager
    def installed(self, launcher, runner):
        """作用范围限定在本次入口调用，退出或异常时恢复原函数。"""
        missing = object()
        previous = []
        for module, name, replacement in (
            (launcher, "print", self.print),
            (runner, "print", self.print),
            (runner, "progress_bar", self.progress_bar),
        ):
            previous.append((module, name, getattr(module, name, missing)))
            setattr(module, name, replacement)
        try:
            yield self
        finally:
            for module, name, original in reversed(previous):
                if original is missing:
                    delattr(module, name)
                else:
                    setattr(module, name, original)


def demo():
    """人工示例，仅演示排版，不加载材料、模型或正式断点。"""
    display = TerminalDisplay()
    display.epochs = {"pretrain": 300, "downstream": 300}
    builtins.print("DISPLAY DEMO ONLY - synthetic values; no training started.\n")
    for stage, epoch in (("pretrain", 1), ("downstream", 92)):
        for phase in ("Train", "Val"):
            with display.progress_bar(10, f"{stage} {phase} {epoch}/300") as bar:
                for batch in range(10):
                    bar.set_postfix(loss=f"{1/(batch+1):.5f}")
                    bar.update()
                    time.sleep(0.12)
        if stage == "pretrain":
            metrics = dict(loss=2.1, period_loss=.4, group_loss=.6, distance_loss=1.1,
                           period_accuracy=.85, group_accuracy=.75, distance_accuracy=.65)
        else:
            metrics = dict(loss=.2, mae_eV_per_atom=.2)
        display.print(json.dumps(dict(epoch=epoch, step=10, lr=.001,
                                      train=metrics, val=metrics)), flush=True)

