"""Compare completed experiments on identical held-out materials."""
import csv
import json
import math
from pathlib import Path
import numpy as np
import torch
from .data import digest, sha256
from .state import source_fingerprint


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def compare(root, modes=None):
    root = Path(root)
    protocol = read_json(root / "protocol.json")
    modes = list(protocol["configs"]) if modes is None else list(modes)
    if not {"learnable", "zero"}.issubset(modes) or not set(modes).issubset(protocol["configs"]):
        raise ValueError("Comparison requires both primary groups and configured modes only")
    manifest = read_json(root / "data10000/manifest.json")
    sources = source_fingerprint()
    if sources != protocol["sources"]:
        raise ValueError("Source fingerprint changed before comparison")
    result, predictions, histories = {}, {}, {}
    test_ids = manifest["splits"]["test"]
    for mode in modes:
        cfg = protocol["configs"][mode]
        info, histories[mode] = {}, {}
        for stage in ("pretrain", "downstream"):
            run = root / mode / stage
            if not (run / "TRAINING_COMPLETE").exists():
                raise ValueError(f"Incomplete stage: {run}")
            signature = read_json(run / "config.json")
            assert signature["sources"] == sources and signature["config"] == cfg
            assert signature["manifest"] == digest(manifest)
            if stage == "downstream":
                assert signature["pretrained_sha256"] == sha256(root / mode / "pretrain/best.pt")
                assert signature["targets_sha256"] == sha256(root / "data10000/targets.json")
            history = read_json(run / "history.json")
            assert [row["epoch"] for row in history] == list(range(1, cfg[stage]["epochs"]+1))
            assert all(math.isfinite(row[split]["loss"]) for row in history for split in ("train", "val"))
            pool = "pretrain_train" if stage == "pretrain" else "train"
            expected_steps = math.ceil(len(manifest["splits"][pool])/cfg[stage]["batch_size"]) * cfg[stage]["epochs"]
            assert history[-1]["step"] == expected_steps
            summary = read_json(run / "run_summary.json")
            best = min(history, key=lambda row: row["val"]["loss"])
            assert summary["best_epoch"] == best["epoch"]
            assert summary["best_val_loss"] == best["val"]["loss"]
            info[stage] = summary
            histories[mode][stage] = history
        pre = torch.load(root / mode / "pretrain/best.pt", map_location="cpu", weights_only=False)
        down = torch.load(root / mode / "downstream/best.pt", map_location="cpu", weights_only=False)
        assert all(torch.equal(value, down["model"]["pretrained."+key]) for key, value in pre["model"].items())
        info["frozen_weights_identical"] = True
        info["mask_norms"] = {key: float(pre["model"][key].norm()) for key in ("atom_mask", "edge_mask")}
        assert all(value > 0 if mode == "learnable" else value == 0 for value in info["mask_norms"].values())
        del pre, down
        with (root / mode / "downstream/test_predictions.csv").open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        assert [row["id"] for row in rows] == test_ids
        y = np.array([float(row["target_eV_per_atom"]) for row in rows])
        p = np.array([float(row["prediction_eV_per_atom"]) for row in rows])
        targets = read_json(root / "data10000/targets.json")
        assert np.allclose(y, [targets[name] for name in test_ids], rtol=1e-6, atol=1e-6)
        assert np.isfinite(p).all()
        test = read_json(root / mode / "downstream/test_metrics.json")
        mae = float(np.abs(y-p).mean())
        r2 = float(1 - ((y-p)**2).sum()/((y-y.mean())**2).sum())
        assert test["n"] == len(test_ids)
        assert math.isclose(mae, test["mae_eV_per_atom"], rel_tol=1e-10)
        assert math.isclose(r2, test["r2"], rel_tol=1e-10)
        info["test"] = {**test, "rmse_eV_per_atom": float(np.sqrt(((y-p)**2).mean()))}
        result[mode], predictions[mode] = info, (y, p)
    assert all(np.array_equal(predictions[mode][0], predictions["zero"][0]) for mode in modes)
    delta = result["learnable"]["test"]["mae_eV_per_atom"] - result["zero"]["test"]["mae_eV_per_atom"]
    report = {"modes": result, "test_mae_learnable_minus_zero": delta,
              "limitation": "One seed; no statistical significance claim or supervised-only baseline comparison."}
    if "zero_direct" in modes:
        report["test_mae_zero_compressed_minus_direct"] = (
            result["zero"]["test"]["mae_eV_per_atom"] - result["zero_direct"]["test"]["mae_eV_per_atom"])
    (root / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (root / "paired_test_errors.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "target_eV_per_atom"] + [f"{m}_prediction" for m in modes]
                        + [f"{m}_absolute_error" for m in modes])
        for i, name in enumerate(test_ids):
            y = predictions["zero"][0][i]
            values = [predictions[m][1][i] for m in modes]
            writer.writerow([name, y, *values, *[abs(y-p) for p in values]])
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    colors = {"learnable": "#C76A2A", "zero": "#087E8B", "zero_direct": "#245881"}
    for mode in modes:
        color = colors[mode]
        for column, stage in enumerate(("pretrain", "downstream")):
            history = histories[mode][stage]
            for row, split in enumerate(("train", "val")):
                axes[row, column].plot([v["epoch"] for v in history], [v[split]["loss"] for v in history],
                                       label=mode, color=color)
                axes[row, column].set_title(f"{stage} / {split}")
                axes[row, column].set_ylabel("Sum of cross-entropies" if stage == "pretrain" else "MAE (eV/atom)")
    for ax in axes.flat:
        ax.set_xlabel("Epoch")
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(root / "training_curves.png", dpi=180)
    fig.savefig(root / "training_curves.pdf")
    plt.close(fig)
    lines = ["# 可学习向量、零向量与传递维度对比", "", "已完成的各组在同一份测试集上评估，分别使用验证集选出的最佳模型。", "",
             "| 方案 | 预训练最佳验证损失 | 下游最佳轮次 | 测试 MAE | 测试 RMSE | 测试 R² |",
             "|---|---:|---:|---:|---:|---:|"]
    for mode, info in result.items():
        test = info["test"]
        lines.append(f"| {mode} | {info['pretrain']['best_val_loss']:.6f} | {info['downstream']['best_epoch']} | "
                     f"{test['mae_eV_per_atom']:.6f} | {test['rmse_eV_per_atom']:.6f} | {test['r2']:.6f} |")
    lines += ["", f"可学习减零向量的测试 MAE 差值为 {delta:+.6f} eV/atom；负数表示可学习方案更低。",
              "各组下游冻结权重与其自身预训练权重逐张量一致；源码指纹、轮数、步数和测试预测指标均已核验。",
              "本次只有一个训练种子，不能宣称统计显著，也没有据此证明预训练优于从头监督训练。",
              "先运行的组承担首次图缓存构建开销，不宜直接比较各组耗时。"]
    if "zero_direct" in modes:
        lines += ["", f"零向量压缩 Z=64 减直接 Z=256 的测试 MAE 差值为 {report['test_mae_zero_compressed_minus_direct']:+.6f} eV/atom。",
                  "该对照比较完整的瓶颈设计，包括预训练重建头前的投影及下游输入映射，不是对同一个已训练编码器做事后裁剪。"]
    (root / "对比结果.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Comparison complete: {root / '对比结果.md'}", flush=True)
