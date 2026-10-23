"""Single-process reference trainer with batch-boundary crash recovery."""
import csv
import json
import math
from pathlib import Path
import random
import signal
import time

from filelock import FileLock
import numpy as np
import torch
from tqdm import tqdm
from .data import CrystalData, digest, sha256
from .model import MaskedModel, FrozenDownstream, sample_masks, reconstruction_loss
from .state import atomic_save, rng_state, restore_rng, seed_all, source_fingerprint, tensor_hash


def validate_config(cfg):
    # 新实验不再限制 hidden 小于 92；两种宽度独立校验，避免把 256 维
    # 隐藏状态误当作 256 维 RBF 中间嵌入。旧小型测试仍可省略该可选字段。
    for name, width in (("dim", cfg["model"]["dim"]),
                        ("z_dim", cfg["model"].get("z_dim", cfg["model"]["dim"])),
                        ("embedding_features", cfg["model"].get("embedding_features", cfg["model"]["dim"]))):
        if type(width) is not int or width <= 1:
            raise ValueError(f"model.{name} must be an integer greater than 1")
    for key in ("node_ratio", "edge_ratio"):
        if not 0 < cfg["pretrain"][key] <= 1:
            raise ValueError(key)
    if cfg["pretrain"]["mask_kind"] not in ("zero", "learnable"):
        raise ValueError("mask_kind")
    if len(cfg["pretrain"]["loss_weights"]) != 3 or any(w <= 0 for w in cfg["pretrain"]["loss_weights"]):
        raise ValueError("Three positive loss weights required")
    if min(cfg["model"]["alignn_layers"], cfg["model"]["gcn_layers"]) < 1:
        raise ValueError("Both graph layer types are required")
    if cfg["checkpoint_every_steps"] < 1:
        raise ValueError("checkpoint_every_steps must be positive")
    for stage in ("pretrain", "downstream"):
        if cfg[stage]["epochs"] < 1 or cfg[stage]["batch_size"] < 1:
            raise ValueError("Positive epochs and batch size required")
        micro = cfg[stage].get("microbatch_size", cfg[stage]["batch_size"])
        if micro < 1 or cfg[stage]["batch_size"] % micro:
            raise ValueError("batch_size must be divisible by microbatch_size")


def progress_bar(total, description, initial=0, enabled=True):
    """One terminal line: completed batches, elapsed time, ETA and live metrics."""
    return tqdm(total=total, initial=initial, desc=description, unit="batch",
                dynamic_ncols=True, mininterval=0.5, disable=not enabled)


def optimizer_parameters(model, grouped_decay):
    # Match the original formal ALIGNN optimizer: bias/BN/norm have no decay.
    if not grouped_decay:
        return [p for p in model.parameters() if p.requires_grad]
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            target = no_decay if any(word in name for word in ("bias", "bn", "norm")) else decay
            target.append(parameter)
    return [{"params": decay}, {"params": no_decay, "weight_decay": 0.0}]


def backward_batch(model, data, items, stage, cfg, device, mask_seed):
    """Accumulate microbatches, weighting by the actual supervised item counts.

    A pretraining task counts masked atoms/bonds, not crystals. Dividing every
    microbatch loss by two would be wrong when their mask counts differ.
    BatchNorm still observes each microbatch, just as in the formal baseline.
    """
    masks, totals = None, None
    if stage == "pretrain":
        masks = sample_masks({"x": torch.empty(0),
                              "node_counts": [v["g"].num_nodes() for _, v in items],
                              "pair_ids": [v["pair_ids"] for _, v in items]},
                             cfg["node_ratio"], cfg["edge_ratio"], mask_seed)
        totals = [int(masks[0].sum()), int(masks[0].sum()), int(masks[1].sum())]
    micro = cfg.get("microbatch_size", cfg["batch_size"])
    stats = empty_stats()
    node_offset = edge_offset = 0
    for start in range(0, len(items), micro):
        batch = data.collate(items[start:start + micro], device)
        if stage == "pretrain":
            n, e = batch["g"].num_nodes(), batch["g"].num_edges()
            local_masks = (masks[0][node_offset:node_offset+n].to(device),
                           masks[1][edge_offset:edge_offset+e].to(device))
            counts = [int(local_masks[0].sum()), int(local_masks[0].sum()), int(local_masks[1].sum())]
            weights = [w * count / total for w, count, total in zip(cfg["loss_weights"], counts, totals)]
            loss, losses, counts, correct = reconstruction_loss(model, batch, local_masks, weights)
            node_offset, edge_offset = node_offset+n, edge_offset+e
        else:
            loss, losses, counts, correct = step_loss(model, batch, stage, cfg, mask_seed)
            loss = loss * len(batch["names"]) / len(items)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss; last committed checkpoint retained")
        loss.backward()
        add_stats(stats, losses, counts, correct)
    return stats


def load_pretrained(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("stage") != "pretrain" or "model_config" not in checkpoint:
        raise ValueError("Use the pretraining best.pt file")
    model = MaskedModel(checkpoint["model_config"], checkpoint["mask_kind"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device), checkpoint


def empty_stats():
    return {"sums": [0., 0., 0.], "counts": [0, 0, 0], "correct": [0, 0, 0]}


def add_stats(stats, losses, counts, correct):
    for i in range(len(losses)):
        stats["sums"][i] += losses[i] * counts[i]
        stats["counts"][i] += counts[i]
        stats["correct"][i] += correct[i]


def summarize(stats, stage, weights):
    means = [s / max(n, 1) for s, n in zip(stats["sums"], stats["counts"])]
    if stage == "downstream":
        return {"loss": means[0], "mae_eV_per_atom": means[0]}
    result = {"loss": sum(a * b for a, b in zip(means, weights))}
    for i, key in enumerate(("period", "group", "distance")):
        result[key + "_loss"] = means[i]
        result[key + "_accuracy"] = stats["correct"][i] / max(stats["counts"][i], 1)
    return result


def step_loss(model, batch, stage, cfg, mask_seed):
    if stage == "pretrain":
        masks = sample_masks(batch, cfg["node_ratio"], cfg["edge_ratio"], mask_seed)
        return reconstruction_loss(model, batch, masks, cfg["loss_weights"])
    prediction = model(batch)
    loss = (prediction - batch["target"]).abs().mean()
    return loss, [float(loss.detach())], [len(batch["names"])], [0]


@torch.no_grad()
def evaluate(model, data, names, stage, cfg, device, progress=True, description="Validation"):
    model.eval()
    stats = empty_stats()
    size = cfg.get("microbatch_size", cfg["batch_size"])
    with progress_bar(math.ceil(len(names)/size), description, enabled=progress) as bar:
        for index, items in enumerate(data.iter_items(names, size)):
            batch = data.collate(items, device)
            loss, losses, counts, correct = step_loss(model, batch, stage, cfg, 9_000_000 + index)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite validation loss")
            add_stats(stats, losses, counts, correct)
            bar.set_postfix(loss=f"{summarize(stats, stage, cfg.get('loss_weights', [1, 1, 1]))['loss']:.5f}", refresh=False)
            bar.update()
    return summarize(stats, stage, cfg.get("loss_weights", [1, 1, 1]))


def train(cfg, dataset, out, stage, device, resume=False, pretrained=None, stop_after_steps=None):
    validate_config(cfg)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    with FileLock(str(out / ".run.lock"), timeout=0):
        return _train(cfg, dataset, out, stage, device, resume, pretrained, stop_after_steps)


def _train(cfg, dataset, out, stage, device, resume, pretrained, stop_after_steps):
    checkpoint_path = out / "training_state.pt"
    if checkpoint_path.exists() and not resume:
        raise FileExistsError("Existing run; pass --resume or choose a new directory")
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    if not resume and (out / "config.json").exists():
        raise FileExistsError("Incomplete existing run; choose a new directory")
    stage_cfg = cfg[stage]
    seed_all(cfg["seed"])
    torch.set_num_threads(cfg["cpu_threads"])
    sources = source_fingerprint()
    data = CrystalData(dataset, cfg["graph"], sources, supervised=stage == "downstream",
                       use_lmdb=cfg.get("use_lmdb", False), num_workers=cfg.get("num_workers", 0),
                       pin_memory=cfg.get("pin_memory", False))
    splits = data.manifest["splits"]
    names = splits["pretrain_train"] if stage == "pretrain" else splits["train"]
    validation = splits["pretrain_val"] if stage == "pretrain" else splits["val"]
    frozen_hash = None
    if stage == "pretrain":
        model = MaskedModel(cfg["model"], cfg["pretrain"]["mask_kind"]).to(device)
    else:
        if not pretrained:
            raise ValueError("Downstream requires --pretrained path/to/pretrain/best.pt")
        encoder, pre = load_pretrained(pretrained, device)
        if pre["manifest_hash"] != digest(data.manifest) or pre["model_config"] != cfg["model"] or pre["graph_config"] != cfg["graph"]:
            raise ValueError("Pretrained encoder dataset/graph/dimension protocol mismatch")
        model = FrozenDownstream(encoder, cfg["model"]).to(device)
        frozen_hash = tensor_hash(model.pretrained)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(optimizer_parameters(model, cfg.get("group_decay", False)),
                                  lr=stage_cfg["lr"], weight_decay=stage_cfg["weight_decay"])
    batches = math.ceil(len(names) / stage_cfg["batch_size"])
    total_steps = batches * stage_cfg["epochs"]
    if total_steps < 4:
        raise ValueError("OneCycle smoke run requires at least 4 optimizer steps")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=stage_cfg["lr"], total_steps=total_steps,
        pct_start=stage_cfg["warmup_fraction"])
    signature = {"config": cfg, "stage": stage, "manifest": digest(data.manifest),
                 "sources": sources, "pretrained_sha256": sha256(pretrained) if pretrained else None,
                 "targets_sha256": sha256(Path(dataset) / "targets.json") if stage == "downstream" else None,
                 "torch": torch.__version__, "device": str(device)}
    state = {"epoch": 0, "next_batch": 0, "step": 0, "best_loss": float("inf"),
             "best_model": None, "best_epoch": None, "history": [], "stats": empty_stats(), "seconds": 0.}
    if resume:
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if saved["signature"] != signature:
            raise ValueError("Resume refused: config, data, code, checkpoint, runtime or device changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        state = saved["progress"]
        restore_rng(saved["rng"])
    else:
        (out / "config.json").write_text(json.dumps(signature, indent=2), encoding="utf-8")
    paused = [False]
    old_handlers = {}
    def request_stop(signum, frame):
        paused[0] = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig, request_stop)
    last_time = time.monotonic()

    def save():
        nonlocal last_time
        now = time.monotonic()
        state["seconds"] += now - last_time
        last_time = now
        atomic_save({"signature": signature, "model": model.state_dict(),
                     "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                     "rng": rng_state(), "progress": state}, checkpoint_path)

    def publish_best():
        if state["best_model"] is not None:
            atomic_save({"stage": stage, "model": state["best_model"],
                         "model_config": cfg["model"], "graph_config": cfg["graph"],
                         "mask_kind": cfg["pretrain"]["mask_kind"],
                         "manifest_hash": digest(data.manifest), "sources": sources,
                         "epoch": state["best_epoch"], "validation_loss": state["best_loss"]}, out / "best.pt")

    print(f"{stage}: epochs={stage_cfg['epochs']}, effective batch={stage_cfg['batch_size']}, "
          f"microbatch={stage_cfg.get('microbatch_size', stage_cfg['batch_size'])}, "
          f"steps/epoch={batches}, checkpoint every {cfg['checkpoint_every_steps']} optimizer step(s), "
          f"device={device}", flush=True)
    if resume:
        print(f"RESUME: completed epochs={state['epoch']}, next batch={state['next_batch']}, "
              f"committed step={state['step']}", flush=True)
    try:
        save()  # A valid checkpoint exists even before the first optimizer step.
        if (out / "PAUSED").exists():
            (out / "PAUSED").unlink()
        while state["epoch"] < stage_cfg["epochs"]:
            epoch = state["epoch"]
            order = list(names)
            random.Random(cfg["seed"] + epoch).shuffle(order)
            model.train()
            first = state["next_batch"]
            with progress_bar(batches, f"{stage} Train {epoch+1}/{stage_cfg['epochs']}",
                              initial=first, enabled=cfg.get("progress", True)) as bar:
                iterator = data.iter_items(order[first * stage_cfg["batch_size"]:], stage_cfg["batch_size"])
                for index, items in enumerate(iterator, start=first):
                    optimizer.zero_grad(set_to_none=True)
                    partial = backward_batch(model, data, items, stage, stage_cfg, device,
                                             cfg["seed"] + epoch * batches + index)
                    limit = cfg.get("gradient_clip")
                    torch.nn.utils.clip_grad_norm_(params, float("inf") if limit is None else limit,
                                                  error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    for key in ("sums", "counts", "correct"):
                        state["stats"][key] = [a+b for a, b in zip(state["stats"][key], partial[key])]
                    state["step"] += 1
                    state["next_batch"] = index + 1
                    stop = paused[0] or (out / "STOP_REQUESTED").exists() or (
                        stop_after_steps is not None and state["step"] >= stop_after_steps)
                    committed = state["step"] % cfg["checkpoint_every_steps"] == 0 or stop
                    if committed:
                        save()
                    metrics = summarize(state["stats"], stage, stage_cfg.get("loss_weights", [1, 1, 1]))
                    bar.set_postfix(loss=f"{metrics['loss']:.5f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                                    step=state["step"], saved=state["step"] if committed else "pending", refresh=False)
                    bar.update()
                    if stop:
                        (out / "PAUSED").write_text(f"Committed step {state['step']}\n")
                        if (out / "STOP_REQUESTED").exists():
                            (out / "STOP_REQUESTED").unlink()
                        publish_best()
                        print(f"PAUSED at step {state['step']}", flush=True)
                        return state
            train_metrics = summarize(state["stats"], stage, stage_cfg.get("loss_weights", [1, 1, 1]))
            val_metrics = evaluate(model, data, validation, stage, stage_cfg, device,
                                   cfg.get("progress", True), f"{stage} Val {epoch+1}/{stage_cfg['epochs']}")
            if frozen_hash is not None and tensor_hash(model.pretrained) != frozen_hash:
                raise AssertionError("Frozen encoder changed")
            if val_metrics["loss"] < state["best_loss"]:
                state["best_loss"] = val_metrics["loss"]
                state["best_epoch"] = epoch + 1
                state["best_model"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            record = {"epoch": epoch + 1, "step": state["step"], "train": train_metrics, "val": val_metrics,
                      "lr": optimizer.param_groups[0]["lr"]}
            state["history"].append(record)
            state["epoch"], state["next_batch"], state["stats"] = epoch + 1, 0, empty_stats()
            save()
            publish_best()
            (out / "history.json").write_text(json.dumps(state["history"], indent=2))
            print(json.dumps(record), flush=True)
            # Also honor a pause requested during validation, including the
            # last epoch, before testing or starting another training stage.
            if paused[0] or (out / "STOP_REQUESTED").exists():
                (out / "PAUSED").write_text(f"Committed step {state['step']}\n")
                (out / "STOP_REQUESTED").unlink(missing_ok=True)
                print(f"PAUSED after epoch {state['epoch']}", flush=True)
                return state
        publish_best()
        # Test structures are touched only once, after validation-based model selection.
        if stage == "downstream" and not (out / "test_metrics.json").exists():
            model.load_state_dict(state["best_model"])
            test_downstream(model, data, splits["test"], stage_cfg.get("microbatch_size", stage_cfg["batch_size"]),
                            device, out, cfg.get("progress", True))
        (out / "history.json").write_text(json.dumps(state["history"], indent=2))
        (out / "run_summary.json").write_text(json.dumps({
            "stage": stage, "best_epoch": state["best_epoch"], "best_val_loss": state["best_loss"],
            "seconds": state["seconds"], "trainable_parameters": sum(p.numel() for p in params),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "frozen_hash": frozen_hash, "cuda_peak_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else 0}, indent=2))
        (out / "TRAINING_COMPLETE").write_text("Completed successfully\n")
        return state
    finally:
        # Never serialize partially executed backward/optimizer work in an exception handler.
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


@torch.no_grad()
def test_downstream(model, data, names, batch_size, device, out, progress=True):
    model.eval()
    rows = []
    with progress_bar(math.ceil(len(names)/batch_size), "Test", enabled=progress) as bar:
        for items in data.iter_items(names, batch_size):
            batch = data.collate(items, device)
            prediction = model(batch).cpu().tolist()
            rows.extend(zip(batch["names"], batch["target"].cpu().tolist(), prediction))
            bar.update()
    y, p = np.array([r[1] for r in rows]), np.array([r[2] for r in rows])
    if not np.isfinite(p).all():
        raise FloatingPointError("Nonfinite test predictions")
    mae, mad = float(abs(y - p).mean()), float(abs(y - y.mean()).mean())
    denominator = float(((y - y.mean()) ** 2).sum())
    metrics = {"n": len(rows), "mae_eV_per_atom": mae, "mad_eV_per_atom": mad,
               "mae_over_mad": mae / mad if mad else None,
               "mad_over_mae": mad / mae if mae else None,
               "r2": 1 - float(((p-y)**2).sum()) / denominator if denominator else None}
    with open(out / "test_predictions.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "target_eV_per_atom", "prediction_eV_per_atom"])
        writer.writerows(rows)
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2))


@torch.no_grad()
def export_z(cfg, dataset, pretrained, out, device):
    model, checkpoint = load_pretrained(pretrained, device)
    model.eval().requires_grad_(False)
    data = CrystalData(dataset, cfg["graph"], source_fingerprint())
    if checkpoint["manifest_hash"] != digest(data.manifest) or checkpoint["graph_config"] != cfg["graph"]:
        raise ValueError("Export dataset/graph protocol mismatch")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cp_hash = sha256(pretrained)
    for split in ("train", "val", "test"):
        for name in data.manifest["splits"][split]:
            batch = data.batch([name], device)
            z = model.encode(batch).cpu()
            atomic_save({"id": name, "Z": z, "node_index": torch.arange(z.shape[0]),
                         "elements": data.get(name)["elements"],
                         "structure_sha256": data.manifest["structure_sha256"][name],
                         "graph_config_hash": digest(cfg["graph"]), "checkpoint_sha256": cp_hash}, out / (name + ".pt"))
    (out / "manifest.json").write_text(json.dumps({"checkpoint_sha256": cp_hash,
        "dataset_hash": digest(data.manifest),
        "dim": checkpoint["model_config"].get("z_dim", checkpoint["model_config"]["dim"])}, indent=2))
