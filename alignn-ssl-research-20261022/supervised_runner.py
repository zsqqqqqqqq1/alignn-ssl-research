"""Supervised control: original ALIGNN forward with the existing trainer protocol.

The training/checkpoint loop is copied unchanged from ssl_alignn.runner except
for model creation and the additive source fingerprint. Shared loss, optimizer,
microbatch accumulation, evaluation and final test functions are reused.
"""
import copy
import json
import math
import random
import signal
import time
from pathlib import Path
import torch
from filelock import FileLock
from ssl_alignn.data import digest, sha256
from ssl_alignn.runner import (optimizer_parameters, backward_batch, empty_stats,
    summarize, evaluate, progress_bar, test_downstream, validate_config as _validate)
from ssl_alignn.state import (atomic_save, rng_state, restore_rng, seed_all,
    source_fingerprint as _source_fingerprint, tensor_hash)
from supervised_model import OriginalSupervisedModel, OriginalCrystalData as CrystalData


def validate_config(cfg):
    # Reuse the existing structural checks without adding a pretraining stage.
    probe=copy.deepcopy(cfg)
    probe['pretrain']={**probe['downstream'], 'mask_kind':'zero',
        'node_ratio':0.15,'edge_ratio':0.15,'loss_weights':[1.,1.,1.]}
    _validate(probe)


def source_fingerprint():
    result=_source_fingerprint()
    root=Path(__file__).resolve().parent
    for name in ('supervised_model.py','supervised_runner.py','supervised_run.py',
                 'experiment_options.py','run_formal_display.py'):
        result['supervised_control/'+name]=sha256(root/name)
    result['supervised_control/method']='original-ALIGNN-forward-no-pretraining-v1'
    return result


def train(cfg, dataset, out, stage, device, resume=False, pretrained=None, stop_after_steps=None):
    validate_config(cfg)
    if stage != "downstream" or pretrained is not None:
        raise ValueError("Pure supervised training does not accept pretraining or pretrained weights")
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
    model = OriginalSupervisedModel(cfg['model']).to(device)
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
                         "method": "supervised_original",
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

