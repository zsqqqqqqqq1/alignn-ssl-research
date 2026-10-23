"""User-started 300-epoch comparison; never launched on import."""
import argparse
import copy
from contextlib import redirect_stdout, redirect_stderr
import json
import os
import shutil
from pathlib import Path
import sys
from datetime import datetime, timezone, timedelta

from filelock import FileLock
import torch
from ssl_alignn.data import sha256, digest
from ssl_alignn.runner import train, validate_config
from ssl_alignn.state import source_fingerprint

CODE = Path(__file__).resolve().parent
PROJECT = CODE
DEFAULT_OUT = CODE / "runs" / "formal_10000"
# Portable release: prepared dataset may be supplied using --data.
REFERENCE_DATA = CODE / "data" / "data10000"
MODES = ("learnable", "zero", "zero_direct")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Tee:
    """Keep terminal progress visible while retaining the same text in a log."""
    def __init__(self, terminal, logfile):
        self.terminal, self.logfile = terminal, logfile

    def write(self, value):
        self.terminal.write(value)
        self.logfile.write(value)
        self.flush()
        return len(value)

    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

    def isatty(self):
        return self.terminal.isatty()


def update_status(root, **values):
    write_json(root / "status.json", {
        "updated_at": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        **values})


def copy_reference_data(destination):
    """复制固定的 10,000 材料划分，避免重新抽样改变原版基线的对照集。"""
    from tqdm import tqdm
    from dataset_utils import verify_prepared_data
    verify_prepared_data(REFERENCE_DATA)
    manifest = read_json(REFERENCE_DATA / "manifest.json")
    targets = read_json(REFERENCE_DATA / "targets.json")
    names = sum((manifest["splits"][key] for key in ("train", "val", "test")), [])
    if len(names) != 10000 or len(set(names)) != len(names) or set(names) != set(targets):
        raise ValueError("Reference dataset must contain the original 10,000 unique materials")
    if set(names) != set(manifest["structure_sha256"]):
        raise ValueError("Reference structure manifest does not match split IDs")
    if any(Path(name).name != name for name in names):
        raise ValueError("Unsafe structure filename in reference manifest")
    (destination / "structures").mkdir(parents=True)
    for name in tqdm(names, desc="Copy/verify fixed dataset", unit="material", mininterval=1):
        target = destination / "structures" / name
        # 只复制内容；在 /mnt/e 上逐文件同步 chmod/时间戳会显著拖慢准备。
        shutil.copyfile(REFERENCE_DATA / "structures" / name, target)
        if sha256(target) != manifest["structure_sha256"][name]:
            raise ValueError(f"Reference structure checksum mismatch: {name}")
    for name in ("manifest.json", "targets.json"):
        shutil.copyfile(REFERENCE_DATA / name, destination / name)
    print("Reused original 10,000 materials and exact split; structure checksums verified.", flush=True)


def execute(root, mode, configs, prepare_only=False):
    # 三组配置一起锁定，但默认只运行 learnable 和 zero；zero_direct 手动另启。
    # Re-running the same command resumes; changing the experiment is rejected.
    protocol = {"size": 10000, "seed": 123, "configs": configs,
                "sources": source_fingerprint(), "launcher_sha256": sha256(Path(__file__)),
                "torch": torch.__version__, "device": "cuda"}
    path = root / "protocol.json"
    if path.exists() and read_json(path) != protocol:
        raise ValueError("Protocol changed. Use the original files to resume, or choose a new --out directory.")
    if not path.exists():
        write_json(path, protocol)
        (root / "configs").mkdir(exist_ok=True)
        for name, config in configs.items():
            write_json(root / "configs" / f"formal_{name}.json", config)
    data = root / "data10000"
    if not data.exists():
        temporary = root / "data10000.preparing"
        if temporary.exists():
            raise RuntimeError(f"Interrupted data preparation: inspect and remove only {temporary}, then retry.")
        update_status(root, state="preparing")
        copy_reference_data(temporary)
        temporary.rename(data)
    manifest = read_json(data / "manifest.json")
    expected = {"train": 8000, "val": 1000, "test": 1000,
                "pretrain_train": 7200, "pretrain_val": 800}
    if {key: len(value) for key, value in manifest["splits"].items()} != expected or manifest["seed"] != 123:
        raise ValueError("Unexpected data split")
    dataset_hash = digest(manifest)
    hash_file = root / "dataset_hash.json"
    if hash_file.exists() and read_json(hash_file) != dataset_hash:
        raise ValueError("Data manifest changed")
    write_json(hash_file, dataset_hash)
    if prepare_only:
        if not (root / "status.json").exists() or read_json(root / "status.json").get("state") == "preparing":
            update_status(root, state="prepared")
        print("PREPARE OK. Exact dataset and protocol are ready; no training started.", flush=True)
        return
    modes = ("learnable", "zero") if mode == "both" else (mode,)
    for name in modes:
        for stage in ("pretrain", "downstream"):
            if (root / "STOP_REQUESTED").exists():
                update_status(root, state="paused", mode=name, stage=stage)
                print("PAUSED: remove root STOP_REQUESTED before continuing.")
                return
            out = root / name / stage
            if (out / "TRAINING_COMPLETE").exists():
                print(f"SKIP complete: {name}/{stage}", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            update_status(root, state="running", mode=name, stage=stage,
                          pid=os.getpid(), active_directory=str(out))
            with (out / "console.log").open("a", encoding="utf-8") as logfile:
                with redirect_stdout(Tee(sys.stdout, logfile)), redirect_stderr(Tee(sys.stderr, logfile)):
                    print(f"START {name}/{stage}", flush=True)
                    train(configs[name], data, out, stage, torch.device("cuda"),
                          resume=(out / "training_state.pt").exists(),
                          pretrained=root / name / "pretrain/best.pt" if stage == "downstream" else None)
            if not (out / "TRAINING_COMPLETE").exists():
                update_status(root, state="paused", mode=name, stage=stage)
                print("PAUSED safely. Re-run the same command to resume.", flush=True)
                return
    complete_modes = [name for name in MODES if all(
        (root / name / stage / "TRAINING_COMPLETE").exists()
        for stage in ("pretrain", "downstream"))]
    if "learnable" in complete_modes and "zero" in complete_modes:
        update_status(root, state="comparing")
        from ssl_alignn.report import compare
        compare(root, modes=complete_modes)
        (root / "COMPRESSED_COMPARISON_COMPLETE").write_text("Both Z=64 groups completed.\n")
        if len(complete_modes) == len(MODES):
            (root / "COMPARISON_COMPLETE").write_text("All six stages and audit completed.\n")
        update_status(root, state="complete" if len(complete_modes) == len(MODES)
                      else "compressed_complete", completed_modes=complete_modes)
    else:
        update_status(root, state="selected_mode_complete", mode=mode)


def main():
    global REFERENCE_DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("both", *MODES), default="both",
                        help="both runs only the two Z=64 groups; zero_direct is a separate later run")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--data", type=Path, default=REFERENCE_DATA,
                        help="Prepared fixed 10,000-material dataset; copied and verified on first start")
    parser.add_argument("--check", action="store_true", help="Show settings and check imports; do not train or prepare data")
    parser.add_argument("--prepare-only", action="store_true", help="Copy exact reference dataset and freeze protocol; do not train")
    parser.add_argument("--pause", action="store_true", help="Ask the active run to checkpoint and pause")
    args = parser.parse_args()
    REFERENCE_DATA = args.data.resolve()
    root = args.out.resolve()
    if root == REFERENCE_DATA or root.is_relative_to(REFERENCE_DATA):
        raise ValueError("The output directory must not be inside the source dataset")
    if args.pause:
        if not (root / "status.json").exists():
            raise RuntimeError("No experiment status found at --out")
        (root / "STOP_REQUESTED").touch()
        status = read_json(root / "status.json")
        if status.get("state") == "running":
            active = Path(status["active_directory"]).resolve()
            if not active.is_relative_to(root):
                raise ValueError("Unsafe active directory in status")
            (active / "STOP_REQUESTED").touch()
        print("Pause requested; wait for PAUSED before closing the training terminal.")
        return
    configs = {name: read_json(CODE / "configs" / f"formal_{name}.json") for name in MODES}
    expected = copy.deepcopy(configs["learnable"])
    expected["pretrain"]["mask_kind"] = "zero"
    if expected != configs["zero"]:
        raise ValueError("Configurations must differ only by mask_kind")
    expected = copy.deepcopy(configs["zero"])
    expected["model"]["z_dim"] = expected["model"]["dim"]
    if expected != configs["zero_direct"]:
        raise ValueError("zero_direct must differ from zero only by z_dim")
    for config in configs.values():
        validate_config(config)
    import lmdb, tqdm, matplotlib
    print(f"Training source: {CODE}\nResults: {root}\nMode: {args.mode}")
    print(json.dumps(configs, ensure_ascii=False, indent=2))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; activate a compatible Linux/WSL GPU environment (see docs/ENVIRONMENT.md)")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    if args.check:
        print("CHECK OK. No training started and no dataset created.")
        return
    root.mkdir(parents=True, exist_ok=True)
    # The outer lock prevents two launchers from writing one comparison folder.
    with FileLock(str(root / ".formal.lock"), timeout=0):
        try:
            execute(root, args.mode, configs, prepare_only=args.prepare_only)
        except Exception as error:
            update_status(root, state="failed", error=str(error))
            raise


if __name__ == "__main__":
    main()
