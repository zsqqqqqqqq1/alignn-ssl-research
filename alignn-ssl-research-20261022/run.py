"""Run from Windows or WSL; all paths are explicit and relocatable."""
import argparse
import json
from pathlib import Path
import torch
from ssl_alignn.data import prepare
from ssl_alignn.runner import train, export_z


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--source", required=True, help="Directory with id_prop.csv and POSCAR files")
    prep.add_argument("--split", required=True, help="Baseline ids_train_val_test.json")
    prep.add_argument("--out", required=True)
    prep.add_argument("--size", type=int, default=400, help="0 reuses all baseline IDs")
    prep.add_argument("--seed", type=int, default=123)
    for command in ("pretrain", "downstream", "export"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        p.add_argument("--data", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
        if command != "pretrain":
            p.add_argument("--pretrained", required=True)
        if command != "export":
            p.add_argument("--resume", action="store_true")
            p.add_argument("--stop-after-steps", type=int, help="Controlled interruption for resume verification")
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.source, args.split, args.out, args.size, args.seed))
        return
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if args.command == "export":
        export_z(cfg, args.data, args.pretrained, args.out, device)
    else:
        train(cfg, args.data, args.out, args.command, device, args.resume,
              getattr(args, "pretrained", None), args.stop_after_steps)


if __name__ == "__main__":
    main()
