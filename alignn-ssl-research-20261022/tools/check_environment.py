"""Inspect the environment and bundled upstream imports; no GPU allocation."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ssl_alignn
import torch
import dgl
import alignn
from ssl_alignn.model import MaskedModel
from ssl_alignn.data import build_graph

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cuda', action='store_true')
    args = parser.parse_args()
    cuda = torch.cuda.is_available()
    names = ['torch','dgl','jarvis-tools','numpy','pydantic','pydantic-settings','lmdb','filelock','tqdm']
    print(json.dumps({'python':platform.python_version(),'torch_runtime':torch.__version__,
        'torch_cuda':torch.version.cuda,'cuda_available':cuda,
        'alignn_path':str(Path(alignn.__file__).resolve()),
        'dependencies':{n:metadata.version(n) for n in names}},indent=2))
    if not Path(alignn.__file__).resolve().is_relative_to(ssl_alignn.UPSTREAM):
        raise SystemExit('Wrong upstream ALIGNN imported')
    if args.require_cuda and not cuda:
        raise SystemExit('CUDA is required for formal training')
