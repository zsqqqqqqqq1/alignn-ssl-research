"""Audited split reuse, graph cache and label-free pretraining batches."""
import csv
import io
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal

import dgl
import torch
from jarvis.core.atoms import Atoms
from jarvis.core.specie import Specie
from alignn.graphs import Graph
from .labels import element_targets
from tqdm import tqdm


# Each worker opens its own LMDB handle; never pickle a live database handle.
_LMDB = {}


def _lmdb(path):
    import lmdb
    key = (os.getpid(), str(path))
    if key not in _LMDB:
        path.mkdir(parents=True, exist_ok=True)
        _LMDB[key] = lmdb.open(str(path), map_size=64 * 1024**3,
                              max_readers=256, readahead=False)
    return _LMDB[key]


class GraphItems(torch.utils.data.Dataset):
    def __init__(self, data, names):
        self.data, self.names = data, names

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        return name, self.data.get(name)


def keep_items(items):
    return items


def initialize_worker(worker_id):
    # The parent handles Ctrl+C and commits a whole optimizer step. Workers
    # must not abort graph loading halfway through that graceful pause.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def prepare(source, original_split, out, size=400, seed=123):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite dataset: {out}")
    with open(source / "id_prop.csv", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    labels = {}
    for name, target in rows:
        if Path(name).name != name or name in labels:
            raise ValueError(f"Duplicate or unsafe ID: {name}")
        value = float(target)
        if not math.isfinite(value):
            raise ValueError(f"Nonfinite target: {name}")
        labels[name] = value
    original = json.loads(Path(original_split).read_text(encoding="utf-8"))
    pools = [original[k] for k in ("id_train", "id_val", "id_test")]
    flat = sum(pools, [])
    if len(set(flat)) != len(flat) or set(flat) != set(labels):
        raise ValueError("Baseline split is overlapping, duplicated or not equal to dataset IDs")
    if size != 0 and size < 20:
        raise ValueError("Use >=20 materials, or 0 for the complete baseline split")
    counts = [int(size * .8), int(size * .1)]
    counts.append(size - sum(counts))
    rng = random.Random(seed)
    selected = {}
    for key, pool, n in zip(("train", "val", "test"), pools, counts):
        selected[key] = rng.sample(pool, n) if size else list(pool)
    train = selected["train"][:]
    rng.shuffle(train)
    monitor_n = max(1, round(len(train) * .1))
    selected["pretrain_val"] = train[:monitor_n]
    selected["pretrain_train"] = train[monitor_n:]
    # Finish validation before creating a partial dataset directory.
    names = selected["train"] + selected["val"] + selected["test"]
    for name in tqdm(names, desc="Check structures", unit="material", dynamic_ncols=True):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)
    out.mkdir(parents=True)
    structures = out / "structures"
    structures.mkdir()
    hashes = {}
    for name in tqdm(names, desc="Copy structures", unit="material", dynamic_ncols=True):
        shutil.copy2(source / name, structures / name)
        hashes[name] = sha256(structures / name)
    manifest = {"schema": 1, "seed": seed, "splits": selected,
                "structure_sha256": hashes,
                "baseline_split_sha256": sha256(original_split),
                "source_id_prop_sha256": sha256(source / "id_prop.csv"),
                "pretraining_uses_formation_energy": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    # Supervised labels have their own file and are never opened by pretraining.
    (out / "targets.json").write_text(json.dumps({n: labels[n] for n in names}), encoding="utf-8")
    return {k: len(v) for k, v in selected.items()}


def build_graph(path, cfg):
    atoms = Atoms.from_poscar(str(path))
    g, lg = Graph.atom_dgl_multigraph(
        atoms, neighbor_strategy="k-nearest", atom_features="cgcnn",
        cutoff=cfg["cutoff"], max_neighbors=cfg["max_neighbors"],
        cutoff_extra=cfg["cutoff_extra"], use_canonize=True,
        compute_line_graph=True, dtype="float32")
    u, v = g.edges()
    r = g.edata["r"]
    # The unchanged upstream builder emits consecutive forward/reverse pairs.
    # Validate that contract; do not guess reverse partners from node IDs (PBC).
    if g.num_edges() % 2 or not (torch.equal(u[::2], v[1::2]) and
            torch.equal(v[::2], u[1::2]) and torch.allclose(r[::2], -r[1::2])):
        raise ValueError("Upstream reverse-edge ordering changed")
    if lg.num_nodes() != g.num_edges() or g.num_edges() == 0 or lg.num_edges() == 0:
        raise ValueError("Empty or inconsistent graph; k-nearest periodic graph required")
    # PBC self-edges can occur more than twice in upstream graphs. All duplicate
    # copies of a physical bond must be masked together, not just adjacent pairs.
    pair_lookup, pair_ids = {}, []
    for src, dst, vector in zip(u.tolist(), v.tolist(), r.tolist()):
        displacement = tuple(round(float(c), 5) for c in vector)
        key = min((src, dst, displacement), (dst, src, tuple(-c for c in displacement)))
        if key not in pair_lookup:
            pair_lookup[key] = len(pair_lookup)
        pair_ids.append(pair_lookup[key])
    x = g.ndata["atom_features"].float().clone()
    if x.shape != (atoms.num_atoms, 92) or not torch.isfinite(x).all():
        raise ValueError("Expected finite CGCNN 92-dimensional features")
    periods, groups = [], []
    for element in atoms.elements:
        row, col = element_targets(element)
        periods.append(row)
        groups.append(col)
    result = {"g": g, "lg": lg, "x": x,
              "distance": r.norm(dim=1).clone(),
              "angles": lg.edata["h"].float().clamp(-1, 1).clone(),
              "period": torch.tensor(periods), "group": torch.tensor(groups),
              "elements": list(atoms.elements), "pair_ids": torch.tensor(pair_ids)}
    # No coordinates, raw vectors, original features, or labels on either graph.
    for graph in (g, lg):
        for key in list(graph.ndata):
            graph.ndata.pop(key)
        for key in list(graph.edata):
            graph.edata.pop(key)
    return result


class CrystalData:
    def __init__(self, root, graph_cfg, source_fingerprint, supervised=False,
                 use_lmdb=False, num_workers=0, pin_memory=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        splits = self.manifest["splits"]
        downstream = sum([splits[k] for k in ("train", "val", "test")], [])
        pretraining = splits["pretrain_train"] + splits["pretrain_val"]
        if (any(not values for values in splits.values()) or
                len(downstream) != len(set(downstream)) or
                len(pretraining) != len(set(pretraining)) or
                set(pretraining) != set(splits["train"]) or
                set(downstream) != set(self.manifest["structure_sha256"])):
            raise ValueError("Invalid/overlapping split or pretraining outside training pool")
        self.cfg = graph_cfg
        self.cache = self.root / "graph_cache" / digest([graph_cfg, source_fingerprint, dgl.__version__])[:16]
        self.cache.mkdir(parents=True, exist_ok=True)
        self.use_lmdb = use_lmdb
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.targets = json.loads((self.root / "targets.json").read_text()) if supervised else None
        if self.targets is not None and (set(self.targets) != set(downstream) or
                not all(math.isfinite(v) for v in self.targets.values())):
            raise ValueError("Invalid supervised targets")

    def get(self, name):
        path = self.root / "structures" / name
        expected = self.manifest["structure_sha256"][name]
        if sha256(path) != expected:
            raise ValueError(f"Structure changed since preparation: {name}")
        if self.use_lmdb:
            env = _lmdb(self.cache / "graphs.lmdb")
            key = expected.encode("ascii")
            with env.begin() as transaction:
                payload = transaction.get(key)
            if payload is not None:
                return torch.load(io.BytesIO(payload), map_location="cpu", weights_only=False)
            value = build_graph(path, self.cfg)
            buffer = io.BytesIO()
            torch.save(value, buffer)
            # LMDB serializes concurrent writers and commits each complete graph.
            with env.begin(write=True) as transaction:
                transaction.put(key, buffer.getvalue(), overwrite=False)
            return value
        cache = self.cache / (expected + ".pt")
        if cache.exists():
            value = torch.load(cache, map_location="cpu", weights_only=False)
        else:
            from .state import atomic_save
            value = build_graph(path, self.cfg)
            atomic_save(value, cache)
        return value

    def batch(self, names, device):
        values = [self.get(n) for n in names]
        return self.collate(list(zip(names, values)), device)

    def iter_items(self, names, batch_size):
        # A dedicated RNG prevents worker creation from changing model RNG state,
        # which is essential for exact batch-boundary checkpoint recovery.
        loader = torch.utils.data.DataLoader(
            GraphItems(self, names), batch_size=batch_size, shuffle=False,
            num_workers=self.num_workers, collate_fn=keep_items,
            worker_init_fn=initialize_worker,
            pin_memory=self.pin_memory, generator=torch.Generator().manual_seed(0),
            **({"multiprocessing_context": "spawn"} if self.num_workers else {}))
        yield from loader

    def collate(self, items, device):
        names, values = zip(*items)
        names = list(names)
        batch = {key: torch.cat([v[key] for v in values]).to(device)
                 for key in ("x", "distance", "angles", "period", "group")}
        batch.update(g=dgl.batch([v["g"] for v in values]).to(device),
                     lg=dgl.batch([v["lg"] for v in values]).to(device),
                     node_counts=[v["g"].num_nodes() for v in values],
                     edge_counts=[v["g"].num_edges() for v in values], names=names,
                     pair_ids=[v["pair_ids"] for v in values])
        if self.targets is not None:
            batch["target"] = torch.tensor([self.targets[n] for n in names], device=device)
        return batch
