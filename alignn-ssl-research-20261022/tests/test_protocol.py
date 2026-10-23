import copy
import contextlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import torch
from jarvis.core.atoms import Atoms
from ssl_alignn.data import build_graph, CrystalData, sha256
from ssl_alignn.model import MaskedModel, FrozenDownstream, sample_masks, reconstruction_loss
from ssl_alignn.runner import train, optimizer_parameters, validate_config, load_pretrained
from ssl_alignn.report import compare
from ssl_alignn.state import source_fingerprint, tensor_hash
from ssl_alignn.labels import element_targets, FALLBACK


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = json.loads((Path(__file__).parent / "fixtures/zero_test_config.json").read_text())
        self.cfg.update(cpu_threads=1, checkpoint_every_steps=1, progress=False)
        self.cfg["model"] = {"dim": 8, "alignn_layers": 1, "gcn_layers": 1}
        for stage in ("pretrain", "downstream"):
            self.cfg[stage].update(epochs=2, batch_size=1)
        self.data_dir = self.root / "data"
        (self.data_dir / "structures").mkdir(parents=True)
        hashes = {}
        for i in range(8):
            atoms = Atoms(lattice_mat=[[3 + i*.1, 0, 0], [0, 3, 0], [0, 0, 3]],
                          coords=[[0, 0, 0]], elements=["Si" if i % 2 else "Al"], cartesian=False)
            path = self.data_dir / "structures" / f"sample{i}.vasp"
            atoms.write_poscar(str(path))
            hashes[path.name] = sha256(path)
        names = list(hashes)
        manifest = {"structure_sha256": hashes, "splits": {
            "train": names[:4], "val": names[4:6], "test": names[6:],
            "pretrain_train": names[:3], "pretrain_val": names[3:4]}}
        (self.data_dir / "manifest.json").write_text(json.dumps(manifest))
        (self.data_dir / "targets.json").write_text(json.dumps({n: -.1 * i for i, n in enumerate(names)}))
        self.data = CrystalData(self.data_dir, self.cfg["graph"], source_fingerprint())
        self.batch = self.data.batch(names[:2], torch.device("cpu"))

    def tearDown(self):
        self.temp.cleanup()

    def test_masks_geometry_and_no_feature_bypass(self):
        b = self.batch
        self.assertEqual(b["lg"].num_nodes(), b["g"].num_edges())
        self.assertEqual(len(b["g"].ndata), 0)
        self.assertEqual(len(b["lg"].ndata), 0)
        model = MaskedModel(self.cfg["model"], "zero").eval()
        nm, em = sample_masks(b, .15, .15, 123)
        self.assertTrue(torch.equal(em[::2], em[1::2]))
        offset = 0
        for ids in b["pair_ids"]:
            local = em[offset:offset + len(ids)]
            for pid in ids.unique():
                self.assertEqual(local[ids == pid].unique().numel(), 1)
            offset += len(ids)
        x0 = b["x"].clone()
        x, rbf = model.inputs(b, nm, em)
        self.assertTrue(torch.equal(x[nm], torch.zeros_like(x[nm])))
        self.assertTrue(torch.equal(rbf[em], torch.zeros_like(rbf[em])))
        self.assertTrue(torch.equal(b["x"], x0))
        altered = dict(b, x=b["x"].clone(), distance=b["distance"].clone())
        altered["x"][nm] = 12345
        altered["distance"][em] = 12345
        before = model(b, nm, em)
        after = model(altered, nm, em)
        for a, c in zip(before, after):
            self.assertTrue(torch.equal(a, c), "Masked input values leaked into logits")
        self.assertEqual(model.encode(b).shape, (2, 8))

    def test_missing_element_labels(self):
        self.assertEqual(len(FALLBACK), 118)
        self.assertEqual(element_targets("Pm"), (7, 6))
        self.assertEqual(element_targets("Eu"), (7, 8))
        self.assertEqual(element_targets("Lu"), (5, 2))

    def test_hidden256_projection_gradient_and_frozen_transfer(self):
        # 使用正式层数检查 256 -> 64 压缩与 256 直传，覆盖真实张量接口。
        for kind, z_dim in (("learnable", 64), ("zero", 64), ("zero", 256)):
            with self.subTest(kind=kind, z_dim=z_dim):
                cfg = dict(dim=256, z_dim=z_dim, embedding_features=64,
                           alignn_layers=4, gcn_layers=4)
                whole = copy.deepcopy(self.cfg)
                whole["model"] = cfg
                validate_config(whole)
                model = MaskedModel(cfg, kind)
                masks = sample_masks(self.batch, .5, .5, 2)
                loss, *_ = reconstruction_loss(model, self.batch, masks, [1, 1, 1])
                loss.backward()
                self.assertEqual(model.encode(self.batch).shape, (2, z_dim))
                self.assertEqual(model.period_head.in_features, z_dim)
                self.assertEqual(model.distance_head.in_features, 256)
                if z_dim == 64:
                    grad = model.atom_projection.weight.grad
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(float(grad.abs().sum()), 0)
                else:
                    self.assertIsInstance(model.atom_projection, torch.nn.Identity)
                model.zero_grad(set_to_none=True)
                down = FrozenDownstream(model, cfg).train()
                before = tensor_hash(model)
                optimizer = torch.optim.AdamW(p for p in down.parameters() if p.requires_grad)
                prediction = down(self.batch)
                self.assertEqual(prediction.shape, (2,))
                prediction.square().mean().backward()
                if z_dim == 64:
                    self.assertTrue(any(p.grad is not None for p in down.network.atom_embedding.parameters()))
                else:
                    self.assertIsInstance(down.network.atom_embedding, torch.nn.Identity)
                optimizer.step()
                self.assertEqual(tensor_hash(model), before)
                self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_compressed256_checkpoint_resume(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["model"].update(dim=256, z_dim=64, embedding_features=64)
        direct, resumed = self.root / "wide_direct", self.root / "wide_resumed"
        train(cfg, self.data_dir, direct, "pretrain", torch.device("cpu"))
        train(cfg, self.data_dir, resumed, "pretrain", torch.device("cpu"), stop_after_steps=2)
        train(cfg, self.data_dir, resumed, "pretrain", torch.device("cpu"), resume=True)
        a = torch.load(direct / "training_state.pt", weights_only=False)
        b = torch.load(resumed / "training_state.pt", weights_only=False)
        self.assertEqual(a["progress"]["history"], b["progress"]["history"])
        for key in a["model"]:
            self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
        restored, _ = load_pretrained(resumed / "best.pt", torch.device("cpu"))
        self.assertEqual(restored.eval().encode(self.batch).shape, (2, 64))

    def test_learnable_mask_gradient_and_frozen_encoder(self):
        model = MaskedModel(self.cfg["model"], "learnable")
        masks = sample_masks(self.batch, .5, .5, 2)
        loss, *_ = reconstruction_loss(model, self.batch, masks, [1, 1, 1])
        loss.backward()
        for p in (model.atom_mask, model.edge_mask, model.period_head.weight,
                  model.encoder.alignn_layers[0].node_update.src_update.weight):
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
            self.assertGreater(float(p.grad.abs().sum()), 0)
        model.zero_grad(set_to_none=True)
        downstream = FrozenDownstream(model, self.cfg["model"])
        before = tensor_hash(model)
        downstream.train()
        self.assertFalse(model.training)
        optimizer = torch.optim.AdamW([p for p in downstream.parameters() if p.requires_grad])
        downstream(self.batch).square().mean().backward()
        optimizer.step()
        self.assertEqual(tensor_hash(model), before)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        # Original descriptors are needed by frozen encoding, but NOT the downstream backbone.
        self.assertIsInstance(downstream.network.atom_embedding, torch.nn.Identity)

    def test_resume_matches_uninterrupted_and_rejects_changes(self):
        direct, resumed = self.root / "direct", self.root / "resumed"
        train(self.cfg, self.data_dir, direct, "pretrain", torch.device("cpu"))
        train(self.cfg, self.data_dir, resumed, "pretrain", torch.device("cpu"), stop_after_steps=2)
        train(self.cfg, self.data_dir, resumed, "pretrain", torch.device("cpu"), resume=True)
        a = torch.load(direct / "training_state.pt", weights_only=False)
        b = torch.load(resumed / "training_state.pt", weights_only=False)
        self.assertEqual(a["progress"]["history"], b["progress"]["history"])
        self.assertEqual(a["scheduler"], b["scheduler"])
        for key in a["model"]:
            self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
        cfg = copy.deepcopy(self.cfg)
        cfg["pretrain"]["node_ratio"] = .2
        with self.assertRaisesRegex(ValueError, "Resume refused"):
            train(cfg, self.data_dir, resumed, "pretrain", torch.device("cpu"), resume=True)
        # Formation-energy file is irrelevant to pretraining and can be absent.
        (self.data_dir / "targets.json").unlink()
        train(self.cfg, self.data_dir, resumed, "pretrain", torch.device("cpu"), resume=True)

    def test_accumulation_resume_last_short_batch_and_decay(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(use_lmdb=True, group_decay=True, gradient_clip=None)
        cfg["pretrain"].update(batch_size=2, microbatch_size=1, mask_kind="learnable")
        direct, resumed = self.root / "acc_direct", self.root / "acc_resumed"
        train(cfg, self.data_dir, direct, "pretrain", torch.device("cpu"))
        train(cfg, self.data_dir, resumed, "pretrain", torch.device("cpu"), stop_after_steps=1)
        paused = torch.load(resumed / "training_state.pt", weights_only=False)
        self.assertEqual(paused["progress"]["step"], 1)
        self.assertEqual(paused["progress"]["next_batch"], 1)
        train(cfg, self.data_dir, resumed, "pretrain", torch.device("cpu"), resume=True)
        a = torch.load(direct / "training_state.pt", weights_only=False)
        b = torch.load(resumed / "training_state.pt", weights_only=False)
        self.assertEqual(a["progress"]["history"], b["progress"]["history"])
        self.assertEqual(a["scheduler"], b["scheduler"])
        self.assertEqual(a["progress"]["step"], 4)
        for key in a["model"]:
            self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
        model = MaskedModel(cfg["model"], "learnable")
        groups = optimizer_parameters(model, True)
        no_decay_ids = {id(value) for value in groups[1]["params"]}
        for name, parameter in model.named_parameters():
            self.assertEqual(id(parameter) in no_decay_ids, any(v in name for v in ("bias", "bn", "norm")))

    def test_parallel_lmdb_loader_order_and_rng(self):
        data = CrystalData(self.data_dir, self.cfg["graph"], source_fingerprint(),
                           use_lmdb=True, num_workers=2)
        names = list(data.manifest["structure_sha256"])
        before = torch.get_rng_state().clone()
        loaded = [item for items in data.iter_items(names, 3) for item in items]
        self.assertEqual([name for name, _ in loaded], names)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        for name, value in loaded:
            cached = data.get(name)
            self.assertTrue(torch.equal(value["x"], cached["x"]))
            self.assertEqual(value["g"].num_edges(), cached["g"].num_edges())

    def test_progress_downstream_and_comparison(self):
        root = self.root / "comparison"
        shutil.copytree(self.data_dir, root / "data10000")
        configs = {}
        for mode in ("learnable", "zero", "zero_direct"):
            cfg = copy.deepcopy(self.cfg)
            cfg["model"]["z_dim"] = 8 if mode == "zero_direct" else 4
            cfg.update(progress=True, group_decay=True, use_lmdb=True, gradient_clip=None)
            cfg["pretrain"].update(mask_kind="learnable" if mode == "learnable" else "zero", batch_size=2, microbatch_size=1)
            cfg["downstream"].update(batch_size=2, microbatch_size=1)
            configs[mode] = cfg
            log = io.StringIO()
            with contextlib.redirect_stderr(log):
                train(cfg, root / "data10000", root / mode / "pretrain", "pretrain", torch.device("cpu"))
                train(cfg, root / "data10000", root / mode / "downstream", "downstream", torch.device("cpu"),
                      pretrained=root / mode / "pretrain/best.pt")
            for message in ("Train 1/2", "Val 2/2", "saved=", "Test", "100%"):
                self.assertIn(message, log.getvalue())
        (root / "protocol.json").write_text(json.dumps({"configs": configs, "sources": source_fingerprint()}))
        compare(root, modes=["learnable", "zero"])
        partial = json.loads((root / "comparison.json").read_text())
        self.assertEqual(set(partial["modes"]), {"learnable", "zero"})
        compare(root)
        report = json.loads((root / "comparison.json").read_text())
        self.assertTrue(report["modes"]["learnable"]["frozen_weights_identical"])
        self.assertEqual(report["modes"]["zero"]["test"]["n"], 2)
        self.assertEqual(report["modes"]["zero_direct"]["test"]["n"], 2)
        self.assertIn("test_mae_zero_compressed_minus_direct", report)
        self.assertTrue((root / "training_curves.png").exists())


if __name__ == "__main__":
    unittest.main()
