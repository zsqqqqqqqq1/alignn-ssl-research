"""检查显示适配前后数值一致，以及旧断点仍能恢复。"""
import contextlib
import io
import json
import sys
import unittest

import numpy as np
import torch

import run_formal
from ssl_alignn import runner
from terminal_display import TerminalDisplay
from test_protocol import ProtocolTests as _Fixture


class DisplayTests(unittest.TestCase):
    setUp = _Fixture.setUp
    tearDown = _Fixture.tearDown

    def assert_tree_equal(self, a, b):
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, np.ndarray):
            np.testing.assert_array_equal(a, b)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_tree_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for first, second in zip(a, b):
                self.assert_tree_equal(first, second)
        else:
            self.assertEqual(a, b)

    def assert_checkpoint_equal(self, first, second):
        a = torch.load(first / "training_state.pt", weights_only=False)
        b = torch.load(second / "training_state.pt", weights_only=False)
        # 墙钟耗时受打印和系统调度影响，不属于数值一致性的比较对象。
        a["progress"].pop("seconds")
        b["progress"].pop("seconds")
        self.assert_tree_equal(a, b)

    def test_old_checkpoint_resume_and_downstream_unchanged(self):
        direct, resumed = self.root / "direct", self.root / "resumed"
        stream = io.StringIO()
        original_bar = runner.progress_bar
        with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
            runner.train(self.cfg, self.data_dir, direct, "pretrain", torch.device("cpu"))
            runner.train(self.cfg, self.data_dir, resumed, "pretrain", torch.device("cpu"), stop_after_steps=2)
            with TerminalDisplay().installed(run_formal, runner):
                runner.train(self.cfg, self.data_dir, resumed, "pretrain", torch.device("cpu"), resume=True)
            self.assert_checkpoint_equal(direct, resumed)
            # 两次下游使用同一个编码器文件，以便签名中的文件哈希也相同。
            downstream_a, downstream_b = self.root / "down_a", self.root / "down_b"
            runner.train(self.cfg, self.data_dir, downstream_a, "downstream", torch.device("cpu"),
                         pretrained=direct / "best.pt")
            with TerminalDisplay().installed(run_formal, runner):
                runner.train(self.cfg, self.data_dir, downstream_b, "downstream", torch.device("cpu"),
                             pretrained=direct / "best.pt")
            self.assert_checkpoint_equal(downstream_a, downstream_b)
            self.assertEqual((downstream_a / "test_metrics.json").read_bytes(),
                             (downstream_b / "test_metrics.json").read_bytes())
        self.assertIs(runner.progress_bar, original_bar)
        self.assertIn("RowCE", stream.getvalue())
        self.assertIn("MAE(eV/atom)", stream.getvalue())
        self.assertIn("RESUME:", stream.getvalue())

    def test_bar_refresh_excluded_from_log_and_unknown_messages_preserved(self):
        class Terminal(io.StringIO):
            def isatty(self):
                return True
        terminal, logfile = Terminal(), io.StringIO()
        tee = run_formal.Tee(terminal, logfile)
        display = TerminalDisplay()
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            with display.progress_bar(2, "downstream Train 92/300") as bar:
                bar.set_postfix(loss="0.12345", lr="1e-3", step=11376, saved=11376, refresh=True)
                bar.update(2)
            display.print(json.dumps(dict(epoch=92, step=11377, lr=.001,
                          train=dict(loss=.1, mae_eV_per_atom=.1),
                          val=dict(loss=.2, mae_eV_per_atom=.2))))
            display.print("PAUSED at step 11377")
            display.print("unexpected diagnostic", file=sys.stderr)
        self.assertIn("loss=0.12345", terminal.getvalue())
        self.assertNotIn("saved=", terminal.getvalue())
        self.assertNotIn("\r", logfile.getvalue())
        self.assertIn("MAE(eV/atom)", logfile.getvalue())
        self.assertIn("PAUSED at step 11377", logfile.getvalue())
        self.assertIn("unexpected diagnostic", logfile.getvalue())


# 不把引用的夹具测试类作为本模块的另一套测试重复收集。
del _Fixture
