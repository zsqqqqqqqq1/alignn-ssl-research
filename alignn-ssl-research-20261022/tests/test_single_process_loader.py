"""验证关闭并行读取后，从原协议断点继续的数值一致性。"""
import copy
import json
import unittest

import torch
from ssl_alignn.data import CrystalData
from ssl_alignn.runner import train
from single_process_loader import single_process_loading
from test_protocol import ProtocolTests as _Fixture


class SingleProcessTests(unittest.TestCase):
    setUp = _Fixture.setUp
    tearDown = _Fixture.tearDown

    def assert_tensor_tree_equal(self, a, b):
        if torch.is_tensor(a):
            self.assertTrue(torch.equal(a, b))
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_tensor_tree_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                self.assert_tensor_tree_equal(x, y)
        else:
            self.assertEqual(a, b)

    def test_original_parallel_checkpoint_resumes_identically_with_single_process(self):
        cfg = copy.deepcopy(self.cfg)
        cfg.update(use_lmdb=True, num_workers=2)
        direct, recovered = self.root/'direct', self.root/'recovered'
        train(cfg, self.data_dir, direct, 'pretrain', torch.device('cpu'))
        train(cfg, self.data_dir, recovered, 'pretrain', torch.device('cpu'), stop_after_steps=2)
        original = CrystalData.__init__
        with single_process_loading():
            train(cfg, self.data_dir, recovered, 'pretrain', torch.device('cpu'), resume=True)
        self.assertIs(CrystalData.__init__, original)
        a = torch.load(direct/'training_state.pt', weights_only=False)
        b = torch.load(recovered/'training_state.pt', weights_only=False)
        for state in (a, b):
            state['progress'].pop('seconds')
        for key in ('model', 'optimizer', 'scheduler', 'progress', 'signature'):
            self.assert_tensor_tree_equal(a[key], b[key])
        self.assertTrue(torch.equal(a['rng']['torch'], b['rng']['torch']))
        rows = [json.loads(s) for s in (self.root/'loader_runtime.jsonl').read_text().splitlines()]
        self.assertEqual(rows[-1]['configured_num_workers'], 2)
        self.assertEqual(rows[-1]['effective_num_workers'], 0)


del _Fixture

if __name__ == '__main__':
    unittest.main()
