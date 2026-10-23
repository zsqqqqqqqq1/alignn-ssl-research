"""默认两组完成后必须停止；直传对照只有显式选中时才会启动。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_formal


class FormalModeTests(unittest.TestCase):
    def test_direct_control_is_deferred_and_has_own_checkpoint_directory(self):
        configs = {name: run_formal.read_json(run_formal.CODE / 'configs' / f'formal_{name}.json')
                   for name in run_formal.MODES}
        expected = copy.deepcopy(configs['zero'])
        expected['model']['z_dim'] = 256
        self.assertEqual(expected, configs['zero_direct'])
        self.assertEqual(configs['learnable']['model']['z_dim'], 64)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / 'data10000'
            data.mkdir()
            splits = {k: list(range(n)) for k, n in dict(train=8000, val=1000, test=1000,
                                                       pretrain_train=7200, pretrain_val=800).items()}
            (data / 'manifest.json').write_text(json.dumps(dict(seed=123, splits=splits)))
            calls = []

            def fake_train(config, dataset, output, stage, device, **kwargs):
                calls.append((output.parent.name, stage, config['model']['z_dim']))
                (output / 'TRAINING_COMPLETE').touch()

            with patch.object(run_formal, 'train', side_effect=fake_train), \
                    patch('ssl_alignn.report.compare') as compare:
                run_formal.execute(root, 'both', configs)
                self.assertEqual(calls, [('learnable', 'pretrain', 64), ('learnable', 'downstream', 64),
                                         ('zero', 'pretrain', 64), ('zero', 'downstream', 64)])
                self.assertFalse((root / 'zero_direct').exists())
                self.assertFalse((root / 'COMPARISON_COMPLETE').exists())
                compare.assert_called_with(root, modes=['learnable', 'zero'])
                run_formal.execute(root, 'zero_direct', configs)
                self.assertEqual(calls[-2:], [('zero_direct', 'pretrain', 256), ('zero_direct', 'downstream', 256)])
                compare.assert_called_with(root, modes=['learnable', 'zero', 'zero_direct'])
                self.assertTrue((root / 'COMPARISON_COMPLETE').exists())


if __name__ == '__main__':
    unittest.main()
