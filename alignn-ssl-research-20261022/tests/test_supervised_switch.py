import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from experiment_options import training_config, supervised_config, select_experiment, legacy_config_reads
from audit_alignn_call import synthetic_data,audit
from supervised_model import OriginalSupervisedModel
from supervised_runner import train
from ssl_alignn.state import tensor_hash


class SupervisedSwitchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.data=self.root/'data';synthetic_data(self.data)
        self.cfg=supervised_config(json.loads((Path(__file__).resolve().parents[1]/'configs/formal_learnable.json').read_text()))
        self.cfg.update(cpu_threads=1,use_lmdb=False,progress=False)
        self.cfg['model'].update(dim=8,embedding_features=8,alignn_layers=1,gcn_layers=1)
        self.cfg['downstream'].update(epochs=2,batch_size=2,microbatch_size=1)

    def test_boolean_switch_and_false_dispatch(self):
        for value in (True,False):
            self.assertIs(training_config({'self_supervised':value})[0],value)
        for value in ('false', 0, None):
            with self.assertRaises(ValueError):training_config({'self_supervised':value})
        self.assertTrue(training_config({})[0])
        import run_formal_display
        with patch('experiment_options.select_experiment',return_value=(False,self.cfg)),patch('supervised_run.main') as main,patch('sys.argv',['run_formal_display.py']):
            run_formal_display.main();main.assert_called_once_with(self.cfg)

    def test_selected_config_parameters_and_mixed_modes(self):
        cfg=copy.deepcopy(self.cfg);cfg['self_supervised']=False
        cfg['downstream']['lr']=.0003
        for mode in ('learnable','zero','zero_direct'):
            (self.root/f'formal_{mode}.json').write_text(json.dumps(cfg))
        enabled,selected=select_experiment(['--mode','learnable'],self.root)
        self.assertFalse(enabled)
        self.assertEqual(selected['downstream']['lr'],.0003)
        self.assertEqual(selected['model'],cfg['model'])
        self.assertNotIn('self_supervised',selected)
        self.assertEqual(select_experiment([],self.root),(False,selected))
        cfg['self_supervised']=True
        (self.root/'formal_zero.json').write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'matching self_supervised'):
            select_experiment([],self.root)
        self.assertEqual(select_experiment(['--mode','zero'],self.root),(True,None))
        cfg['self_supervised']=False;cfg['downstream']['lr']=.02
        (self.root/'formal_zero.json').write_text(json.dumps(cfg))
        with self.assertRaisesRegex(ValueError,'settings differ'):
            select_experiment([],self.root)

    def test_legacy_protocol_config_is_preserved(self):
        import run_formal
        configs=self.root/'configs';configs.mkdir()
        previous=copy.deepcopy(self.cfg)
        path=configs/'formal_learnable.json'
        path.write_text(json.dumps(dict(previous,self_supervised=True)))
        before=path.read_bytes();reader=run_formal.read_json
        # Non-config JSON, including checkpoints/protocols, must remain unmodified.
        other=self.root/'protocol.json';other.write_bytes(before)
        with patch.object(run_formal,'CODE',self.root):
            with legacy_config_reads(run_formal):
                self.assertEqual(run_formal.read_json(path),previous)
                self.assertIn('self_supervised',run_formal.read_json(other))
                # A real hyperparameter change is still visible to resume validation.
                changed=dict(previous,seed=999,self_supervised=True)
                path.write_text(json.dumps(changed))
                self.assertEqual(run_formal.read_json(path)['seed'],999)
        self.assertIs(run_formal.read_json,reader)

    def test_original_model_and_backbone_outputs_gradients_match(self):
        self.assertTrue(audit(self.data,self.cfg['model'])['passed'])
        model=OriginalSupervisedModel(self.cfg['model'])
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertFalse(hasattr(model,'pretrained'))
        self.assertFalse(any('mask' in n or 'projection' in n for n,_ in model.named_parameters()))

    def test_resume_matches_uninterrupted_and_final_test_exists(self):
        continuous=self.root/'continuous';resumed=self.root/'resumed';device=torch.device('cpu')
        train(self.cfg,self.data,continuous,'downstream',device)
        train(self.cfg,self.data,resumed,'downstream',device,stop_after_steps=1)
        train(self.cfg,self.data,resumed,'downstream',device,resume=True)
        a=torch.load(continuous/'training_state.pt',map_location='cpu',weights_only=False)
        b=torch.load(resumed/'training_state.pt',map_location='cpu',weights_only=False)
        self.assertEqual(a['progress']['step'],4)
        for name in a['model']:torch.testing.assert_close(a['model'][name],b['model'][name],rtol=0,atol=0)
        self.assertEqual(a['progress']['history'],b['progress']['history'])
        self.assertTrue((resumed/'TRAINING_COMPLETE').exists())
        self.assertEqual((continuous/'test_predictions.csv').read_bytes(),(resumed/'test_predictions.csv').read_bytes())
        with self.assertRaises(FileExistsError):train(self.cfg,self.data,resumed,'downstream',device)
        changed=copy.deepcopy(self.cfg);changed['downstream']['lr']=.002
        with self.assertRaisesRegex(ValueError,'Resume refused'):
            train(changed,self.data,resumed,'downstream',device,resume=True)


if __name__=='__main__':unittest.main()
