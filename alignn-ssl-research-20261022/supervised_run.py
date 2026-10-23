"""Original single-backbone ALIGNN control selected by the formal configuration."""
import argparse
import json
from pathlib import Path
import os
import sys
from contextlib import redirect_stdout,redirect_stderr
import torch
from filelock import FileLock
import run_formal
import supervised_runner as runner
from ssl_alignn import runner as shared_runner
from ssl_alignn.data import digest,sha256
from terminal_display import TerminalDisplay

CODE=Path(__file__).resolve().parent
DEFAULT_OUT=CODE/'runs/supervised_10000'


def main(config=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--data',type=Path,default=run_formal.REFERENCE_DATA)
    parser.add_argument('--mode',choices=('both','learnable','zero','zero_direct'),default='both',
                        help='Select the formal configuration; both runs one control only if settings match')
    parser.add_argument('--check',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--pause',action='store_true')
    args=parser.parse_args()
    root=args.out.resolve();data_source=args.data.resolve()
    if root==data_source or root.is_relative_to(data_source):
        raise ValueError('Output must not be inside source dataset')
    if any((root/n).exists() for n in ('learnable','zero','zero_direct')):
        raise ValueError('Refusing to write supervised control into an SSL experiment directory')
    if args.pause:
        status_path=root/'status.json'
        if not status_path.exists(): raise RuntimeError('No supervised run at --out')
        status=run_formal.read_json(status_path)
        if status.get('method')!='supervised_original': raise ValueError('Not a supervised-control directory')
        (root/'STOP_REQUESTED').touch()
        if (root/'supervised').exists(): (root/'supervised/STOP_REQUESTED').touch()
        print('Pause requested; wait for PAUSED before closing the terminal.')
        return
    if config is None:
        from experiment_options import select_experiment
        enabled, config = select_experiment(sys.argv[1:])
        if enabled:
            raise ValueError('Set self_supervised=false in the selected formal configuration first')
    cfg=config
    runner.validate_config(cfg)
    print('self_supervised=false: original ALIGNN, random initialization, ALL parameters trainable.')
    print('No masks, no Z bottleneck, no pretrained checkpoint. --mode does not select extra runs.')
    print(f'Results: {root}\nData source: {data_source}')
    print(f"Training: {cfg['downstream']['epochs']} epochs; hidden={cfg['model']['dim']}; "
          f"batch={cfg['downstream']['batch_size']}; microbatch={cfg['downstream']['microbatch_size']}; workers=0")
    if not torch.cuda.is_available(): raise RuntimeError('Formal supervised training requires CUDA')
    if args.check:
        print('CHECK OK. No training or data preparation started.');return
    root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(root/'.formal.lock'),timeout=0):
        protocol={'method':'supervised_original','config':cfg,'sources':runner.source_fingerprint(),
                  'torch':torch.__version__,'device':'cuda'}
        protocol_path=root/'protocol.json'
        if protocol_path.exists() and run_formal.read_json(protocol_path)!=protocol:
            raise ValueError('Protocol changed; use original control files or a new --out')
        data=root/'data10000'
        if not data.exists():
            temporary=root/'data10000.preparing'
            if temporary.exists(): raise RuntimeError(f'Inspect interrupted preparation: {temporary}')
            run_formal.REFERENCE_DATA=data_source
            run_formal.copy_reference_data(temporary)
            temporary.rename(data)
        manifest=run_formal.read_json(data/'manifest.json')
        counts={k:len(v) for k,v in manifest['splits'].items()}
        if counts!=dict(train=8000,val=1000,test=1000,pretrain_train=7200,pretrain_val=800) or manifest['seed']!=123:
            raise ValueError('Expected the same fixed 10,000-material split')
        identity={'manifest':digest(manifest),'targets':sha256(data/'targets.json')}
        identity_path=root/'dataset_hash.json'
        if identity_path.exists() and run_formal.read_json(identity_path)!=identity:
            raise ValueError('Dataset identity changed')
        run_formal.write_json(protocol_path,protocol)
        run_formal.write_json(identity_path,identity)
        (root/'configs').mkdir(exist_ok=True)
        run_formal.write_json(root/'configs/formal_supervised.json',cfg)
        if args.prepare_only:
            print('PREPARE OK. No training started.');return
        out=root/'supervised'
        if (root/'STOP_REQUESTED').exists() or (out/'STOP_REQUESTED').exists():
            print('PAUSED: remove the requested pause marker only when intentionally resuming.');return
        if (out/'TRAINING_COMPLETE').exists():
            print('SKIP complete: supervised');return
        out.mkdir(exist_ok=True)
        run_formal.update_status(root,state='running',method='supervised_original',pid=os.getpid(),active_directory=str(out))
        display=TerminalDisplay();display.epochs={'downstream':cfg['downstream']['epochs']}
        try:
            with (out/'console.log').open('a',encoding='utf-8') as log:
                with redirect_stdout(run_formal.Tee(sys.stdout,log)),redirect_stderr(run_formal.Tee(sys.stderr,log)):
                    # Both the control loop and reused helpers share the same display.
                    with display.installed(sys.modules[__name__],shared_runner),display.installed(sys.modules[__name__],runner):
                        runner.train(cfg,data,out,'downstream',torch.device('cuda'),
                                     resume=(out/'training_state.pt').exists())
            run_formal.update_status(root,method='supervised_original',
                state='complete' if (out/'TRAINING_COMPLETE').exists() else 'paused')
        except Exception as error:
            run_formal.update_status(root,state='failed',method='supervised_original',error=str(error))
            raise


if __name__=='__main__':
    main()
