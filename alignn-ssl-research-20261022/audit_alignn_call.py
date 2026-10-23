"""Numerically compare the current unmasked Backbone with original ALIGNN.

CPU only. Uses identical model weights and either synthetic crystals or a few
training-split crystals. Does not open test labels or start a training run.
"""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import torch
from torch import nn
from jarvis.core.atoms import Atoms
from ssl_alignn.data import CrystalData,sha256
from ssl_alignn.state import source_fingerprint
from ssl_alignn.model import Backbone
from ssl_alignn.runner import optimizer_parameters
from supervised_model import OriginalCrystalData,OriginalSupervisedModel


def synthetic_data(root):
    root=Path(root);(root/'structures').mkdir(parents=True)
    names=[];hashes={}
    for i in range(8):
        name=f'synthetic-{i}.vasp';names.append(name)
        atoms=Atoms(lattice_mat=[[3.4+i*.05,0,0],[.2,3.6,0],[.1,.3,3.8]],
            coords=[[0,0,0],[.22,.31,.39]],elements=['Si','Al' if i%2 else 'Si'],cartesian=False)
        p=root/'structures'/name;atoms.write_poscar(str(p));hashes[name]=sha256(p)
    manifest={'seed':123,'structure_sha256':hashes,'splits':{'train':names[:4],
        'val':names[4:6],'test':names[6:],'pretrain_train':names[:3],'pretrain_val':names[3:4]}}
    (root/'manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
    (root/'targets.json').write_text(json.dumps({n:-.1*i for i,n in enumerate(names)}),encoding='utf-8')


class ManualBaseline(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.backbone=Backbone(cfg);self.head=nn.Linear(cfg['dim'],1)
    def forward(self,b):
        rbf=self.backbone.edge_embedding[0](b['distance'])
        x,_=self.backbone(b['g'],b['lg'],b['x'],rbf,b['angles'])
        return self.head(self.backbone.readout(b['g'],x)).reshape(-1)


def compare_paths(original_batch,masked_batch,cfg):
    torch.manual_seed(123)
    native=OriginalSupervisedModel(cfg)
    manual=ManualBaseline(cfg)
    state=native.alignn.state_dict()
    manual.backbone.load_state_dict({k:v for k,v in state.items() if not k.startswith('fc.')})
    manual.head.load_state_dict(native.alignn.fc.state_dict())
    # These checks include edge order and line-graph connectivity, not just shape.
    for kind in ('g','lg'):
        for x,y in zip(original_batch[kind].edges(),masked_batch[kind].edges()):
            torch.testing.assert_close(x,y,rtol=0,atol=0)
    torch.testing.assert_close(original_batch['g'].ndata['atom_features'],masked_batch['x'],rtol=0,atol=0)
    torch.testing.assert_close(original_batch['g'].edata['r'].norm(dim=1),masked_batch['distance'],rtol=0,atol=0)
    torch.testing.assert_close(original_batch['lg'].edata['h'],masked_batch['angles'],rtol=0,atol=0)
    report={}
    for mode in ('eval','train'):
        native.train(mode=='train');manual.train(mode=='train')
        native.zero_grad(set_to_none=True);manual.zero_grad(set_to_none=True)
        x=native(original_batch);y=manual(masked_batch)
        torch.testing.assert_close(x,y,rtol=1e-5,atol=1e-6)
        report[mode+'_max_abs_output_difference']=float((x-y).abs().max().detach())
        target=torch.linspace(-1.1,-.3,len(x))
        (x-target).abs().mean().backward();(y-target).abs().mean().backward()
        largest=0.;unused=[]
        for name,p in native.alignn.named_parameters():
            other=dict(manual.head.named_parameters())[name[3:]] if name.startswith('fc.') else dict(manual.backbone.named_parameters())[name]
            if p.grad is None or other.grad is None:
                assert p.grad is None and other.grad is None,name
                # Original ALIGNN's final angle-state output is not read out.
                # Matching disconnected parameters are expected, not a mismatch.
                unused.append(name)
                continue
            torch.testing.assert_close(p.grad,other.grad,rtol=1e-4,atol=1e-6)
            largest=max(largest,float((p.grad-other.grad).abs().max()))
        report[mode+'_max_abs_gradient_difference']=largest
        report[mode+'_parameters_without_gradient_in_both']=unused
    opt1=torch.optim.AdamW(optimizer_parameters(native,True),lr=.001,weight_decay=1e-5)
    opt2=torch.optim.AdamW(optimizer_parameters(manual,True),lr=.001,weight_decay=1e-5)
    opt1.step();opt2.step()
    max_update=0.
    for name,p in native.alignn.named_parameters():
        other=dict(manual.head.named_parameters())[name[3:]] if name.startswith('fc.') else dict(manual.backbone.named_parameters())[name]
        torch.testing.assert_close(p,other,rtol=1e-5,atol=1e-6)
        max_update=max(max_update,float((p-other).abs().max().detach()))
    report.update(max_abs_weight_difference_after_adamw=max_update,
        parameters=sum(p.numel() for p in native.parameters()),
        all_parameters_trainable=all(p.requires_grad for p in native.parameters()),
        geometry_equal=True,passed=True)
    return report


def audit(data,cfg,count=4):
    # Deliberately exclude the supervised labels from the audit dataset objects.
    graph=dict(cutoff=8.,cutoff_extra=3.,max_neighbors=12)
    raw=OriginalCrystalData(data,graph,source_fingerprint(),supervised=False)
    masked=CrystalData(data,graph,source_fingerprint(),supervised=False)
    names=raw.manifest['splits']['train'][:count]
    result=compare_paths(raw.batch(names,torch.device('cpu')),masked.batch(names,torch.device('cpu')),cfg)
    result['materials']=names
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,help='Optional prepared dataset; only the first four train structures are read')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--formal-model',action='store_true',help='Use hidden256 and 4+4 layers instead of a fast tiny model')
    args=p.parse_args();torch.set_num_threads(1)
    cfg=dict(dim=256 if args.formal_model else 8,embedding_features=64 if args.formal_model else 8,
             alignn_layers=4 if args.formal_model else 1,gcn_layers=4 if args.formal_model else 1)
    if args.out.exists(): raise FileExistsError('Choose a new audit output path')
    if args.data: result=audit(args.data,cfg)
    else:
        with tempfile.TemporaryDirectory() as t:
            synthetic_data(t);result=audit(t,cfg)
    result['model_config']=cfg
    result['scope']='Unmasked forward, all parameter gradients, one AdamW update; not a 300-epoch equivalence claim.'
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
