"""Pure supervised control using the ORIGINAL ALIGNN forward, not FrozenDownstream.

The same Graph builder supplies the original node/edge fields. A separate cache
prevents mixing these graphs with the feature-stripped masked pretraining cache.
"""
import io
from pathlib import Path
import json

import torch
from torch import nn
import dgl
from jarvis.core.atoms import Atoms
from ssl_alignn.data import CrystalData, Graph, sha256, _lmdb
from ssl_alignn.state import atomic_save
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig


def original_config(cfg):
    """Match the existing formation-energy baseline's model configuration."""
    return ALIGNNAtomWiseConfig(name='alignn_atomwise', atom_input_features=92,
        hidden_features=cfg['dim'], embedding_features=cfg.get('embedding_features',64),
        alignn_layers=cfg['alignn_layers'], gcn_layers=cfg['gcn_layers'],
        edge_input_features=80, triplet_input_features=40, output_features=1,
        calculate_gradient=False, gradwise_weight=0, stresswise_weight=0,
        atomwise_weight=0, use_penalty=False, energy_mult_natoms=False,
        include_pos_deriv=False, use_cutoff_function=False, extra_features=0,
        classification=False, link='identity', lg_on_fly=True)


class OriginalSupervisedModel(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        # Exactly one randomly initialized upstream network, with its own fc.
        # No pretrained encoder, no atom_projection, no mask or frozen weights.
        self.alignn = ALIGNNAtomWise(original_config(cfg))

    def forward(self,batch):
        # Local graph views isolate upstream forward's writes from cached fields.
        result = self.alignn([batch['g'].local_var(), batch['lg'].local_var(), batch['lattice']])
        return result['out'].reshape(-1)


class OriginalCrystalData(CrystalData):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.cache = self.cache/'original_forward_v1'
        self.cache.mkdir(parents=True,exist_ok=True)

    def build(self,path):
        atoms=Atoms.from_poscar(str(path))
        g,lg=Graph.atom_dgl_multigraph(atoms,neighbor_strategy='k-nearest',
            atom_features='cgcnn',cutoff=self.cfg['cutoff'],
            max_neighbors=self.cfg['max_neighbors'],cutoff_extra=self.cfg['cutoff_extra'],
            use_canonize=True,compute_line_graph=True,dtype='float32')
        return {'g':g,'lg':lg,'lattice':torch.tensor(atoms.lattice_mat,dtype=torch.float32)}

    def get(self,name):
        path=self.root/'structures'/name
        expected=self.manifest['structure_sha256'][name]
        if sha256(path)!=expected:
            raise ValueError(f'Structure changed since preparation: {name}')
        if self.use_lmdb:
            env=_lmdb(self.cache/'graphs.lmdb');key=expected.encode('ascii')
            with env.begin() as txn:
                payload=txn.get(key)
            if payload is not None:
                return torch.load(io.BytesIO(payload),map_location='cpu',weights_only=False)
            value=self.build(path);buffer=io.BytesIO();torch.save(value,buffer)
            with env.begin(write=True) as txn:
                txn.put(key,buffer.getvalue(),overwrite=False)
            return value
        cache=self.cache/(expected+'.pt')
        if cache.exists():
            return torch.load(cache,map_location='cpu',weights_only=False)
        value=self.build(path);atomic_save(value,cache);return value

    def collate(self,items,device):
        names,values=zip(*items)
        batch={'names':list(names),'g':dgl.batch([v['g'] for v in values]).to(device),
               'lg':dgl.batch([v['lg'] for v in values]).to(device),
               'lattice':torch.stack([v['lattice'] for v in values]).to(device)}
        if self.targets is not None:
            batch['target']=torch.tensor([self.targets[n] for n in names],device=device)
        return batch
