"""Thin adapters around the unchanged ALIGNNAtomWise components."""
import torch
from torch import nn
from torch.nn import functional as F
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig


class Backbone(nn.Module):
    def __init__(self, cfg, direct_z=False):
        super().__init__()
        base = ALIGNNAtomWise(ALIGNNAtomWiseConfig(
            name="alignn_atomwise",
            atom_input_features=cfg.get("z_dim", cfg["dim"]) if direct_z else 92,
            # dim 是消息传递宽度；embedding_features 是距离/角度 MLP
            # 中间宽度。z_dim 是预训练末端压缩后的原子表示，三者独立。
            hidden_features=cfg["dim"],
            embedding_features=cfg.get("embedding_features", cfg["dim"]),
            alignn_layers=cfg["alignn_layers"], gcn_layers=cfg["gcn_layers"],
            edge_input_features=80, triplet_input_features=40,
            calculate_gradient=False, gradwise_weight=0, stresswise_weight=0,
            use_penalty=False, energy_mult_natoms=False,
        ))
        # 下游的 64 维 Z 先经上游的 MLPLayer 映射到 hidden=256。
        # 当 Z 与 hidden 都为 256（直传对照）时，用 Identity 直接传入。
        self.atom_embedding = (nn.Identity() if direct_z and
                               cfg.get("z_dim", cfg["dim"]) == cfg["dim"]
                               else base.atom_embedding)
        self.edge_embedding = base.edge_embedding
        self.angle_embedding = base.angle_embedding
        self.alignn_layers = base.alignn_layers
        self.gcn_layers = base.gcn_layers
        self.readout = base.readout

    def forward(self, g, lg, x, edge_rbf, angles):
        # Only explicit tensors enter the model. Graph objects contain topology only.
        x = self.atom_embedding(x)
        y = self.edge_embedding[1:](edge_rbf)
        z = self.angle_embedding(angles)
        for layer in self.alignn_layers:
            x, y, z = layer(g, lg, x, y, z)  # upstream: atom graph THEN line graph
        for layer in self.gcn_layers:
            x, y = layer(g, x, y)
        return x, y


class MaskedModel(nn.Module):
    def __init__(self, cfg, mask_kind):
        super().__init__()
        self.encoder = Backbone(cfg)
        # 可训练瓶颈必须位于重建头之前，才能从原子重建损失中获得梯度。
        # 下游使用同一个瓶颈并一起冻结，不在预训练结束后随机添加投影。
        z_dim = cfg.get("z_dim", cfg["dim"])
        self.atom_projection = (nn.Linear(cfg["dim"], z_dim)
                                if z_dim != cfg["dim"] else nn.Identity())
        if mask_kind not in ("zero", "learnable"):
            raise ValueError(mask_kind)
        for name, width in (("atom_mask", 92), ("edge_mask", 80)):
            value = torch.zeros(width)
            if mask_kind == "learnable":
                self.register_parameter(name, nn.Parameter(value))
            else:
                self.register_buffer(name, value)
        self.period_head = nn.Linear(z_dim, 9)
        self.group_head = nn.Linear(z_dim, 18)
        self.distance_head = nn.Linear(cfg["dim"], 80)

    def inputs(self, batch, node_mask=None, edge_mask=None):
        x = batch["x"]
        rbf = self.encoder.edge_embedding[0](batch["distance"])
        if node_mask is not None:
            x = torch.where(node_mask[:, None], self.atom_mask, x)
            rbf = torch.where(edge_mask[:, None], self.edge_mask, rbf)
        return x, rbf

    def encode(self, batch):
        x, rbf = self.inputs(batch)
        hidden = self.encoder(batch["g"], batch["lg"], x, rbf, batch["angles"])[0]
        return self.atom_projection(hidden)

    def forward(self, batch, node_mask, edge_mask):
        x, rbf = self.inputs(batch, node_mask, edge_mask)
        z, e = self.encoder(batch["g"], batch["lg"], x, rbf, batch["angles"])
        z = self.atom_projection(z)
        return self.period_head(z[node_mask]), self.group_head(z[node_mask]), self.distance_head(e[edge_mask])


def sample_masks(batch, node_ratio, edge_ratio, seed):
    """Per-crystal selection; each physical bond's two directions share a mask."""
    rng = torch.Generator().manual_seed(seed)
    nodes, edges = [], []
    for n, pair_ids in zip(batch["node_counts"], batch["pair_ids"]):
        nm = torch.zeros(n, dtype=torch.bool)
        nm[torch.randperm(n, generator=rng)[:max(1, round(n * node_ratio))]] = True
        pairs = int(pair_ids.max()) + 1
        pm = torch.zeros(pairs, dtype=torch.bool)
        pm[torch.randperm(pairs, generator=rng)[:max(1, round(pairs * edge_ratio))]] = True
        nodes.append(nm)
        edges.append(pm[pair_ids])
    return torch.cat(nodes).to(batch["x"].device), torch.cat(edges).to(batch["x"].device)


def reconstruction_loss(model, batch, masks, weights):
    nm, em = masks
    p, g, d = model(batch, nm, em)
    centers = model.encoder.edge_embedding[0].centers
    bins = (batch["distance"][:, None] - centers).abs().argmin(1)
    losses = [F.cross_entropy(p, batch["period"][nm]),
              F.cross_entropy(g, batch["group"][nm]), F.cross_entropy(d, bins[em])]
    total = sum(w * loss for w, loss in zip(weights, losses))
    counts = [int(nm.sum()), int(nm.sum()), int(em.sum())]
    correct = [int((a.argmax(1) == b).sum()) for a, b in
               ((p, batch["period"][nm]), (g, batch["group"][nm]), (d, bins[em]))]
    return total, [float(v.detach()) for v in losses], counts, correct


class FrozenDownstream(nn.Module):
    def __init__(self, pretrained, cfg):
        super().__init__()
        self.pretrained = pretrained.requires_grad_(False).eval()
        self.network = Backbone(cfg, direct_z=True)
        self.head = nn.Linear(cfg["dim"], 1)

    def train(self, mode=True):
        super().train(mode)
        self.pretrained.eval()
        return self

    def forward(self, batch):
        with torch.no_grad():
            z = self.pretrained.encode(batch)
        rbf = self.network.edge_embedding[0](batch["distance"])
        x, _ = self.network(batch["g"], batch["lg"], z, rbf, batch["angles"])
        return self.head(self.network.readout(batch["g"], x)).reshape(-1)
