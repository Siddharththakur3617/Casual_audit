"""Graph-restricted structural models: one masked MLP per (variant, target), the target seeing only its parents.

A variant is (parents (n_in, n_tg) bool, data source y, use_physics). All variants and targets of one seed are
trained together as one discovery.edge_scores.GroupMLP (same architecture, training loop and standardisation
as the edge scores), then averaged over seeds. Inputs go through env.edge_features, so Acrobot angles enter as
(sin, cos) pairs and a node's mask covers all its columns. Physics, when used, is appended as known inputs
(like score_graph's cond); it is not a graph node.

The counterfactual answer of an SCM is f(o') - f(o): with additive noise the abducted noise cancels.
"""
from __future__ import annotations

import numpy as np
import torch

import envs
from discovery import edge_scores as es


class SCMBank:
    def __init__(self, variants, X, phys, seeds=(0, 1, 2), steps=1500, device='cuda'):
        """variants: {name: (parents (n_in, n_tg), Y (n, n_tg), use_physics)}; X (n, n_in) graph inputs,
        phys (n, n_phys) episode physics of every row."""
        env = envs.current()
        self.names = list(variants)
        Xf, groups = env.edge_features(X)
        self.groups = groups or [[i] for i in range(Xf.shape[1])]
        self.n_tg = next(iter(variants.values()))[1].shape[1]
        self.n_x = Xf.shape[1]
        P = np.asarray(phys, np.float64)
        F = np.column_stack([Xf, P])
        self.fm = F.mean(0)
        self.fs = F.std(0)
        self.fs[self.fs < 1e-9] = np.inf               # constant columns (fixed-physics data) carry nothing
        mask, Ys, self.ym, self.ys = [], [], [], []
        for name in self.names:
            par, Y, use_p = variants[name]
            par = np.asarray(par, bool)
            ym, ys = Y.mean(0), Y.std(0) + 1e-12
            self.ym.append(ym); self.ys.append(ys)
            Ys.append((Y - ym) / ys)
            for j in range(self.n_tg):
                m = np.zeros(F.shape[1])
                for i in np.where(par[:, j])[0]:
                    m[self.groups[i]] = 1.0
                if use_p:
                    m[self.n_x:] = 1.0
                mask.append(m)
        self.mask = torch.tensor(np.asarray(mask), dtype=torch.float32)
        self.device = device if torch.cuda.is_available() else 'cpu'
        Fa = torch.as_tensor((F - self.fm) / self.fs, dtype=torch.float32, device=self.device)
        Ya = torch.as_tensor(np.concatenate(Ys, 1), dtype=torch.float32, device=self.device)
        self.nets = []
        for s in seeds:
            torch.manual_seed(1000 + s)
            net = es.GroupMLP(len(mask), F.shape[1], mask=self.mask).to(self.device)
            es.train(net, Fa, Ya, steps=steps)
            self.nets.append(net.eval())
        with torch.no_grad():
            pred = self._raw(Fa)
        self.train_r2 = {n: (1 - ((pred[:, v] - Ys[v]) ** 2).mean(0) / Ys[v].var(0)).tolist()
                         for v, n in enumerate(self.names)}

    def _raw(self, Fa):
        out = np.mean([net(Fa).cpu().numpy() for net in self.nets], axis=0)
        return out.reshape(len(Fa), len(self.names), self.n_tg)

    @torch.no_grad()
    def predict(self, X, phys):
        """X (m, n_in), phys (m, n_phys) -> {name: (m, n_tg)} in the target's own units."""
        env = envs.current()
        Xf, _ = env.edge_features(np.asarray(X, np.float64))
        F = np.column_stack([Xf, np.asarray(phys, np.float64)])
        out = []
        for i in range(0, len(F), 65536):
            Fa = torch.as_tensor((F[i:i + 65536] - self.fm) / self.fs, dtype=torch.float32, device=self.device)
            out.append(self._raw(Fa))
        raw = np.concatenate(out)
        return {n: raw[:, v] * self.ys[v] + self.ym[v] for v, n in enumerate(self.names)}
