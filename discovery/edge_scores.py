"""Step 5: edge scores for one graph (inputs X -> targets Y).

Primary weight: leave-one-covariate-out (LOCO; Lei et al. 2018). For each
target j, fit a small MLP on all inputs, and one more MLP per input i with that
input removed. Weight(i, j) = log(held-out MSE without i / held-out MSE with all).
This measures the information only input i carries. Permutation importance was
tried first and rejected: it credits inputs that are merely correlated with a real
cause (e.g. vbar -> dtheta via its correlation with wbar).

All variants of one split are trained together as one batched network, from the
same initial weights (common random numbers), so their differences are not
dominated by initialisation noise. Repeated over random half-splits by episode
(fit on one half, score on the other).

Negative controls ("decoys"), two kinds:
  decoys='shuffled'  shuffled copies of u_t and theta (independent of everything).
  decoys='knockoff'  Gaussian model-X knockoffs of all 7 inputs, drawn fresh for every split
                     (Candes et al. 2018). A knockoff copies the input's correlation with the
                     other inputs but carries no information about Y beyond them.

Everything the pruning rules in prune.py need is kept per split (scores['per_split']):
  L     log MSE ratio (LOCO weight)                          (splits, n_cols, n_tg)
  D     MSE increase in units of target variance (R2 drop)   (splits, n_cols, n_tg)
  Dvs   D(edge) - D(strongest decoy), and its episode-clustered standard error
  J     normalised Jacobian effect of the full fit, mean |df/dx_i| * sd(x_i) / sd(y), i.e.
        the same unit as the expert effect sizes (checks/verify_equations.py, check e)

Cross-check: group-lasso component-wise MLP (neural Granger causality,
Tank et al. 2021); weight = L2 norm of the input's first-layer column.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

DECOYS = ['decoy_u_t', 'decoy_theta']
DECOY_SOURCE = [4, 2]        # columns of X that the shuffled decoys are copies of (cart-pole; the active
                             # environment's envs.<env>.DECOY_SOURCE is used at run time)
# Margin of the original pruning rule (prune.rule_margin): an edge must raise held-out error at
# least 50% more than removing a decoy does. Fixed in advance (not tuned on the expert graph).
MARGIN = float(np.log(1.5))


class GroupMLP(nn.Module):
    """K independent MLPs evaluated in parallel. Network k sees x * mask[k]."""

    def __init__(self, k, n_in, hidden=64, mask=None):
        super().__init__()
        s = lambda *shape: nn.Parameter(torch.randn(*shape) / np.sqrt(shape[-2]))  # noqa: E731
        self.W1, self.b1 = s(k, n_in, hidden), nn.Parameter(torch.zeros(k, 1, hidden))
        self.W2, self.b2 = s(k, hidden, hidden), nn.Parameter(torch.zeros(k, 1, hidden))
        self.W3, self.b3 = s(k, hidden, 1), nn.Parameter(torch.zeros(k, 1, 1))
        self.register_buffer('mask', torch.ones(k, 1, n_in) if mask is None else mask[:, None, :])

    def forward(self, x):                       # x: (n, n_in) -> (n, k)
        x = x[None] * self.mask
        h = nn.functional.silu(torch.bmm(x, self.W1) + self.b1)
        h = nn.functional.silu(torch.bmm(h, self.W2) + self.b2)
        return (torch.bmm(h, self.W3) + self.b3)[..., 0].T

    def tie_init(self, group):
        """Copy the initial weights of the first `group` networks to every block of `group`."""
        with torch.no_grad():
            for p in (self.W1, self.b1, self.W2, self.b2, self.W3, self.b3):
                p.copy_(p[:group].repeat(p.shape[0] // group, 1, 1))

    def group_prox(self, thr):
        """Group soft-threshold on each (network, input) first-layer row."""
        with torch.no_grad():
            norm = self.W1.norm(dim=-1, keepdim=True)
            self.W1.mul_(torch.clamp(1 - thr / norm.clamp_min(1e-12), min=0))


def train(net, Xa, Ya, steps=1500, lr=3e-3, lam=0.0):
    """Ya: (n, k) one target column per network."""
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = ((net(Xa) - Ya) ** 2).mean(0).sum()
        loss.backward()
        opt.step()
        if lam > 0:
            net.group_prox(lam * sched.get_last_lr()[0])
        sched.step()
    return net


def half_split(episode, rng):
    ids = np.unique(episode)
    rng.shuffle(ids)
    a = np.isin(episode, ids[:len(ids) // 2])
    return a, ~a


def standardise(Xa, Xb):
    m, s = Xa.mean(0), Xa.std(0) + 1e-12
    return (Xa - m) / s, (Xb - m) / s


def with_decoys(X, rng, source=None):
    source = DECOY_SOURCE if source is None else source
    return np.column_stack([X] + [X[rng.permutation(len(X)), c] for c in source])


def with_group_decoys(X, groups, rng, source):
    """Decoys for grouped nodes: each decoy is a row-shuffled copy of all columns of a source node.
    Returns (X with decoy columns, groups including the decoy nodes)."""
    cols, groups = [X], [list(g) for g in groups]
    n = X.shape[1]
    for src in source:
        c = groups[src]
        cols.append(X[rng.permutation(len(X))][:, c])
        groups.append(list(range(n, n + len(c))))
        n += len(c)
    return np.column_stack(cols), groups


# ---------------------------------------------------------------------------
# Gaussian model-X knockoffs (second-order; Candes, Fan, Janson & Lv 2018)
# ---------------------------------------------------------------------------

def knockoff_s(Sigma):
    """SDP knockoff parameters: maximise sum(s) s.t. 2*Sigma - diag(s) >= 0, 0 <= s <= 1
    (Sigma is a correlation matrix). Small p, so SLSQP on the smallest eigenvalue is enough."""
    from scipy.optimize import minimize
    p = len(Sigma)
    lam_min = float(np.linalg.eigvalsh(Sigma).min())
    s0 = np.full(p, min(1.0, 2 * lam_min) * 0.99)                  # equicorrelated start (feasible)
    cons = {'type': 'ineq', 'fun': lambda s: np.linalg.eigvalsh(2 * Sigma - np.diag(s)).min() - 1e-6}
    res = minimize(lambda s: -s.sum(), s0, bounds=[(0, 1)] * p, constraints=[cons], method='SLSQP',
                   options={'maxiter': 500, 'ftol': 1e-10})
    s = res.x if res.success and cons['fun'](res.x) >= -1e-9 else s0
    return np.clip(s, 0, 1)


class GaussianKnockoffs:
    def __init__(self, X):
        self.m, self.sd = X.mean(0), X.std(0) + 1e-12
        Z = (X - self.m) / self.sd
        self.Sigma = np.corrcoef(Z, rowvar=False)
        self.s = knockoff_s(self.Sigma)
        D = np.diag(self.s)
        Si = np.linalg.inv(self.Sigma)
        self.A = np.eye(len(D)) - Si @ D                             # E[Xk | X] = Z A
        V = 2 * D - D @ Si @ D
        w, U = np.linalg.eigh((V + V.T) / 2)
        self.Lc = U * np.sqrt(np.clip(w, 0, None))                   # V = Lc Lc^T

    def sample(self, X, rng):
        Z = (X - self.m) / self.sd
        Zk = Z @ self.A + rng.standard_normal(Z.shape) @ self.Lc.T
        return Zk * self.sd + self.m


# ---------------------------------------------------------------------------

def _cluster_mean_se(a, n_e):
    """a: (E, ...) per-episode sums, n_e: (E,) counts. Ratio-estimator mean and its
    episode-clustered standard error."""
    N = n_e.sum()
    theta = a.sum(0) / N
    r = a - theta[None] * n_e.reshape(-1, *([1] * (a.ndim - 1)))
    E = len(n_e)
    return theta, np.sqrt(E / (E - 1) * (r ** 2).sum(0)) / N


def loco_split(Xa, Ya, Xb, Yb, ep_b, seed, device, steps, n_dec, drop=None):
    """One split. Returns per-split arrays (see module docstring). The last n_dec columns of X are
    decoys/knockoffs. drop: the columns that get a leave-out variant (default all); other columns
    (conditioning inputs) are seen by every variant and get no score. Outputs are indexed by drop."""
    n_cols, n_tg = Xa.shape[1], Ya.shape[1]
    drop = list(range(n_cols)) if drop is None else list(drop)
    drop = [list(d) if isinstance(d, (list, tuple)) else [d] for d in drop]   # a node = one or more columns
    n_in = len(drop)
    n_var = n_in + 1                                       # variant 0 = all inputs
    mask = torch.ones(n_var, n_cols)
    for v, cols in enumerate(drop):
        mask[v + 1, cols] = 0.0
    mask = mask.repeat_interleave(n_tg, dim=0)              # network k = variant k//n_tg, target k%n_tg
    torch.manual_seed(seed)
    net = GroupMLP(n_var * n_tg, n_cols, mask=mask).to(device)
    net.tie_init(n_tg)
    T = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)  # noqa: E731
    train(net, T(Xa), T(Ya).repeat(1, n_var), steps=steps)
    xb = T(Xb).requires_grad_(True)
    out = net(xb)
    J = np.zeros((n_in, n_tg))
    single = all(len(c) == 1 for c in drop)
    for j in range(n_tg):                                   # networks 0..n_tg-1 = full model
        g, = torch.autograd.grad(out[:, j].sum(), xb, retain_graph=j < n_tg - 1)
        if single:
            J[:, j] = g.abs().mean(0).detach().cpu().numpy()[[c[0] for c in drop]]
        else:                                               # grouped node: gradient norm over its columns
            g = g.detach()
            J[:, j] = [g[:, c].norm(dim=1).mean().item() for c in drop]
    with torch.no_grad():
        se = ((out.detach() - T(Yb).repeat(1, n_var)) ** 2).reshape(len(Xb), n_var, n_tg).cpu().numpy()
    vy = Yb.var(0)
    full, drop = se[:, 0].mean(0), se[:, 1:].mean(0)        # (n_tg,), (n_in, n_tg)
    L = np.log(drop / full[None])
    D = (drop - full[None]) / vy[None]
    # strongest decoy per target, and the edge-minus-decoy contrast with an episode-clustered SE
    dmax = np.argmax(L[n_in - n_dec:], axis=0) + n_in - n_dec   # (n_tg,)
    inc = (se[:, 1:] - se[:, :1]) / vy[None, None]          # (n, n_in, n_tg) per-sample increase
    contrast = inc - inc[:, dmax, np.arange(n_tg)][:, None, :]
    eps_u, inv = np.unique(ep_b, return_inverse=True)
    a = np.zeros((len(eps_u), n_in, n_tg))
    np.add.at(a, inv, contrast)
    Dvs, Dvs_se = _cluster_mean_se(a, np.bincount(inv).astype(float))
    return {'L': L, 'D': D, 'Dvs': Dvs, 'Dvs_se': Dvs_se, 'J': J, 'R2': 1 - full / vy}


def score_graph(X, Y, episode, n_splits=50, seed=0, device='cuda', steps=1500, decoys='shuffled',
                verbose=False, cond=None):
    """Scores for every edge (inputs X -> targets Y). Returns summary arrays plus 'per_split'.

    decoys='shuffled' reproduces the original scores exactly (same random stream).
    cond: optional (n, k) conditioning inputs (e.g. episode physics). Every fit sees them, none is
    left out and they are not graph nodes: the edges are then scored with cond held known."""
    import envs
    env = envs.current()
    X, groups = env.edge_features(X)          # cart-pole: X itself, one column per node (groups None)
    device = device if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(seed)
    n_in = X.shape[1] if groups is None else len(groups)
    n_tg = Y.shape[1]
    n_c = 0 if cond is None else np.asarray(cond).shape[1]
    if decoys == 'shuffled':
        n_dec, ko = len(env.DECOY_SOURCE), None
    elif decoys == 'knockoff':
        if groups is not None:
            raise NotImplementedError('knockoff decoys are not implemented for grouped (sin, cos) nodes')
        n_dec, ko = n_in, GaussianKnockoffs(X)
    else:
        raise ValueError(decoys)
    keys = ('L', 'D', 'Dvs', 'Dvs_se', 'J', 'R2')
    per = {k: [] for k in keys}
    for s in range(n_splits):
        if groups is None:
            Xd = (with_decoys(X, rng, env.DECOY_SOURCE) if ko is None
                  else np.column_stack([X, ko.sample(X, rng)]))
            drop = None
            if n_c:                                         # columns: inputs, decoys, then cond
                drop = list(range(Xd.shape[1]))
        else:
            Xd, drop = with_group_decoys(X, groups, rng, env.DECOY_SOURCE)
        if n_c:
            Xd = np.column_stack([Xd, cond])
        a, b = half_split(episode, rng)
        Xa, Xb = standardise(Xd[a], Xd[b])
        Ya, Yb = standardise(Y[a], Y[b])
        r = loco_split(Xa, Ya, Xb, Yb, episode[b], seed * 1000 + s, device, steps, n_dec, drop=drop)
        for k in keys:
            per[k].append(r[k])
        if verbose and (s + 1) % 10 == 0:
            print(f'    split {s + 1}/{n_splits}  held-out R2={np.round(np.mean(per["R2"], 0), 4)}', flush=True)
    per = {k: np.asarray(v) for k, v in per.items()}
    L = per['L']
    decoy_max = L[:, n_in:, :].max(1)                       # (splits, n_tg)
    out = {
        'decoys': decoys,
        'weight': L[:, :n_in, :].mean(0),
        'weight_sd': L[:, :n_in, :].std(0),
        'weight_r2_drop': per['D'][:, :n_in, :].mean(0),
        'decoy_weight': L[:, n_in:, :].mean(0),
        'decoy_max_mean': decoy_max.mean(0),
        'jacobian_effect': per['J'][:, :n_in, :].mean(0),
        'heldout_R2': per['R2'].mean(0),
        'n_splits': n_splits,
        'n_cond': n_c,
        'per_split': per,
    }
    if decoys == 'shuffled':                                 # the original rule's stability
        out['stability'] = (L[:, :n_in, :] > decoy_max[:, None, :] + MARGIN).mean(0)
    else:
        out['knockoff_s'] = ko.s
    return out


def group_lasso_weights(X, Y, episode, lams=(0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1), tol=0.01,
                        seed=0, device='cuda', steps=1500):
    """Neural-Granger cross-check. Per target, the largest lambda whose held-out R2 is
    within `tol` of the unpenalised fit; weight = first-layer row norm of each input."""
    import envs
    X, groups = envs.current().edge_features(X)
    device = device if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(seed + 7)
    a, b = half_split(episode, rng)
    Xa, Xb = standardise(X[a], X[b])
    Ya, Yb = standardise(Y[a], Y[b])
    T = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)  # noqa: E731
    fits = []
    for lam in lams:
        torch.manual_seed(seed)
        net = train(GroupMLP(Y.shape[1], X.shape[1]).to(device), T(Xa), T(Ya), steps=steps, lam=lam)
        with torch.no_grad():
            r2 = (1 - ((net(T(Xb)) - T(Yb)) ** 2).mean(0) / T(Yb).var(0)).cpu().numpy()
        w = net.W1.detach().norm(dim=-1).cpu().numpy().T                     # (n_cols, n_tg)
        if groups is not None:                                              # node norm over its columns
            w = np.stack([np.sqrt((w[g] ** 2).sum(0)) for g in groups])
        fits.append((lam, r2, w))
    ref = fits[0][1]
    weight = np.zeros((fits[0][2].shape[0], Y.shape[1]))
    chosen = []
    for j in range(Y.shape[1]):
        lam, _, w = [f for f in fits if f[1][j] >= ref[j] - tol][-1]
        weight[:, j] = w[:, j]
        chosen.append(lam)
    return {'weight': weight, 'lambda': chosen}


def rank_agreement(w1, w2):
    return float(spearmanr(np.ravel(w1), np.ravel(w2)).statistic)
