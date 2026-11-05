"""Step 5: edge weights for one graph (inputs X -> targets Y).

Primary weight: leave-one-covariate-out (LOCO; Lei et al. 2018). For each
target j, fit a small MLP on all inputs, and one more MLP per input i with that
input removed. Weight(i, j) = log(held-out MSE without i / held-out MSE with all).
This measures the information only input i carries. Permutation importance was
tried first and rejected: it credits inputs that are merely correlated with a real
cause (e.g. vbar -> dtheta via its correlation with wbar).

All variants of one split are trained together as one batched network, from the
same initial weights (common random numbers), so their differences are not
dominated by initialisation noise.

Two decoy inputs (shuffled copies of u_t and theta) are added every time; their
LOCO weight is the "no effect" level. Repeated over random half-splits by
episode (fit on one half, score on the other). Stability = fraction of splits in
which an edge beats the strongest decoy for that target (used by prune.py).

Cross-check: group-lasso component-wise MLP (neural Granger causality,
Tank et al. 2021); weight = L2 norm of the input's first-layer column.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

DECOYS = ['decoy_u_t', 'decoy_theta']
DECOY_SOURCE = [4, 2]        # columns of X that the decoys are shuffled copies of
# An edge must raise held-out error at least 50% more than removing a decoy does.
# Fixed in advance (not tuned on the expert graph). Needed because inputs that are
# correlated with a real cause sit slightly above independent decoys.
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


def with_decoys(X, rng):
    return np.column_stack([X] + [X[rng.permutation(len(X)), c] for c in DECOY_SOURCE])


def loco_split(Xa, Ya, Xb, Yb, seed, device, steps):
    """One split: held-out MSE of the full model and of each drop-one model."""
    n_in, n_tg = Xa.shape[1], Ya.shape[1]
    n_var = n_in + 1                                       # variant 0 = all inputs
    mask = torch.ones(n_var, n_in)
    for i in range(n_in):
        mask[i + 1, i] = 0.0
    mask = mask.repeat_interleave(n_tg, dim=0)              # network k = variant k//n_tg, target k%n_tg
    torch.manual_seed(seed)
    net = GroupMLP(n_var * n_tg, n_in, mask=mask).to(device)
    net.tie_init(n_tg)
    T = lambda v: torch.as_tensor(v, dtype=torch.float32, device=device)  # noqa: E731
    train(net, T(Xa), T(Ya).repeat(1, n_var), steps=steps)
    with torch.no_grad():
        err = ((net(T(Xb)) - T(Yb).repeat(1, n_var)) ** 2).mean(0).reshape(n_var, n_tg)
    err = err.cpu().numpy()
    return err[0], err[1:]                                  # (n_tg,), (n_in, n_tg)


def score_graph(X, Y, episode, n_splits=50, seed=0, device='cuda', steps=1500, verbose=False):
    """Returns LOCO weights (n_in, n_tg), decoy weights, stability, held-out R2."""
    device = device if torch.cuda.is_available() else 'cpu'
    rng = np.random.default_rng(seed)
    n_in, n_tg = X.shape[1], Y.shape[1]
    L = np.zeros((n_splits, n_in + len(DECOYS), n_tg))     # log MSE ratio
    D = np.zeros_like(L)                                    # MSE increase (= R2 drop, standardised)
    R2 = np.zeros((n_splits, n_tg))
    for s in range(n_splits):
        Xd = with_decoys(X, rng)
        a, b = half_split(episode, rng)
        Xa, Xb = standardise(Xd[a], Xd[b])
        Ya, Yb = standardise(Y[a], Y[b])
        full, drop = loco_split(Xa, Ya, Xb, Yb, seed * 1000 + s, device, steps)
        L[s] = np.log(drop / full[None])
        D[s] = drop - full[None]
        R2[s] = 1 - full / Yb.var(0)
        if verbose and (s + 1) % 10 == 0:
            print(f'    split {s + 1}/{n_splits}  held-out R2={np.round(R2[:s + 1].mean(0), 4)}')
    decoy_max = L[:, n_in:, :].max(1)                       # (splits, n_tg)
    beats = L[:, :n_in, :] > decoy_max[:, None, :] + MARGIN
    return {
        'weight': L[:, :n_in, :].mean(0),
        'weight_sd': L[:, :n_in, :].std(0),
        'weight_r2_drop': D[:, :n_in, :].mean(0),
        'decoy_weight': L[:, n_in:, :].mean(0),
        'decoy_max_mean': decoy_max.mean(0),
        'stability': beats.mean(0),
        'heldout_R2': R2.mean(0),
        'n_splits': n_splits,
    }


def group_lasso_weights(X, Y, episode, lams=(0.0, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1), tol=0.01,
                        seed=0, device='cuda', steps=1500):
    """Neural-Granger cross-check. Per target, the largest lambda whose held-out R2 is
    within `tol` of the unpenalised fit; weight = first-layer row norm of each input."""
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
        fits.append((lam, r2, net.W1.detach().norm(dim=-1).cpu().numpy().T))   # (n_in, n_tg)
    ref = fits[0][1]
    weight = np.zeros((X.shape[1], Y.shape[1]))
    chosen = []
    for j in range(Y.shape[1]):
        lam, _, w = [f for f in fits if f[1][j] >= ref[j] - tol][-1]
        weight[:, j] = w[:, j]
        chosen.append(lam)
    return {'weight': weight, 'lambda': chosen}


def rank_agreement(w1, w2):
    return float(spearmanr(np.ravel(w1), np.ravel(w2)).statistic)
