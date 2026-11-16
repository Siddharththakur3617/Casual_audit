"""Effects and agreement measures for the counterfactual study.

Effect of intervention node i on target j in one window, in the expert effect-size unit:
    e = sign * (change' - change_factual) / (kappa * sd_train(target_j))
averaged over the valid signs (central difference when both are valid). Since delta_i = kappa * sd(o_i),
e is a finite-difference estimate of d target / d o_i * sd(o_i) / sd(target), the unit of
expert_graph_final_*.json effect_size.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr

SIGN_MIN = 0.05            # sign agreement is scored only where |reference effect| > SIGN_MIN (fixed in advance)
N_BOOT = 1000


def node_groups(Q):
    """{node name: [(variant index, sign), ...]} for every intervened node / physics parameter."""
    g = {}
    for v, (name, kind, sg) in enumerate(zip(Q['variants'], Q['kind'], Q['sign'])):
        if kind == 'factual':
            continue
        g.setdefault(str(name)[:-1], []).append((v, int(sg)))
    return g


def effects(D, Q, kappa, sd_t, groups=None):
    """D (n_var, n, ...) one source's prediction per variant (last axis = targets). Returns
    {node: (n, ...) effect, NaN where no sign of that node is valid}."""
    groups = groups or node_groups(Q)
    out = {}
    for node, vs in groups.items():
        acc = np.zeros(D.shape[1:])
        cnt = np.zeros(D.shape[1:2])
        for v, sg in vs:
            ok = Q['valid'][v]
            diff = sg * (D[v] - D[0]) / (kappa * sd_t)
            acc[ok] += diff[ok]
            cnt[ok] += 1
        e = acc / np.maximum(cnt, 1).reshape(-1, *([1] * (acc.ndim - 1)))
        e[cnt == 0] = np.nan
        out[node] = e
    return out


def ce(e):
    """Mean |effect| over the valid windows, per target (and horizon)."""
    return np.nanmean(np.abs(e), axis=0)


def ace(e):
    return np.nanmean(e, axis=0)


def bootstrap_ce(e, episode, n_boot=N_BOOT, seed=0):
    """95% episode-bootstrap interval of mean |e|. e (n, ...)."""
    ok = np.isfinite(e).reshape(len(e), -1).all(1)
    a, ep = np.abs(e[ok]), episode[ok]
    if len(a) == 0:
        nan = np.full(e.shape[1:], np.nan)
        return nan, nan
    u, inv = np.unique(ep, return_inverse=True)
    sums = np.zeros((len(u),) + a.shape[1:])
    np.add.at(sums, inv, a)
    cnt = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=n_boot).astype(float)    # (B, E)
    flat = sums.reshape(len(u), -1)
    stat = (w @ flat) / (w @ cnt)[:, None]
    lo, hi = np.percentile(stat, [2.5, 97.5], axis=0)
    return lo.reshape(a.shape[1:]), hi.reshape(a.shape[1:])


def agreement(e_s, e_ref):
    """Unit-level agreement of one source's effects with a reference's, for one cell (1-d arrays over windows)."""
    ok = np.isfinite(e_s) & np.isfinite(e_ref)
    s, r = e_s[ok], e_ref[ok]
    if len(r) < 2:
        return {'n': int(len(r)), 'cf_r2': None, 'rmse': None, 'sign_agree': None}
    var = r.var()
    big = np.abs(r) > SIGN_MIN
    return {'n': int(len(r)),
            'cf_r2': float(1 - ((s - r) ** 2).mean() / var) if var > 1e-12 else None,
            'rmse': float(np.sqrt(((s - r) ** 2).mean())),
            'sign_agree': float((np.sign(s[big]) == np.sign(r[big])).mean()) if big.any() else None}


def matrix_agreement(E_s, E_ref, rows, cols):
    """Pooled agreement over the cells (rows x cols) of two effect dicts {node: (n, n_tg)}."""
    cells, ce_s, ce_r, num, den = {}, [], [], 0.0, 0.0
    for node in rows:
        for j in cols:
            a = agreement(E_s[node][:, j], E_ref[node][:, j])
            cells[(node, j)] = a
            ce_s.append(np.nanmean(np.abs(E_s[node][:, j])))
            ce_r.append(np.nanmean(np.abs(E_ref[node][:, j])))
            ok = np.isfinite(E_s[node][:, j]) & np.isfinite(E_ref[node][:, j])
            r = E_ref[node][ok, j]
            num += ((E_s[node][ok, j] - r) ** 2).sum()
            den += ((r - r.mean()) ** 2).sum() if len(r) else 0.0
    ce_s, ce_r = np.asarray(ce_s), np.asarray(ce_r)
    fin = np.isfinite(ce_s) & np.isfinite(ce_r)
    sp = spearmanr(ce_s[fin], ce_r[fin]).statistic if fin.sum() > 2 and np.ptp(ce_r[fin]) > 0 else np.nan
    return {'cells': cells, 'ce_spearman': float(sp), 'pooled_cf_r2': float(1 - num / den) if den > 0 else None}


def prf(truth, pred):
    truth, pred = np.asarray(truth, bool), np.asarray(pred, bool)
    tp, fp, fn = int((truth & pred).sum()), int((~truth & pred).sum()), int((truth & ~pred).sum())
    p = tp / (tp + fp) if tp + fp else float('nan')
    r = tp / (tp + fn) if tp + fn else float('nan')
    return {'precision': p, 'recall': r, 'f1': 2 * p * r / (p + r) if tp else 0.0, 'tp': tp, 'fp': fp, 'fn': fn}
