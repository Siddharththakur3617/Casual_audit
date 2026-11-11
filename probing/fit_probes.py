"""Step 4: the differential probe, the only readout of the model's predicted change.

Q_D is an MLP (192 -> 256 -> 256 -> 4, SiLU, 40 epochs) fitted on the ENCODER's real latent
changes, Q_D(z_{t+1} - z_t) -> true change d_{t+1} - d_t, cross-fitted by episode (5 folds).
The frozen fold nets then read the PREDICTOR's change zhat_{t+1} - z_t. Q_D sees only the latent
change (never z_t), and the predictor is never used to fit it.

The normal (state-difference) and hybrid readouts, and the legacy linear readouts, were removed
on 2026-11-09.
"""
from __future__ import annotations

import numpy as np

N_FOLDS = 5
TESTABLE_R2 = 0.5          # fixed in advance: a target whose encoder-change R2 is below this is untestable
ACTION_EPS = 0.1           # newtons: step of the finite difference along d zhat / d u (cart-pole default;
                           # fit_differential uses the active environment's ACTION_EPS)
NET_SEED = 200


def r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / np.maximum(((y - y.mean(0)) ** 2).sum(0), 1e-12)


def folds(episode, k=N_FOLDS, seed=0):
    ids = np.unique(episode)
    np.random.default_rng(seed).shuffle(ids)
    return [np.isin(episode, part) for part in np.array_split(ids, k)]


def mlp_cross_fit(z, y, episode, hidden=256, epochs=40, batch=512, lr=1e-3, seed=0, eval_sets=None):
    """Out-of-fold predictions of y from z with a 2-hidden-layer MLP, and per-variable R2.

    eval_sets: optional {name: array shaped like z}. Each array's rows in a held-out
    fold are passed through that fold's frozen net (same input standardisation), so
    e.g. predictor latents are read by a probe that never saw their episode and was
    never fitted on predictor output. Then returns (pred, r2, {name: prediction}).
    Uses the GPU only if one is visible (set CUDA_VISIBLE_DEVICES=-1 to force CPU)."""
    import torch
    import torch.nn as nn
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    T = lambda v: torch.as_tensor(v, dtype=torch.float32, device=dev)  # noqa: E731
    pred = np.zeros_like(y, dtype=np.float64)
    extra = {n: np.zeros((len(a), y.shape[1])) for n, a in (eval_sets or {}).items()}
    for k, te in enumerate(folds(episode)):
        zm, zs = z[~te].mean(0), np.maximum(z[~te].std(0), 1e-6)
        ym, ys = y[~te].mean(0), np.maximum(y[~te].std(0), 1e-12)
        Xtr, Ytr = T((z[~te] - zm) / zs), T((y[~te] - ym) / ys)
        torch.manual_seed(seed + k)
        net = nn.Sequential(nn.Linear(z.shape[1], hidden), nn.SiLU(), nn.Linear(hidden, hidden),
                            nn.SiLU(), nn.Linear(hidden, y.shape[1])).to(dev)
        n = len(Xtr)
        steps = epochs * ((n + batch - 1) // batch)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        for _ in range(epochs):
            perm = torch.randperm(n, device=dev)
            for i in range(0, n, batch):
                idx = perm[i:i + batch]
                loss = ((net(Xtr[idx]) - Ytr[idx]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
        net.eval()
        with torch.no_grad():
            pred[te] = net(T((z[te] - zm) / zs)).cpu().numpy() * ys + ym
            for n, a in (eval_sets or {}).items():
                extra[n][te] = net(T((a[te] - zm) / zs)).cpu().numpy() * ys + ym
    if eval_sets is None:
        return pred, r2(y, pred)
    return pred, r2(y, pred), extra


def fit_differential(E, seed=0, shuffle_control=True, eps=None):
    """E: output of extract.extract(). Returns {'dd_pred', 'dd_enc', 'action_effect', 'report'}.

    action_effect: finite difference of Q_D along d zhat / d u for u_t, u_{t-1}, u_{t-2}, normalised
    like the expert effect sizes: mean |d change / d u| * sd(u) / sd(change)."""
    import envs
    env = envs.current()
    eps = env.ACTION_EPS if eps is None else eps
    z, zh, zn, ep = E['z'], E['zhat_next'], E['z_next'], E['episode']
    dd_true, U = E['targets'], E['inputs'][:, env.ACTION_COLS]
    g = E['dzhat_du']                                   # (n, d, 3): d zhat / d u per newton
    dz_enc, dz_pred = zn - z, zh - z
    ev_in = {'pred': dz_pred}
    for a in range(g.shape[-1]):
        ev_in[f'a{a}+'] = dz_pred + eps * g[:, :, a]
        ev_in[f'a{a}-'] = dz_pred - eps * g[:, :, a]
    dd_enc, _, ev = mlp_cross_fit(dz_enc, dd_true, ep, seed=seed * 1000 + NET_SEED, eval_sets=ev_in)
    dd_pred = ev['pred']

    sd_u, sd_t = U.std(0), np.maximum(dd_pred.std(0), 1e-12)
    effect = np.zeros((U.shape[1], dd_true.shape[1]))
    for a in range(U.shape[1]):
        effect[a] = np.abs((ev[f'a{a}+'] - ev[f'a{a}-']) / (2 * eps)).mean(0) * sd_u[a] / sd_t

    rep = {'readout': 'differential', 'seed': seed}
    if shuffle_control:
        perm = np.random.default_rng(seed + 1).permutation(len(z))
        rep['R2_shuffled_label_control'] = mlp_cross_fit(
            dz_enc, dd_true[perm], ep, seed=seed * 1000 + NET_SEED + 50)[1].tolist()
    r_enc, r_pred = r2(dd_true, dd_enc), r2(dd_true, dd_pred)
    rep['R2_change_encoder_vs_true'] = r_enc.tolist()
    rep['R2_change_predictor_vs_true'] = r_pred.tolist()
    rep['R2_gap_encoder_minus_predictor'] = (r_enc - r_pred).tolist()
    rep['R2_change_predictor_vs_encoder'] = r2(dd_enc, dd_pred).tolist()
    rep['testable_threshold'] = TESTABLE_R2
    rep[testable_key()] = [bool(v >= TESTABLE_R2) for v in r_enc]
    if 'physics' in E and np.ptp(E['physics'], axis=0).max() > 0:
        rep['R2_by_physics_tercile'] = r2_by_physics(E['physics'], dd_true, dd_enc, dd_pred, names=env.PHYS_KEYS)
    return {'dd_pred': dd_pred, 'dd_enc': dd_enc, 'report': rep, 'action_effect': effect}


def testable_key():
    """Report key, e.g. 'testable(dx,dvbar,dtheta,dwbar)' for cart-pole."""
    import envs
    return 'testable(' + ','.join(envs.current().TARGET_NAMES) + ')'


def r2_by_physics(phys, dd_true, dd_enc, dd_pred, names=('M', 'm', 'l', 'g')):
    """Change R2 (encoder, predictor) within the low / mid / high third of each physics parameter.
    Shows whether the readout still works for physics the JEPA was not trained on."""
    out = {}
    for i, n in enumerate(names):
        edges = np.quantile(phys[:, i], [0, 1 / 3, 2 / 3, 1])
        out[n] = {}
        for b, lab in enumerate(('low', 'mid', 'high')):
            sel = (phys[:, i] >= edges[b]) & ((phys[:, i] <= edges[b + 1]) if b == 2 else (phys[:, i] < edges[b + 1]))
            out[n][lab] = {'range': [float(edges[b]), float(edges[b + 1])],
                           'R2_encoder': r2(dd_true[sel], dd_enc[sel]).tolist(),
                           'R2_predictor': r2(dd_true[sel], dd_pred[sel]).tolist()}
    return out
