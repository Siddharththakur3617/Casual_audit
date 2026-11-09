"""Step 4b: linear probes from a single latent z_t (no history) to the state.

The probe P(z) = W z + b maps z_t -> d_t = (x, vbar, theta, wbar). It is
cross-fitted by episode (5 folds): each transition is read with a probe that
never saw its episode. Because P is linear,
  model change   P(zhat_{t+1}) - P(z_t) = W (zhat_{t+1} - z_t)
  encoder change P(z_{t+1})    - P(z_t) = W (z_{t+1}   - z_t)
use the same W as the state read-out ("state" readout).

Delta readout: a second linear probe Q is fitted on the encoder's real latent
changes, Q(z_{t+1} - z_t) -> true change d_{t+1} - d_t, and then applied unchanged to
the predictor's change zhat_{t+1} - z_t. It sees only the latent change (never z_t),
so it cannot inject state information that the change does not carry. The predictor
is never used to fit Q.

MLP probe (diagnostic only): a small nonlinear probe z_t -> d_t. If it beats the
linear probe by a wide margin, the state is in the latent but not linearly readable.

fit_all / Ridge / 'state' / 'delta' are the LEGACY linear readouts, kept unchanged so
old results stay reproducible. The final study uses fit_arms (bottom of this file):
three separately trained MLP readout arms, normal, differential and hybrid.
"""
from __future__ import annotations

import numpy as np

N_FOLDS = 5
READOUTS = ('state', 'delta')


class Ridge:
    def __init__(self, ridge=1e-3):
        self.ridge = ridge

    def fit(self, z, y):
        self.zm, self.zs = z.mean(0), np.maximum(z.std(0), 1e-6)
        self.ym = y.mean(0)
        x = (z - self.zm) / self.zs
        # ridge is relative to x.T x / n (inputs are standardised)
        self.W = np.linalg.solve(x.T @ x + self.ridge * len(x) * np.eye(x.shape[1]),
                                 x.T @ (y - self.ym))
        return self

    def __call__(self, z):
        return ((z - self.zm) / self.zs) @ self.W + self.ym

    def linear(self, dz):
        """Apply to a latent difference (no offset)."""
        return (dz / self.zs) @ self.W


def r2(y, p):
    return 1 - ((y - p) ** 2).sum(0) / np.maximum(((y - y.mean(0)) ** 2).sum(0), 1e-12)


def folds(episode, k=N_FOLDS, seed=0):
    ids = np.unique(episode)
    np.random.default_rng(seed).shuffle(ids)
    return [np.isin(episode, part) for part in np.array_split(ids, k)]


def cross_fit(z, y, episode):
    """Out-of-fold predictions of y from z, and per-variable R2."""
    pred = np.zeros_like(y, dtype=np.float64)
    for te in folds(episode):
        pred[te] = Ridge().fit(z[~te], y[~te])(z[te])
    return pred, r2(y, pred)


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


def fit_all(E, gate=0.8, readout='state', mlp=True):
    """E: output of extract.extract(). Returns probed variables and the probe report.
    readout: 'state' -> changes read with the state probe W (original method);
             'delta' -> changes read with the delta probe Q fitted on encoder changes."""
    assert readout in READOUTS, readout
    z, zh, zn, ep = E['z'], E['zhat_next'], E['z_next'], E['episode']
    d_true, dd_true = E['inputs'][:, :4], E['targets']
    d_hat = np.zeros_like(d_true)
    dd = {k: np.zeros_like(d_true) for k in ('pred_state', 'enc_state', 'pred_delta', 'enc_delta')}
    for te in folds(ep):
        P = Ridge().fit(z[~te], d_true[~te])
        d_hat[te] = P(z[te])
        dd['pred_state'][te] = P.linear(zh[te] - z[te])
        dd['enc_state'][te] = P.linear(zn[te] - z[te])
        Q = Ridge().fit(zn[~te] - z[~te], dd_true[~te])
        dd['pred_delta'][te] = Q(zh[te] - z[te])
        dd['enc_delta'][te] = Q(zn[te] - z[te])
    rep = {'readout': readout, 'R2_state(x,vbar,theta,wbar)': r2(d_true, d_hat).tolist()}
    inst = np.column_stack([E['inputs'][:, 0], E['xdot'], E['inputs'][:, 2], E['thetadot']])
    rep['R2_instantaneous(x,xdot,theta,thetadot)'] = cross_fit(z, inst, ep)[1].tolist()
    rep['R2_change_predictor_vs_true'] = r2(dd_true, dd['pred_state']).tolist()
    rep['R2_change_encoder_vs_true'] = r2(dd_true, dd['enc_state']).tolist()
    rep['R2_change_delta_probe_predictor_vs_true'] = r2(dd_true, dd['pred_delta']).tolist()
    rep['R2_change_delta_probe_encoder_vs_true'] = r2(dd_true, dd['enc_delta']).tolist()
    if mlp:
        rep['R2_state_mlp(x,vbar,theta,wbar)'] = mlp_cross_fit(z, d_true, ep)[1].tolist()
    perm = np.random.default_rng(1).permutation(len(z))
    rep['R2_shuffled_label_control'] = cross_fit(z, d_true[perm], ep)[1].tolist()
    rep['gate'] = gate
    rep['passes_gate'] = [bool(v >= gate) for v in rep['R2_state(x,vbar,theta,wbar)']]
    return {'d_hat': d_hat, 'dd_pred': dd[f'pred_{readout}'], 'dd_enc': dd[f'enc_{readout}']}, rep


# ---------------------------------------------------------------------------
# Three readout arms (final plan). Every arm trains its own MLP probe(s), only on
# encoder latents, cross-fitted by episode with the same folds for every arm, and
# reads the predictor's output with the frozen nets.
#   normal       S_N: z -> d                 dd = S_N(zhat_{t+1}) - S_N(z_t)
#   differential Q_D: dz -> dd               dd = Q_D(zhat_{t+1} - z_t)
#   hybrid       H_L: z -> (vbar, wbar)      dx = DT * vbar(zhat_{t+1}), dtheta = DT * wbar(zhat_{t+1})
#                H_D: dz -> (dvbar, dwbar)   dvbar, dwbar = H_D(zhat_{t+1} - z_t)
# The hybrid's kinematic rows are exact: vbar_{t+1} = (x_{t+1} - x_t) / DT, so dx = DT * vbar_{t+1}.
# ---------------------------------------------------------------------------
ARMS = ('normal', 'differential', 'hybrid')
TESTABLE_R2 = 0.5          # fixed in advance: a target whose encoder-change R2 is below this is untestable
ACTION_EPS = 0.1           # newtons: step of the finite difference along d zhat / d u
_NET_SEED = {'normal': 100, 'differential': 200, 'hybrid_level': 300, 'hybrid_delta': 400}


def _perturbed(base, g, eps):
    """base +- eps * g[..., lag] for each action lag, keyed 'a{lag}+' / 'a{lag}-'."""
    out = {}
    for a in range(g.shape[-1]):
        out[f'a{a}+'] = base + eps * g[:, :, a]
        out[f'a{a}-'] = base - eps * g[:, :, a]
    return out


def _action_effect(plus_minus, dd_pred, U, eps):
    """plus_minus(lag) -> (change at +eps, change at -eps). Normalised like the expert effect sizes:
    mean |d change / d u| * sd(u) / sd(change)."""
    sd_u, sd_t = U.std(0), np.maximum(dd_pred.std(0), 1e-12)
    eff = np.zeros((U.shape[1], dd_pred.shape[1]))
    for a in range(U.shape[1]):
        hi, lo = plus_minus(a)
        eff[a] = np.abs((hi - lo) / (2 * eps)).mean(0) * sd_u[a] / sd_t
    return eff


def fit_arms(E, arms=ARMS, seed=0, shuffle_control=True, eps=ACTION_EPS):
    """E: output of extract.extract(). Returns {arm: {'dd_pred', 'dd_enc', 'action_effect', 'report'}}."""
    from cc_common import DT
    z, zh, zn, ep = E['z'], E['zhat_next'], E['z_next'], E['episode']
    d_true, dd_true, U = E['inputs'][:, :4], E['targets'], E['inputs'][:, 4:7]
    g = E['dzhat_du']                                   # (n, d, 3): d zhat / d u per newton
    dz_enc, dz_pred = zn - z, zh - z
    perm = np.random.default_rng(seed + 1).permutation(len(z))
    fit = lambda x, y, net, ev: mlp_cross_fit(x, y, ep, seed=seed * 1000 + _NET_SEED[net], eval_sets=ev)  # noqa: E731
    shuf = lambda x, y, net: (mlp_cross_fit(x, y[perm], ep, seed=seed * 1000 + _NET_SEED[net] + 50)[1].tolist()  # noqa: E731
                              if shuffle_control else None)
    out = {}
    for arm in arms:
        rep = {'arm': arm, 'seed': seed}
        if arm == 'normal':
            S, r2_lvl, ev = fit(z, d_true, 'normal', {'pred': zh, 'enc': zn, **_perturbed(zh, g, eps)})
            dd_pred, dd_enc = ev['pred'] - S, ev['enc'] - S
            pm = lambda a, ev=ev: (ev[f'a{a}+'], ev[f'a{a}-'])  # noqa: E731
            rep['R2_level(x,vbar,theta,wbar)'] = r2_lvl.tolist()
            rep['R2_shuffled_label_control'] = {'S_N': shuf(z, d_true, 'normal')}
        elif arm == 'differential':
            dd_enc, r2_enc, ev = fit(dz_enc, dd_true, 'differential', {'pred': dz_pred, **_perturbed(dz_pred, g, eps)})
            dd_pred = ev['pred']
            pm = lambda a, ev=ev: (ev[f'a{a}+'], ev[f'a{a}-'])  # noqa: E731
            rep['R2_shuffled_label_control'] = {'Q_D': shuf(dz_enc, dd_true, 'differential')}
        elif arm == 'hybrid':
            L, r2_lvl, evL = fit(z, d_true[:, [1, 3]], 'hybrid_level', {'pred': zh, 'enc': zn, **_perturbed(zh, g, eps)})
            D, r2_d, evD = fit(dz_enc, dd_true[:, [1, 3]], 'hybrid_delta',
                               {'pred': dz_pred, **_perturbed(dz_pred, g, eps)})
            asm = lambda lv, dv: np.column_stack([DT * lv[:, 0], dv[:, 0], DT * lv[:, 1], dv[:, 1]])  # noqa: E731
            dd_pred, dd_enc = asm(evL['pred'], evD['pred']), asm(evL['enc'], D)
            pm = lambda a, evL=evL, evD=evD: (asm(evL[f'a{a}+'], evD[f'a{a}+']),  # noqa: E731
                                              asm(evL[f'a{a}-'], evD[f'a{a}-']))
            rep['R2_level(vbar,wbar)'] = r2_lvl.tolist()
            rep['R2_shuffled_label_control'] = {'H_L': shuf(z, d_true[:, [1, 3]], 'hybrid_level'),
                                                'H_D': shuf(dz_enc, dd_true[:, [1, 3]], 'hybrid_delta')}
        else:
            raise ValueError(arm)
        r_enc, r_pred = r2(dd_true, dd_enc), r2(dd_true, dd_pred)
        rep['R2_change_encoder_vs_true'] = r_enc.tolist()
        rep['R2_change_predictor_vs_true'] = r_pred.tolist()
        rep['R2_gap_encoder_minus_predictor'] = (r_enc - r_pred).tolist()
        rep['R2_change_predictor_vs_encoder'] = r2(dd_enc, dd_pred).tolist()
        rep['testable_threshold'] = TESTABLE_R2
        rep['testable(dx,dvbar,dtheta,dwbar)'] = [bool(v >= TESTABLE_R2) for v in r_enc]
        out[arm] = {'dd_pred': dd_pred, 'dd_enc': dd_enc, 'report': rep,
                    'action_effect': _action_effect(pm, dd_pred, U, eps)}
    return out
