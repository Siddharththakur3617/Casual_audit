"""Step 4b: linear probes from a single latent z_t (no history) to the state.

The probe P(z) = W z + b maps z_t -> d_t = (x, vbar, theta, wbar). It is
cross-fitted by episode (5 folds): each transition is read with a probe that
never saw its episode. Because P is linear,
  model change   P(zhat_{t+1}) - P(z_t) = W (zhat_{t+1} - z_t)
  encoder change P(z_{t+1})    - P(z_t) = W (z_{t+1}   - z_t)
use the same W as the state read-out.
"""
from __future__ import annotations

import numpy as np

N_FOLDS = 5


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


def fit_all(E, gate=0.8):
    """E: output of extract.extract(). Returns probed variables and the probe report."""
    z, zh, zn, ep = E['z'], E['zhat_next'], E['z_next'], E['episode']
    d_true = E['inputs'][:, :4]
    d_hat = np.zeros_like(d_true)
    dd_pred = np.zeros_like(d_true)
    dd_enc = np.zeros_like(d_true)
    for te in folds(ep):
        P = Ridge().fit(z[~te], d_true[~te])
        d_hat[te] = P(z[te])
        dd_pred[te] = P.linear(zh[te] - z[te])
        dd_enc[te] = P.linear(zn[te] - z[te])
    rep = {'R2_state(x,vbar,theta,wbar)': r2(d_true, d_hat).tolist()}
    inst = np.column_stack([E['inputs'][:, 0], E['xdot'], E['inputs'][:, 2], E['thetadot']])
    rep['R2_instantaneous(x,xdot,theta,thetadot)'] = cross_fit(z, inst, ep)[1].tolist()
    rep['R2_change_predictor_vs_true'] = r2(E['targets'], dd_pred).tolist()
    rep['R2_change_encoder_vs_true'] = r2(E['targets'], dd_enc).tolist()
    perm = np.random.default_rng(1).permutation(len(z))
    rep['R2_shuffled_label_control'] = cross_fit(z, d_true[perm], ep)[1].tolist()
    rep['gate'] = gate
    rep['passes_gate'] = [bool(v >= gate) for v in rep['R2_state(x,vbar,theta,wbar)']]
    return {'d_hat': d_hat, 'dd_pred': dd_pred, 'dd_enc': dd_enc}, rep
