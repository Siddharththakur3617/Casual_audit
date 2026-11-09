"""Step 6: pruning rules. Every rule is blind to the expert graph and uses only the per-split
scores from edge_scores.score_graph. Constants are fixed in advance and are not tuned.

  margin      ORIGINAL rule. Kept if, in >= 70% of half-splits, the LOCO weight beats the strongest
              shuffled decoy by log 1.5 (stability selection with a decoy threshold;
              Meinshausen & Buhlmann 2010, Barber & Candes 2015).
  fdr         LOCO test against the decoys. Per split, a one-sided z-test of
              D(edge) - D(strongest decoy) > 0 with an episode-clustered standard error
              (Lei et al. 2018; Williamson et al. 2023). The 50 split p-values are combined
              by quantile aggregation, p = min(1, 2 * median) (Meinshausen, Meier & Buhlmann 2009),
              then Benjamini-Hochberg at q = 0.05 over all 28 edges.
  min_effect  Same test, but against a minimum relevant effect: H0 is "the edge explains at most
              1% of the target's variance more than a decoy does" (DELTA = 0.01).
  effect      Effect size in the expert graph's own unit. Normalised Jacobian of the full fit,
              mean |df/dx_i| * sd(x_i) / sd(y), must reach 0.1 (the expert 'strong' cut-off,
              checks/verify_equations.py check e) and beat every decoy's, in >= 70% of splits.
  effect_fdr  Both of the above: the edge must carry held-out information beyond the decoys (fdr)
              AND be material in the expert's unit (effect). Specified after the 2-split smoke test
              showed that 'effect' alone keeps edges on the shuffled-target null (an overfit noise
              fit has large Jacobians), before any 50-split result was seen.
  knockoff    Derandomised model-X knockoffs (Candes et al. 2018; Ren, Wei & Candes 2023).
              Per split, W = LOCO(input) - LOCO(its knockoff); knockoff+ at q = 0.1 pooled over
              the 28 edges; kept if selected in >= 50% of splits. Needs decoys='knockoff' scores.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

STABILITY_MIN = 0.7
FDR_Q = 0.05
DELTA = 0.01
EFFECT_MIN = 0.1
KNOCKOFF_Q = 0.1
KNOCKOFF_FREQ = 0.5

RULES = ('margin', 'fdr', 'min_effect', 'effect', 'effect_fdr', 'knockoff')
NEEDS_KNOCKOFF = ('knockoff',)


def _n_in(scores):
    return np.asarray(scores['weight']).shape[0]


def rule_margin(s, stability_min=STABILITY_MIN):
    return (np.asarray(s['stability']) >= stability_min).astype(int)


def _split_pvalues(s, delta):
    P = s['per_split']
    n = _n_in(s)
    z = (np.asarray(P['Dvs'])[:, :n] - delta) / np.maximum(np.asarray(P['Dvs_se'])[:, :n], 1e-12)
    return norm.sf(z)                                        # (splits, n_in, n_tg)


def aggregated_pvalues(s, delta=0.0):
    """Quantile aggregation (gamma = 0.5) of the per-split p-values."""
    return np.minimum(1.0, 2 * np.median(_split_pvalues(s, delta), axis=0))


def benjamini_hochberg(p, q):
    flat = np.ravel(p)
    m = len(flat)
    order = np.argsort(flat)
    ok = flat[order] <= q * np.arange(1, m + 1) / m
    keep = np.zeros(m, bool)
    if ok.any():
        keep[order[:np.max(np.where(ok)[0]) + 1]] = True
    return keep.reshape(np.shape(p))


def rule_fdr(s, q=FDR_Q):
    return benjamini_hochberg(aggregated_pvalues(s, 0.0), q).astype(int)


def rule_min_effect(s, q=FDR_Q, delta=DELTA):
    return benjamini_hochberg(aggregated_pvalues(s, delta), q).astype(int)


def rule_effect(s, effect_min=EFFECT_MIN, stability_min=STABILITY_MIN):
    J = np.asarray(s['per_split']['J'])
    n = _n_in(s)
    bar = np.maximum(effect_min, J[:, n:, :].max(1))         # (splits, n_tg)
    return ((J[:, :n, :] >= bar[:, None, :]).mean(0) >= stability_min).astype(int)


def knockoff_plus_threshold(W, q):
    """Smallest t > 0 with (1 + #{W <= -t}) / max(1, #{W >= t}) <= q (Barber & Candes 2015)."""
    for t in np.sort(np.abs(W[W != 0])):
        if (1 + (W <= -t).sum()) / max(1, (W >= t).sum()) <= q:
            return t
    return np.inf


def knockoff_selection_frequency(s, q=KNOCKOFF_Q):
    assert s.get('decoys') == 'knockoff', 'the knockoff rule needs decoys="knockoff" scores'
    L = np.asarray(s['per_split']['L'])
    n = _n_in(s)
    W = L[:, :n, :] - L[:, n:2 * n, :]                       # (splits, n_in, n_tg)
    sel = np.zeros_like(W, dtype=bool)
    for k in range(len(W)):
        sel[k] = W[k] >= knockoff_plus_threshold(W[k].ravel(), q)
    return sel.mean(0)


def rule_knockoff(s, q=KNOCKOFF_Q, freq=KNOCKOFF_FREQ):
    return (knockoff_selection_frequency(s, q) >= freq).astype(int)


def prune(scores, rule='margin', knockoff_scores=None):
    """scores: score_graph(decoys='shuffled') output. The knockoff rule reads knockoff_scores."""
    if rule == 'margin':
        return rule_margin(scores)
    if rule == 'fdr':
        return rule_fdr(scores)
    if rule == 'min_effect':
        return rule_min_effect(scores)
    if rule == 'effect':
        return rule_effect(scores)
    if rule == 'effect_fdr':
        return rule_effect(scores) & rule_fdr(scores)
    if rule == 'knockoff':
        return rule_knockoff(knockoff_scores if knockoff_scores is not None else scores)
    raise ValueError(rule)
