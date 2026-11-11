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
  null_cal    min_effect with a calibrated threshold (specified 2026-11-10, before any result).
              Statistic per edge: median over splits of (Dvs - DELTA) / se, i.e. the min_effect test
              statistic before it is turned into a p-value. The null: the same 50-split scoring
              repeated on K row-shuffled copies of the targets (discovery/null_calibration.py), so
              every edge has K null statistics. Exact permutation p-value
              p = (1 + #{null >= real}) / (K + 1); kept if p <= 0.05 (K = 20: must beat every null).
              Uses no expert information. (Phipson & Smyth 2010 for the exact p-value.)
  null_cal_fwer  Same statistic, but each real edge is compared with the LARGEST statistic over all
              28 edges in each null repeat (max-T; Westfall & Young 1993), so the whole graph, not
              each edge, has a 5% false-positive rate. Declared together with null_cal.
  null_cal_vif_fwer  null_cal_fwer with the minimum effect in the expert's unit instead of in unique
              variance (specified 2026-11-10 AFTER the null_cal results showed which strong edges
              are missed, so it must be judged on the validation graphs and on datasets/cc, not on
              the model graphs). LOCO removes only an input's unique part: for a (locally linear)
              effect b, D ~ b^2 * (1 - R2_i), where R2_i is how well the other inputs predict input
              i. DELTA = 0.01 = 0.1^2 is the expert's strong cut-off only when R2_i = 0, so here
              delta_i = DELTA * (1 - R2_i), with R2_i the in-sample linear R2 of input i on the other
              six inputs (floored at 1e-3). The null statistics use the same delta_i.
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

NULL_ALPHA = 0.05
VIF_FLOOR = 1e-3

RULES = ('margin', 'fdr', 'min_effect', 'effect', 'effect_fdr', 'knockoff', 'null_cal', 'null_cal_fwer',
         'null_cal_vif_fwer')
NEEDS_KNOCKOFF = ('knockoff',)
NEEDS_NULL = ('null_cal', 'null_cal_fwer')
NEEDS_NULL_VIF = ('null_cal_vif_fwer',)


def input_collinearity(X):
    """R2_i: in-sample linear R2 of each input column on all the others."""
    X = np.asarray(X, float)
    r2 = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        A = np.column_stack([np.delete(X, i, axis=1), np.ones(len(X))])
        coef, *_ = np.linalg.lstsq(A, X[:, i], rcond=None)
        r2[i] = 1 - np.var(X[:, i] - A @ coef) / np.var(X[:, i])
    return r2


def vif_delta(X, delta=DELTA):
    """Per-input minimum effect delta_i = delta * (1 - R2_i), shape (n_in, 1) (broadcasts over targets)."""
    return (delta * np.maximum(1 - input_collinearity(X), VIF_FLOOR))[:, None]


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


def null_statistic(Dvs, Dvs_se=None, delta=DELTA):
    """Calibration statistic per edge: median over splits of (Dvs - delta) / se. Takes either a
    score_graph output or the (splits, n_in, n_tg) arrays Dvs, Dvs_se (inputs only)."""
    if Dvs_se is None:
        n = _n_in(Dvs)
        Dvs, Dvs_se = Dvs['per_split']['Dvs'][:, :n], Dvs['per_split']['Dvs_se'][:, :n]
    z = (np.asarray(Dvs) - delta) / np.maximum(np.asarray(Dvs_se), 1e-12)
    return np.median(z, axis=0)                              # (n_in, n_tg)


def _null_array(null_stats):
    N = np.asarray(null_stats)
    assert N.ndim == 3 and len(N) >= 1, 'null_stats must be (K, n_in, n_tg); run discovery/null_calibration.py'
    return N


def null_pvalues(s, null_stats, fwer=False, delta=DELTA):
    """Exact permutation p-values of the real statistic against the K null statistics (which must
    have been computed with the same delta)."""
    N, real = _null_array(null_stats), null_statistic(s, delta=delta)
    if fwer:
        N = N.max(axis=(1, 2))[:, None, None]                # per null repeat: the largest of all edges
    return (1 + (N >= real[None]).sum(0)) / (len(N) + 1)


def rule_null_cal(s, null_stats, alpha=NULL_ALPHA, fwer=False, delta=DELTA):
    return (null_pvalues(s, null_stats, fwer, delta) <= alpha).astype(int)


def prune(scores, rule='margin', knockoff_scores=None, null_stats=None, null_vif=None):
    """scores: score_graph(decoys='shuffled') output. The knockoff rule reads knockoff_scores;
    the null_cal rules read null_stats (K, n_in, n_tg) from discovery/null_calibration.py;
    null_cal_vif_fwer reads null_vif = (null stats computed with delta_i, delta_i) from
    null_calibration.load(..., delta=vif_delta(X))."""
    if rule in NEEDS_NULL_VIF:
        assert null_vif is not None, f'rule {rule} needs null_vif (null stats with delta_i, delta_i)'
        stats, delta = null_vif
        return rule_null_cal(scores, stats, fwer=True, delta=delta)
    if rule in NEEDS_NULL:
        assert null_stats is not None, f'rule {rule} needs null_stats (discovery/null_calibration.py)'
        return rule_null_cal(scores, null_stats, fwer=(rule == 'null_cal_fwer'))
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
