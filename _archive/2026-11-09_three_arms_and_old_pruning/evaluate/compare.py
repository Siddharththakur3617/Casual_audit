"""Step 7: compare learned graphs with the expert graph and with each other."""
from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from cc_common import INPUT_NAMES, RESULTS, TARGET_NAMES


def load_expert():
    e = json.loads((RESULTS / 'expert_graph_final.json').read_text())
    return np.array(e['adjacency']), np.array(e['tiers'], dtype=object), np.array(e['effect_size'])


def _prf(y, p):
    tp, fp, fn = int((y & p).sum()), int((~y & p).sum()), int((y & ~p).sum())
    prec = tp / (tp + fp) if tp + fp else float('nan')
    rec = tp / (tp + fn) if tp + fn else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return dict(precision=prec, recall=rec, f1=f1, shd=fp + fn, false_pos=fp, false_neg=fn)


def _auc(y, w):
    if y.all() or (~y).all():
        return dict(auroc=float('nan'), auprc=float('nan'))
    return dict(auroc=float(roc_auc_score(y, w)), auprc=float(average_precision_score(y, w)))


def vs_expert(kept, weight, A, tiers):
    kept, A = np.asarray(kept).astype(bool), np.asarray(A).astype(bool)
    w = np.asarray(weight)
    strong_or_absent = (tiers == 'strong') | (tiers == 'absent')
    ys = (tiers == 'strong')[strong_or_absent]
    weak = tiers == 'weak'
    return {
        'strong_graph': {**_prf(ys, kept[strong_or_absent]), **_auc(ys, w[strong_or_absent]),
                         'n_pos': int(ys.sum()), 'n_neg': int((~ys).sum())},
        'full_graph': {**_prf(A.ravel(), kept.ravel()), **_auc(A.ravel(), w.ravel()),
                       'n_pos': int(A.sum()), 'n_neg': int((~A).sum())},
        'weak_edges_recovered': f'{int(kept[weak].sum())}/{int(weak.sum())}',
        'absent_edges_wrongly_kept': [f'{INPUT_NAMES[i]}->{TARGET_NAMES[j]}'
                                      for i, j in zip(*np.where(kept & (tiers == 'absent')))],
        'strong_edges_missed': [f'{INPUT_NAMES[i]}->{TARGET_NAMES[j]}'
                                for i, j in zip(*np.where(~kept & (tiers == 'strong')))],
    }


def vs_expert_testable(kept, weight, A, tiers, testable):
    """Same as vs_expert's strong-graph scores, but only over edges into testable targets.
    testable: (n_targets,) bool. Edges into untestable targets are dropped, not counted as misses."""
    cols = np.asarray(testable, dtype=bool)
    kept, w = np.asarray(kept).astype(bool), np.asarray(weight)
    sel = ((tiers == 'strong') | (tiers == 'absent')) & cols[None, :]
    ys = (tiers == 'strong')[sel]
    res = {'testable_targets': [t for t, c in zip(TARGET_NAMES, cols) if c],
           'n_pos': int(ys.sum()), 'n_neg': int((~ys).sum())}
    if not sel.any():
        return {**res, 'f1': float('nan'), 'auroc': float('nan')}
    return {**res, **_prf(ys, kept[sel]), **_auc(ys, w[sel])}


def emergence(epochs, kept_series):
    """kept_series: (n_epochs, n_in, n_tg) 0/1. Emergence epoch of an edge = the first checkpoint
    from which it is kept at every later checkpoint; None if it is not kept at the last one."""
    K = np.asarray(kept_series).astype(bool)
    out = np.full(K.shape[1:], None, dtype=object)
    for i, j in np.ndindex(*K.shape[1:]):
        col = K[:, i, j]
        if not col[-1]:
            continue
        k = len(col) - 1
        while k > 0 and col[k - 1]:
            k -= 1
        out[i, j] = int(epochs[k])
    return out


def arm_summary(res_by_arm):
    """Markdown side-by-side table of the readout arms for one checkpoint."""
    arms = list(res_by_arm)
    rows = ['| metric | ' + ' | '.join(arms) + ' |', '|---|' + '---|' * len(arms)]

    def row(name, f):
        cells = []
        for a in arms:
            try:
                v = f(res_by_arm[a])
            except (KeyError, TypeError):
                v = '–'
            cells.append(f'{v:.3f}' if isinstance(v, float) else str(v))
        rows.append(f'| {name} | ' + ' | '.join(cells) + ' |')

    for g in ('predictor', 'encoder'):
        row(f'{g} strong F1', lambda r, g=g: r['vs_expert'][g]['strong_graph']['f1'])
        row(f'{g} strong AUROC', lambda r, g=g: r['vs_expert'][g]['strong_graph']['auroc'])
        row(f'{g} full AUROC', lambda r, g=g: r['vs_expert'][g]['full_graph']['auroc'])
        row(f'{g} kept / false pos.', lambda r, g=g: f"{int(np.sum(r['graphs'][g]['kept']))} / "
                                                     f"{len(r['vs_expert'][g]['absent_edges_wrongly_kept'])}")
        row(f'{g} strong F1 (testable only)', lambda r, g=g: r['vs_expert_testable'][g]['f1'])
    row('Spearman data vs predictor', lambda r: r['attribution']['data_vs_predictor']['weight_spearman'])
    row('Spearman encoder vs predictor', lambda r: r['attribution']['encoder_vs_predictor']['weight_spearman'])
    for j, t in enumerate(TARGET_NAMES):
        row(f'change R² {t} (enc / pred)',
            lambda r, j=j: f"{r['probe']['R2_change_encoder_vs_true'][j]:.2f} / "
                           f"{r['probe']['R2_change_predictor_vs_true'][j]:.2f}")
    row('missed strong edges (predictor)', lambda r: ', '.join(r['vs_expert']['predictor']['strong_edges_missed']) or '–')
    return '\n'.join(rows) + '\n'


def between(g1, g2):
    """Disagreement between two learned graphs."""
    k1, k2 = np.asarray(g1['kept']).astype(bool), np.asarray(g2['kept']).astype(bool)
    from scipy.stats import spearmanr
    return {
        'shd': int((k1 != k2).sum()),
        'only_in_first': [f'{INPUT_NAMES[i]}->{TARGET_NAMES[j]}' for i, j in zip(*np.where(k1 & ~k2))],
        'only_in_second': [f'{INPUT_NAMES[i]}->{TARGET_NAMES[j]}' for i, j in zip(*np.where(~k1 & k2))],
        'weight_spearman': float(spearmanr(np.ravel(g1['weight']), np.ravel(g2['weight'])).statistic),
    }


def edge_table(graphs, A, tiers, effect):
    """Markdown per-edge table: tier, expert effect size, and weight/stability/kept per graph."""
    names = list(graphs)
    head = '| edge | tier | expert effect | ' + ' | '.join(f'{n} weight (stab.)' for n in names) + ' |'
    rows = [head, '|' + '---|' * (3 + len(names))]
    order = sorted(((i, j) for i in range(len(INPUT_NAMES)) for j in range(len(TARGET_NAMES))),
                   key=lambda ij: ({'strong': 0, 'weak': 1, 'absent': 2}[tiers[ij]], ij[1], ij[0]))
    for i, j in order:
        cells = []
        for n in names:
            g = graphs[n]
            mark = '**' if g['kept'][i][j] else ''
            cells.append(f'{mark}{g["weight"][i][j]:.3f}{mark} ({g["stability"][i][j]:.2f})')
        rows.append(f'| {INPUT_NAMES[i]}→{TARGET_NAMES[j]} | {tiers[i, j]} | {effect[i, j]:.3f} | '
                    + ' | '.join(cells) + ' |')
    return '\n'.join(rows) + '\n\nBold = kept after pruning. Weight = LOCO log(held-out MSE without the input / with all inputs).'


def heatmaps(graphs, A, tiers, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['expert'] + list(graphs)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.6), constrained_layout=True)
    for ax, n in zip(axes, names):
        if n == 'expert':
            M = np.vectorize({'strong': 1.0, 'weak': 0.4, 'absent': 0.0}.get)(tiers).astype(float)
            title = 'expert (strong / weak)'
        else:
            M = np.asarray(graphs[n]['weight'])
            M = M / max(M.max(), 1e-12)
            title = f'{n} (weight, kept = dot)'
        ax.imshow(M, cmap='Blues', vmin=0, vmax=1)
        if n != 'expert':
            ki, kj = np.where(np.asarray(graphs[n]['kept']).astype(bool))
            ax.scatter(kj, ki, s=14, c='#d1495b')
        ax.set_xticks(range(len(TARGET_NAMES)), TARGET_NAMES, rotation=45)
        ax.set_yticks(range(len(INPUT_NAMES)), INPUT_NAMES)
        ax.set_title(title, fontsize=9)
    fig.savefig(path, dpi=110)
    plt.close(fig)
