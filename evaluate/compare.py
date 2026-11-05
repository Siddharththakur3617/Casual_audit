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
    return '\n'.join(rows) + '\n\nBold = kept after pruning. Weight = drop in held-out R² when the input is permuted.'


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
