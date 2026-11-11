"""Is the hidden per-episode physics confounding the true-data graph on a varied-physics dataset?

The expert graph holds physics fixed (Jacobian at each row's own M, m, l, g); the 'data' graph of
compare_pruning.py averages over hidden physics. If inputs stand in for physics, absent edges can
look real (suspects on cc_phys: vbar->dwbar kept by most rules; wbar->dwbar fitted effect 1.17 vs
expert 0.09). Following CD-NOD (Huang et al. 2020, a domain index as a conditioning variable),
the true-data graph is scored three ways, same rows, seeds and splits:

  data          no physics (the cached compare_pruning score)
  data_phys     the episode's (M, m, l, g) given to every fit as conditioning inputs (not nodes)
  data_physshuf the same four columns permuted across episodes: control for "four extra inputs"

Declared before running: confounding explains an edge if its statistic drops below the
null_cal_fwer threshold with true physics but not with shuffled physics. The null calibration
was computed without conditioning columns, so the rule columns for data_phys are approximate;
the effect sizes (Jacobian, D) are the main readout.

  python diagnose_physics.py --data datasets/cc_phys --run-dir runs/jepa_cc_phys
Output: results/physics_conditioning_<data>/{report.md, scores/*.json}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cc_common import INPUT_NAMES, PHYS_KEYS, RESULTS, TARGET_NAMES, load_split_episodes, stack_episodes
from compare_pruning import load_scores
from discovery import edge_scores, null_calibration, prune as pr
from evaluate import compare
from run_all import jsonable


def physics_rows(data_dir):
    eps = load_split_episodes(data_dir, 'train', with_physics=True)
    V = stack_episodes([(s, a) for s, a, _ in eps])
    P = np.array([[ph[k] for k in PHYS_KEYS] for _, _, ph in eps])     # (episodes, 4)
    return V, P


def scored(out, name, X, Y, ep, n_splits, cond):
    path = out / 'scores' / f'{name}_shuffled.json'
    if not path.exists():
        s = edge_scores.score_graph(X, Y, ep, n_splits=n_splits, cond=cond)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(jsonable(s)))
        print(f'  scored {name}: held-out R2 {np.round(s["heldout_R2"], 3)}', flush=True)
    return load_scores(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--run-dir', required=True, help='only to find the cached compare_pruning data score')
    p.add_argument('--splits', type=int, default=50)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    name = Path(a.data).name
    out = Path(a.out) if a.out else RESULTS / f'physics_conditioning_{name}'
    out.mkdir(parents=True, exist_ok=True)
    A, tiers, effect = compare.load_expert(a.data)

    V, P = physics_rows(a.data)
    X, Y, ep = V['inputs'], V['targets'], V['episode']
    assert P.std(0).max() > 0, 'dataset has fixed physics: nothing to condition on'
    perm = np.random.default_rng(4321).permutation(len(P))           # shuffle physics across episodes
    cached = RESULTS / f'pruning_comparison_{name}_{Path(a.run_dir).name}' / 'scores' / 'data_shuffled.json'
    S = {'data': load_scores(cached) if (cached.exists() and a.splits == 50 and not a.out)
         else scored(out, 'data', X, Y, ep, a.splits, None),
         'data_phys': scored(out, 'data_phys', X, Y, ep, a.splits, P[ep]),
         'data_physshuf': scored(out, 'data_physshuf', X, Y, ep, a.splits, P[perm][ep])}

    null_stats, _ = null_calibration.load(a.data, a.splits)
    d_vif = pr.vif_delta(X)
    null_vif = (null_calibration.load(a.data, a.splits, delta=d_vif)[0], d_vif)
    thr = null_stats.max(axis=(1, 2))
    rules = ('min_effect', 'null_cal_fwer', 'null_cal_vif_fwer')
    kept = {g: {r: pr.prune(s, r, null_stats=null_stats, null_vif=null_vif) for r in rules} for g, s in S.items()}
    stat = {g: pr.null_statistic(s) for g, s in S.items()}

    G = list(S)
    L = [f'# Physics conditioning of the true-data graph ({a.data}, {a.splits} splits)', '',
         'Held-out R2 (dx, dvbar, dtheta, dwbar): ' + '; '.join(
             f'{g} {np.round(S[g]["heldout_R2"], 3).tolist()}' for g in G), '',
         f'null_cal_fwer threshold (largest null statistic per repeat): max {thr.max():.2f}, median {np.median(thr):.2f}. '
         'Statistic = median over splits of (Dvs - 0.01)/se. The null was computed without conditioning columns.', '',
         '## kept edges, strong-graph F1 / kept / false positives', '',
         '| rule | ' + ' | '.join(G) + ' |', '|---|' + '---|' * len(G)]
    for r in rules:
        cells = []
        for g in G:
            v = compare.vs_expert(kept[g][r], stat[g], A, tiers)
            cells.append(f'{v["strong_graph"]["f1"]:.2f} · {int(kept[g][r].sum())} · '
                         f'{len(v["absent_edges_wrongly_kept"])}')
        L.append(f'| {r} | ' + ' | '.join(cells) + ' |')
    L += ['', '## per edge: Jacobian effect · statistic (✔ = kept by null_cal_fwer)', '',
          '| edge | tier | expert effect | ' + ' | '.join(G) + ' |', '|' + '---|' * (3 + len(G))]
    order = sorted(((i, j) for i in range(len(INPUT_NAMES)) for j in range(len(TARGET_NAMES))),
                   key=lambda ij: ({'strong': 0, 'weak': 1, 'absent': 2}[tiers[ij]], ij[1], ij[0]))
    for i, j in order:
        cells = [f'{np.asarray(S[g]["jacobian_effect"])[i, j]:.3f} · {stat[g][i, j]:.1f}'
                 f'{" ✔" if kept[g]["null_cal_fwer"][i, j] else ""}' for g in G]
        L.append(f'| {INPUT_NAMES[i]}→{TARGET_NAMES[j]} | {tiers[i, j]} | {effect[i, j]:.3f} | ' + ' | '.join(cells) + ' |')
    (out / 'report.md').write_text('\n'.join(L) + '\n')
    print(f'-> {out / "report.md"}')


if __name__ == '__main__':
    main()
