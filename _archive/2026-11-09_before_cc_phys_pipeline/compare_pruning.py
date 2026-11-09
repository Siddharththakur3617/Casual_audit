"""Compare the pruning rules of discovery/prune.py on the same edge scores.

Method-validation graphs (ground truth known, so the rule can be judged without looking at the model):
  data          true targets, noise-free: the expert graph should come out.
  data_noisy    true targets + independent Gaussian noise that caps the achievable R2 at the level the
                epoch-200 predictor reaches (NOISE_R2). Same truth as 'data'; tests whether a rule
                still works at the model's noise floor.
  null          targets shuffled across rows: every edge is absent, a rule should keep nothing.
Model graphs (differential probe, true-state inputs): encoder and predictor at epochs 200 and 0.

Every graph is scored once with shuffled decoys and once with knockoff decoys (50 splits each); all
rules are then applied to the same scores. Scores are cached, so the script resumes after a crash.

  python compare_pruning.py --data datasets/cc --run-dir runs/jepa_cc
Output: results/pruning_comparison/{report.md, comparison.json, scores/*.json}
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from cc_common import INPUT_NAMES, RESULTS, TARGET_NAMES, load_split_episodes, stack_episodes
from discovery import edge_scores, prune as pr
from evaluate import compare
from probing import fit_probes
from run_all import jsonable, load_or_extract

OUT = RESULTS / 'pruning_comparison'
NOISE_R2 = [0.972, 0.985, 0.935, 0.955]    # epoch-200 predictor held-out R2 per target (dx, dvbar, dtheta, dwbar)
EPOCHS = (200, 0)
RULE_SCORE = {   # the continuous score each rule thresholds (for a threshold-free ranking check)
    'margin': lambda s, k: np.asarray(s['weight']),
    'fdr': lambda s, k: -np.log10(np.maximum(pr.aggregated_pvalues(s, 0.0), 1e-300)),
    'min_effect': lambda s, k: -np.log10(np.maximum(pr.aggregated_pvalues(s, pr.DELTA), 1e-300)),
    'effect': lambda s, k: np.asarray(s['jacobian_effect']),
    'effect_fdr': lambda s, k: np.asarray(s['jacobian_effect']),
    'knockoff': lambda s, k: pr.knockoff_selection_frequency(k),
}


def load_scores(path):
    s = json.loads(Path(path).read_text())
    s['per_split'] = {k: np.asarray(v) for k, v in s['per_split'].items()}
    return s


def scored(name, X, Y, ep, n_splits, decoys):
    path = OUT / 'scores' / f'{name}_{decoys}.json'
    if path.exists():
        return load_scores(path)
    t0 = time.time()
    s = edge_scores.score_graph(X, Y, ep, n_splits=n_splits, decoys=decoys)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(s)))
    print(f'  scored {name} ({decoys}): held-out R2 {np.round(s["heldout_R2"], 3)} ({time.time() - t0:.0f}s)',
          flush=True)
    return load_scores(path)


def model_targets(run_dir, data_dir, epoch):
    path = OUT / 'probe' / f'epoch{epoch:04d}.npz'
    if path.exists():
        with np.load(path) as f:
            return {k: f[k] for k in f.files}
    ck = Path(run_dir) / 'checkpoints' / f'checkpoint_epoch{epoch:04d}.pt'
    E = load_or_extract(ck, data_dir)
    P = fit_probes.fit_differential(E, seed=0, shuffle_control=False)
    print(f'  [epoch {epoch}] change R2 enc {np.round(P["report"]["R2_change_encoder_vs_true"], 3)}'
          f' pred {np.round(P["report"]["R2_change_predictor_vs_true"], 3)}', flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {'X': E['inputs'], 'episode': E['episode'], 'encoder': P['dd_enc'], 'predictor': P['dd_pred']}
    np.savez(path, **d)
    return d


def evaluate(kept, score, A, tiers):
    v = compare.vs_expert(kept, score, A, tiers)
    sg = v['strong_graph']
    return {'kept': int(np.sum(kept)), 'f1': sg['f1'], 'precision': sg['precision'], 'recall': sg['recall'],
            'false_pos': len(v['absent_edges_wrongly_kept']), 'weak': v['weak_edges_recovered'],
            'auroc_score': sg['auroc'], 'missed': v['strong_edges_missed'],
            'absent_kept': v['absent_edges_wrongly_kept']}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--splits', type=int, default=50)
    p.add_argument('--out', default=None, help='output folder (default results/pruning_comparison)')
    a = p.parse_args()
    global OUT
    OUT = Path(a.out) if a.out else OUT
    OUT.mkdir(parents=True, exist_ok=True)
    A, tiers, effect = compare.load_expert()

    V = stack_episodes(load_split_episodes(a.data, 'train'))
    X, Y, ep = V['inputs'], V['targets'], V['episode']
    rng = np.random.default_rng(12345)
    sd = Y.std(0) * np.sqrt((1 - np.array(NOISE_R2)) / np.array(NOISE_R2))
    graphs = {'data': (X, Y, ep),
              'data_noisy': (X, Y + rng.standard_normal(Y.shape) * sd, ep),
              'null': (X, Y[rng.permutation(len(Y))], ep)}
    for e in EPOCHS:
        M = model_targets(a.run_dir, a.data, e)
        graphs[f'predictor_e{e}'] = (M['X'], M['predictor'], M['episode'])
        graphs[f'encoder_e{e}'] = (M['X'], M['encoder'], M['episode'])

    res, kept_all = {}, {}
    for g, (Xg, Yg, eg) in graphs.items():
        print(f'[{g}]', flush=True)
        s = scored(g, Xg, Yg, eg, a.splits, 'shuffled')
        k = scored(g, Xg, Yg, eg, a.splits, 'knockoff')
        res[g], kept_all[g] = {}, {}
        for rule in pr.RULES:
            kept = pr.prune(s, rule, knockoff_scores=k)
            kept_all[g][rule] = kept
            res[g][rule] = evaluate(kept, RULE_SCORE[rule](s, k), A, tiers)
        res[g]['_heldout_R2'] = s['heldout_R2']
        res[g]['_knockoff_s'] = k['knockoff_s']
        res[g]['_jacobian_effect'] = s['jacobian_effect']
        print('  ' + ' | '.join(f'{r}: kept {res[g][r]["kept"]} F1 {res[g][r]["f1"]:.2f} FP {res[g][r]["false_pos"]}'
                                for r in pr.RULES), flush=True)
    (OUT / 'comparison.json').write_text(json.dumps(jsonable({'results': res, 'kept': kept_all,
                                                              'noise_R2': NOISE_R2}), indent=2))
    (OUT / 'report.md').write_text(report(res, kept_all, tiers, effect))
    print(f'-> {OUT / "report.md"}')


def report(res, kept_all, tiers, effect):
    G = list(res)
    L = ['# Pruning rules compared (differential probe, true-state inputs)', '',
         'Cell = strong-graph F1 · edges kept · false positives (absent edges kept). '
         'Expert graph: 10 strong, 7 weak, 11 absent. `null` should keep 0.', '',
         '| rule | ' + ' | '.join(G) + ' |', '|---|' + '---|' * len(G)]
    for r in pr.RULES:
        L.append(f'| {r} | ' + ' | '.join(f'{res[g][r]["f1"]:.2f} · {res[g][r]["kept"]} · {res[g][r]["false_pos"]}'
                                         for g in G) + ' |')
    L += ['', 'Threshold-free ranking quality: AUROC (strong vs absent) of the score each rule thresholds.', '',
          '| rule | ' + ' | '.join(G) + ' |', '|---|' + '---|' * len(G)]
    for r in pr.RULES:
        L.append(f'| {r} | ' + ' | '.join(f'{res[g][r]["auroc_score"]:.2f}' for g in G) + ' |')
    L += ['', 'Weak edges recovered (of 7):', '', '| rule | ' + ' | '.join(G) + ' |', '|---|' + '---|' * len(G)]
    for r in pr.RULES:
        L.append(f'| {r} | ' + ' | '.join(res[g][r]['weak'] for g in G) + ' |')
    for g in G:
        L += ['', f'## {g}: kept edges per rule', '',
              '| edge | tier | expert effect | ' + ' | '.join(pr.RULES) + ' | Jacobian effect |',
              '|' + '---|' * (4 + len(pr.RULES))]
        J = np.asarray(res[g]['_jacobian_effect'])
        order = sorted(((i, j) for i in range(len(INPUT_NAMES)) for j in range(len(TARGET_NAMES))),
                       key=lambda ij: ({'strong': 0, 'weak': 1, 'absent': 2}[tiers[ij]], ij[1], ij[0]))
        for i, j in order:
            marks = ['✔' if kept_all[g][r][i][j] else '' for r in pr.RULES]
            if tiers[i, j] == 'absent' and not any(marks):
                continue
            L.append(f'| {INPUT_NAMES[i]}→{TARGET_NAMES[j]} | {tiers[i, j]} | {effect[i, j]:.3f} | '
                     + ' | '.join(marks) + f' | {J[i, j]:.3f} |')
        L.append(f'\nHeld-out R2 {np.round(res[g]["_heldout_R2"], 3).tolist()}; knockoff s '
                 f'{np.round(res[g]["_knockoff_s"], 3).tolist()} (0 = knockoff identical to the input). '
                 'Absent edges not kept by any rule are omitted.')
    return '\n'.join(L) + '\n'


if __name__ == '__main__':
    main()
