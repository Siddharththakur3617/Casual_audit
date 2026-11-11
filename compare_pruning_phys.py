"""compare_pruning.py with every graph scored conditional on the episode physics (M, m, l, g).

diagnose_physics.py showed that on a varied-physics dataset the hidden physics confounds the
true-data graph: vbar->dvbar, vbar->dwbar, x->dvbar/dwbar look real and wbar->dwbar is inflated
(1.17 vs expert 0.09) unless the fits are told the physics. The same edges are the two false
positives of null_cal_vif_fwer on the epoch-200 predictor. Here every graph (data, data_noisy, null,
predictor/encoder at the last checkpoint and epoch 0) is scored with the physics as conditioning
inputs to every fit (not graph nodes, not left out), and the null_cal rules use a null calibrated
with the same conditioning.

Declared before running:
  - null_cal_vif_fwer is the candidate main rule. Its delta_i = 0.01 * (1 - R2_i) uses R2_i of
    input i on the other inputs AND the physics, because that is the information the conditioned
    LOCO fit has when input i is left out. (delta_i from the inputs only is reported as well.)
  - Accept it if: null keeps 0, predictor_e0 keeps 0, data / data_noisy are no worse than
    null_cal_fwer, and the conditioned predictor_e200 graph has no false positive.
  - Control: predictor_e200 with the physics permuted across episodes. If the two false positives
    disappear with true physics but not with permuted physics, they were physics confounding.
  Otherwise null_cal_fwer (unconditioned) stays the main rule.

Needs compare_pruning.py to have run (it caches the probe targets of both checkpoints).
  python compare_pruning_phys.py --data datasets/cc_phys --run-dir runs/jepa_cc_phys
Output: results/pruning_comparison_phys_<data>_<run>/{report.md, comparison.json, scores/},
        null repeats in results/null_calibration_<data>_phys/
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cc_common import INPUT_NAMES, PHYS_KEYS, RESULTS, TARGET_NAMES, load_split_episodes, stack_episodes
from compare_pruning import NOISE_R2_CLIP, load_scores
from discovery import edge_scores, null_calibration, prune as pr
from evaluate import compare
from run_all import jsonable

RULES = ('min_effect', 'null_cal_fwer', 'null_cal_vif_fwer', 'null_cal_vif_fwer_inputs_only')
PERM_SEED = 4321                        # same physics permutation as diagnose_physics.py


def scored(out, name, X, Y, ep, n_splits, cond, reuse=None):
    path = out / 'scores' / f'{name}.json'
    if not path.exists() and reuse is not None and Path(reuse).exists():
        return load_scores(reuse)       # identical computation already cached by diagnose_physics.py
    if not path.exists():
        t0 = time.time()
        s = edge_scores.score_graph(X, Y, ep, n_splits=n_splits, cond=cond)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(jsonable(s)))
        print(f'  scored {name}: held-out R2 {np.round(s["heldout_R2"], 3)} ({time.time() - t0:.0f}s)', flush=True)
    return load_scores(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--splits', type=int, default=50)
    p.add_argument('--null-reps', type=int, default=null_calibration.REPS)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    name, run = Path(a.data).name, Path(a.run_dir).name
    out = Path(a.out) if a.out else RESULTS / f'pruning_comparison_phys_{name}_{run}'
    out.mkdir(parents=True, exist_ok=True)
    base = RESULTS / f'pruning_comparison_{name}_{run}'
    A, tiers, effect = compare.load_expert(a.data)

    # rows and physics: the train split, exactly the rows of compare_pruning.py and the null
    eps = load_split_episodes(a.data, 'train', with_physics=True)
    V = stack_episodes([(s, u) for s, u, _ in eps])
    X, Y, ep = V['inputs'], V['targets'], V['episode']
    P = np.array([[ph[k] for k in PHYS_KEYS] for _, _, ph in eps])
    assert P.std(0).max() > 0, 'dataset has fixed physics: nothing to condition on'
    C, C_shuf = P[ep], P[np.random.default_rng(PERM_SEED).permutation(len(P))][ep]

    probes = sorted((base / 'probe').glob('epoch*.npz'))
    assert probes, f'run compare_pruning.py first (no cached probe targets in {base / "probe"})'
    M = {int(f.stem[5:]): dict(np.load(f)) for f in probes}
    last, first = max(M), min(M)
    for e, m in M.items():
        assert m['X'].shape == X.shape and np.allclose(m['X'], X) and (m['episode'] == ep).all(), \
            f'epoch {e} probe rows differ from the train split'

    # data_noisy and null exactly as in compare_pruning.py (same random stream)
    noise_r2 = np.clip(M[last]['R2_predictor'], *NOISE_R2_CLIP)
    rng = np.random.default_rng(12345)
    sd = Y.std(0) * np.sqrt((1 - noise_r2) / noise_r2)
    Y_noisy = Y + rng.standard_normal(Y.shape) * sd
    Y_null = Y[rng.permutation(len(Y))]
    graphs = {'data': (Y, C), 'data_noisy': (Y_noisy, C), 'null': (Y_null, C)}
    for e in (last, first):
        graphs[f'predictor_e{e}'] = (M[e]['predictor'], C)
        graphs[f'encoder_e{e}'] = (M[e]['encoder'], C)
    graphs[f'predictor_e{last}_physshuf'] = (M[last]['predictor'], C_shuf)     # control

    reuse = {'data': RESULTS / f'physics_conditioning_{name}' / 'scores' / 'data_phys_shuffled.json'}
    S = {}
    for g, (Yg, cond) in graphs.items():
        print(f'[{g}]', flush=True)
        S[g] = scored(out, g, X, Yg, ep, a.splits, cond, reuse.get(g) if a.splits == 50 else None)

    # null calibrated with the same physics conditioning
    null_dir = RESULTS / f'null_calibration_{name}_phys' if not a.out else out / 'null_calibration'
    print(f'[null calibration, physics-conditioned] {a.null_reps} shuffled-target repeats -> {null_dir}', flush=True)
    null_calibration.calibrate(a.data, a.null_reps, a.splits, out_dir=null_dir, cond=C)
    null_stats, null_r2 = null_calibration.load(a.data, a.splits, a.null_reps, out_dir=null_dir)
    r2_cond = pr.input_collinearity(np.column_stack([X, C]))[:X.shape[1]]
    d_cond = pr.DELTA * np.maximum(1 - r2_cond, pr.VIF_FLOOR)[:, None]
    d_in = pr.vif_delta(X)
    nv_cond = (null_calibration.load(a.data, a.splits, a.null_reps, out_dir=null_dir, delta=d_cond)[0], d_cond)
    nv_in = (null_calibration.load(a.data, a.splits, a.null_reps, out_dir=null_dir, delta=d_in)[0], d_in)

    def keep(s, r):
        if r == 'null_cal_vif_fwer_inputs_only':
            return pr.prune(s, 'null_cal_vif_fwer', null_vif=nv_in)
        return pr.prune(s, r, null_stats=null_stats, null_vif=nv_cond)

    kept = {g: {r: keep(s, r) for r in RULES} for g, s in S.items()}
    stat = {g: pr.null_statistic(s, delta=d_cond) for g, s in S.items()}
    res = {g: {r: compare.vs_expert(kept[g][r], stat[g], A, tiers) for r in RULES} for g in S}

    # unconditioned results for comparison
    old = json.loads((base / 'comparison.json').read_text()) if (base / 'comparison.json').exists() else None

    def cell(k, v):
        return f'{v["strong_graph"]["f1"]:.2f} · {int(np.sum(k))} · {len(v["absent_edges_wrongly_kept"])}'

    G = list(S)
    thr = null_stats.max(axis=(1, 2))
    L = [f'# Pruning rules with physics conditioning ({a.data}, {run}, {a.splits} splits)', '',
         'Every fit also sees the episode physics (M, m, l, g); physics is not a graph node. '
         f'Null: {len(null_stats)} shuffled-target repeats with the same conditioning '
         f'(held-out R2 {np.round(null_r2, 3).tolist()}); null_cal_fwer threshold max {thr.max():.2f}, '
         f'median {np.median(thr):.2f}.', '',
         'Cell = strong-graph F1 · edges kept · false positives (absent edges kept). '
         'Accept null_cal_vif_fwer if null = 0, predictor_e0 = 0, data/data_noisy no worse than null_cal_fwer, '
         f'predictor_e{last} has 0 false positives.', '',
         '## with physics conditioning', '',
         '| rule | ' + ' | '.join(G) + ' |', '|---|' + '---|' * len(G)]
    for r in RULES:
        L.append(f'| {r} | ' + ' | '.join(cell(kept[g][r], res[g][r]) for g in G) + ' |')
    if old:
        Go = [g for g in G if g in old['kept']]
        L += ['', '## without physics conditioning (from compare_pruning.py)', '',
              '| rule | ' + ' | '.join(Go) + ' |', '|---|' + '---|' * len(Go)]
        for r in ('min_effect', 'null_cal_fwer', 'null_cal_vif_fwer'):
            R = old['results']
            L.append(f'| {r} | ' + ' | '.join(
                f'{R[g][r]["f1"]:.2f} · {R[g][r]["kept"]} · {R[g][r]["false_pos"]}' for g in Go) + ' |')

    L += ['', '## per-input minimum effect delta_i = 0.01 (1 - R2_i)', '',
          '| input | R2 on other inputs + physics (used) | delta_i | R2 on other inputs only | delta_i |',
          '|---|---|---|---|---|']
    r2_in = pr.input_collinearity(X)
    for i, n in enumerate(INPUT_NAMES):
        L.append(f'| {n} | {r2_cond[i]:.3f} | {d_cond[i, 0]:.5f} | {r2_in[i]:.3f} | {d_in[i, 0]:.5f} |')

    show = [g for g in G if g.startswith(('data', 'predictor'))]
    L += ['', '## per edge: Jacobian effect · null_cal_vif_fwer statistic (✔ = kept by null_cal_vif_fwer)', '',
          '| edge | tier | expert effect | ' + ' | '.join(show) + ' |', '|' + '---|' * (3 + len(show))]
    order = sorted(((i, j) for i in range(len(INPUT_NAMES)) for j in range(len(TARGET_NAMES))),
                   key=lambda ij: ({'strong': 0, 'weak': 1, 'absent': 2}[tiers[ij]], ij[1], ij[0]))
    for i, j in order:
        cells = [f'{np.asarray(S[g]["jacobian_effect"])[i, j]:.3f} · {stat[g][i, j]:.1f}'
                 f'{" ✔" if kept[g]["null_cal_vif_fwer"][i, j] else ""}' for g in show]
        L.append(f'| {INPUT_NAMES[i]}→{TARGET_NAMES[j]} | {tiers[i, j]} | {effect[i, j]:.3f} | ' + ' | '.join(cells) + ' |')
    L += ['', 'Held-out R2 (dx, dvbar, dtheta, dwbar): ' + '; '.join(
        f'{g} {np.round(S[g]["heldout_R2"], 3).tolist()}' for g in G)]
    (out / 'report.md').write_text('\n'.join(L) + '\n', encoding='utf8')
    summary = {g: {r: {'f1': res[g][r]['strong_graph']['f1'], 'kept': int(np.sum(kept[g][r])),
                       'false_pos': res[g][r]['absent_edges_wrongly_kept'],
                       'missed': res[g][r]['strong_edges_missed']} for r in RULES} for g in G}
    (out / 'comparison.json').write_text(json.dumps(jsonable(
        {'kept': kept, 'summary': summary, 'delta_cond': d_cond[:, 0], 'R2_cond': r2_cond,
         'null_heldout_R2': null_r2, 'null_threshold_per_repeat': thr}), indent=2))
    print(f'-> {out / "report.md"}')
    for g in G:
        print(f'  {g}: ' + ' | '.join(f'{r}: {cell(kept[g][r], res[g][r])}' for r in RULES))


if __name__ == '__main__':
    main()
