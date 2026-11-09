"""End-to-end driver (Steps 4-7), differential probe only.

  python run_all.py --data datasets/cc --data-baseline           # discovery on true labels only
  python run_all.py --data datasets/cc --ckpt runs/jepa_cc/checkpoints/checkpoint_epoch0200.pt
  python run_all.py --data datasets/cc --run-dir runs/jepa_cc --curve --splits 20
  python run_all.py --data datasets/cc --ckpt <ckpt> --probe-only --seeds 3   # probe diagnostics only
  python run_all.py ... --prune effect                           # choose the pruning rule (discovery/prune.py)

The model's predicted change is read with the differential probe Q_D (probing/fit_probes.py).
Edge-scoring inputs are the true simulator states plus u_t, u_{t-1}, u_{t-2}.
Outputs: results/<run>_diff_<rule>/epoch####/{results.json, edges.md, graphs.png, summary.md};
the curve goes to results/<run>_diff_<rule>/curve_s<splits>/ plus curve.json and curve.png.
Runs are resumable: finished checkpoints and cached extractions (runs/<run>/extract/) are reused.
Data baseline: results/data_baseline_<rule>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cc_common import INPUT_NAMES, RESULTS, TARGET_NAMES, load_split_episodes, stack_episodes
from discovery import edge_scores
from discovery.prune import NEEDS_KNOCKOFF, RULES, prune
from evaluate import compare
from probing import extract as ex
from probing import fit_probes


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def discover(X, Y, episode, n_splits, label, rule, cross_check=True):
    t0 = time.time()
    s = edge_scores.score_graph(X, Y, episode, n_splits=n_splits)
    ko = (edge_scores.score_graph(X, Y, episode, n_splits=n_splits, decoys='knockoff')
          if rule in NEEDS_KNOCKOFF else None)
    s['rule'] = rule
    s['kept'] = prune(s, rule, knockoff_scores=ko)
    if ko is not None:
        s['knockoff_scores'] = ko
    if cross_check:
        gl = edge_scores.group_lasso_weights(X, Y, episode)
        s['group_lasso_weight'] = gl['weight']
        s['group_lasso_lambda'] = gl['lambda']
        s['group_lasso_rank_agreement'] = edge_scores.rank_agreement(s['weight'], gl['weight'])
    print(f'  [{label}] held-out R2 {np.round(s["heldout_R2"], 4)}  kept {int(s["kept"].sum())} edges'
          f' ({rule})  ({time.time() - t0:.0f}s)', flush=True)
    return s


def data_baseline(data_dir, n_splits, rule):
    tag = '' if Path(data_dir).name == 'cc' else f'_{Path(data_dir).name}'
    out = RESULTS / f'data_baseline{tag}_{rule}.json'
    V = stack_episodes(load_split_episodes(data_dir, 'train'))
    print(f'[data baseline] {len(V["t"])} transitions')
    g = discover(V['inputs'], V['targets'], V['episode'], n_splits, 'data', rule)
    A, tiers, effect = compare.load_expert(data_dir)
    g['vs_expert'] = compare.vs_expert(g['kept'], g['weight'], A, tiers)
    out.write_text(json.dumps(jsonable(g), indent=2))
    return g


def fmt(names, vals):
    return ', '.join(f'{n}={v:.3f}' for n, v in zip(names, vals))


def ckpt_epoch(ckpt):
    stem = Path(ckpt).stem
    if stem[-4:].isdigit():
        return int(stem[-4:])
    import torch
    return int(torch.load(ckpt, map_location='cpu', weights_only=False).get('epoch', -1))


def load_or_extract(ckpt, data_dir):
    """Extraction is the GPU-heavy step: cache it next to the run so a GPU loss never repeats it.
    An empty or unreadable cache (left by a crash mid-write) is redone."""
    ckpt = Path(ckpt)
    cache = ckpt.parent.parent / 'extract' / f'{Path(data_dir).name}_{ckpt.stem}.npz'
    if cache.exists() and cache.stat().st_size > 0:
        try:
            with np.load(cache) as f:
                return {k: f[k] for k in f.files}
        except Exception as e:   # noqa: BLE001
            print(f'  cache {cache} unreadable ({e}); extracting again')
    t0 = time.time()
    E = ex.extract(str(ckpt), data_dir, 'train', out=str(cache))
    print(f'  extracted {len(E["t"])} transitions -> {cache} ({time.time() - t0:.0f}s)')
    return E


def fmt_rep(rep):
    s = ('change R2 enc ' + fmt(TARGET_NAMES, rep['R2_change_encoder_vs_true'])
         + ' | pred ' + fmt(TARGET_NAMES, rep['R2_change_predictor_vs_true']))
    shuf = rep.get('R2_shuffled_label_control')
    return s + (f' | shuffled {np.round(shuf, 3).tolist()}' if shuf is not None else '')


def run_checkpoint(ckpt, data_dir, n_splits, base, out_root, rule, probe_only=False, seeds=1):
    """One extraction, the differential probe, then the encoder and predictor graphs.
    Resumable: a checkpoint whose results.json already exists is not recomputed."""
    ckpt, t0 = Path(ckpt), time.time()
    epoch = ckpt_epoch(ckpt)
    out = out_root / f'epoch{epoch:04d}'
    if not probe_only and (out / 'results.json').exists():
        return json.loads((out / 'results.json').read_text())
    print(f'[epoch {epoch}]', flush=True)
    out.mkdir(parents=True, exist_ok=True)
    E = load_or_extract(ckpt, data_dir)
    ep, X = E['episode'], E['inputs']
    P = fit_probes.fit_differential(E, seed=0)
    print('  ' + fmt_rep(P['report']), flush=True)
    for n, bins in P['report'].get('R2_by_physics_tercile', {}).items():
        print(f'  {n}: ' + ' | '.join(f'{b} [{v["range"][0]:.2f}-{v["range"][1]:.2f}] pred '
                                      + fmt(TARGET_NAMES, v['R2_predictor']) for b, v in bins.items()))
    if probe_only:
        reps = [P['report']]
        for s in range(1, seeds):
            reps.append(fit_probes.fit_differential(E, seed=s, shuffle_control=False)['report'])
            print(f'  [seed {s}] ' + fmt_rep(reps[-1]))
        keys = ('R2_change_encoder_vs_true', 'R2_change_predictor_vs_true')
        spread = {k: {'mean': np.mean([r[k] for r in reps], 0), 'sd': np.std([r[k] for r in reps], 0)}
                  for k in keys}
        (out / 'probe.json').write_text(json.dumps(jsonable({'seeds': reps, 'spread': spread}), indent=2))
        print(f'  -> {out / "probe.json"} ({time.time() - t0:.0f}s)')
        return reps
    A, tiers, effect = compare.load_expert(data_dir)
    graphs = {'data': base,
              'encoder': discover(X, P['dd_enc'], ep, n_splits, 'encoder', rule, cross_check=False),
              'predictor': discover(X, P['dd_pred'], ep, n_splits, 'predictor', rule)}
    testable = P['report']['testable(dx,dvbar,dtheta,dwbar)']
    res = {
        'epoch': epoch, 'checkpoint': str(ckpt), 'readout': 'differential', 'rule': rule,
        'inputs': 'true', 'n_splits': n_splits,
        'probe': P['report'],
        'vs_expert': {n: compare.vs_expert(g['kept'], g['weight'], A, tiers) for n, g in graphs.items()},
        'vs_expert_testable': {n: compare.vs_expert_testable(graphs[n]['kept'], graphs[n]['weight'],
                                                             A, tiers, testable)
                               for n in ('encoder', 'predictor')},
        'attribution': {
            'data_vs_encoder': compare.between(graphs['data'], graphs['encoder']),
            'encoder_vs_predictor': compare.between(graphs['encoder'], graphs['predictor']),
            'data_vs_predictor': compare.between(graphs['data'], graphs['predictor']),
        },
        'action_effect_model': {'rows': ['u_t', 'u_tm1', 'u_tm2'], 'cols': TARGET_NAMES,
                                'values': P['action_effect']},
        'action_effect_expert': effect[4:7],
        'history_sensitivity_mean(z_t..z_t-4)': E['hist_sens'].mean(0),
        'graphs': {n: g for n, g in graphs.items() if n != 'data'},
    }
    (out / 'results.json').write_text(json.dumps(jsonable(res), indent=2))
    (out / 'edges.md').write_text(compare.edge_table(graphs, A, tiers, effect, rule))
    compare.heatmaps(graphs, A, tiers, out / 'graphs.png')
    (out / 'summary.md').write_text(f'# Epoch {epoch}: differential probe, pruning rule "{rule}"\n\n'
                                    f'Inputs: true states. Splits: {n_splits}.\n\n'
                                    + compare.summary(jsonable(res)))
    h = res['vs_expert']['predictor']['strong_graph']
    print(f'  predictor strong graph: F1={h["f1"]:.3f} AUROC={h["auroc"]:.3f} SHD={h["shd"]}'
          f'  -> {out} ({time.time() - t0:.0f}s)', flush=True)
    return jsonable(res)


def curve(run_dir, data_dir, n_splits, base, out_root, rule):
    croot = out_root / f'curve_s{n_splits}'
    cks = sorted(Path(run_dir, 'checkpoints').glob('checkpoint_epoch*.pt'))
    R = [run_checkpoint(ck, data_dir, n_splits, base, croot, rule) for ck in cks]
    A, tiers, effect = compare.load_expert(data_dir)
    strong = [(i, j) for i, j in zip(*np.where(tiers == 'strong'))]
    strong_names = [f'{INPUT_NAMES[i]}→{TARGET_NAMES[j]}' for i, j in strong]
    ep = [r['epoch'] for r in R]
    emerg = {}
    for g in ('encoder', 'predictor'):
        em = compare.emergence(ep, np.array([r['graphs'][g]['kept'] for r in R]))
        emerg[g] = {f'{INPUT_NAMES[i]}->{TARGET_NAMES[j]}': em[i, j]
                    for i in range(len(INPUT_NAMES)) for j in range(len(TARGET_NAMES)) if em[i, j] is not None}
    rows = []
    for r in R:
        rows.append({'epoch': r['epoch'],
                     **{f'{g}_{k}': r['vs_expert'][g]['strong_graph'][k]
                        for g in ('encoder', 'predictor') for k in ('f1', 'auroc', 'shd')},
                     **{f'{g}_full_auroc': r['vs_expert'][g]['full_graph']['auroc'] for g in ('encoder', 'predictor')},
                     **{f'{g}_false_pos': len(r['vs_expert'][g]['absent_edges_wrongly_kept'])
                        for g in ('encoder', 'predictor')},
                     'predictor_f1_testable': r['vs_expert_testable']['predictor']['f1'],
                     'spearman_data_predictor': r['attribution']['data_vs_predictor']['weight_spearman'],
                     'spearman_encoder_predictor': r['attribution']['encoder_vs_predictor']['weight_spearman'],
                     'change_R2_encoder': r['probe']['R2_change_encoder_vs_true'],
                     'change_R2_predictor': r['probe']['R2_change_predictor_vs_true'],
                     'strong_edge_weight_predictor': [r['graphs']['predictor']['weight'][i][j] for i, j in strong],
                     'strong_edge_kept_predictor': [r['graphs']['predictor']['kept'][i][j] for i, j in strong]})
    (out_root / 'curve.json').write_text(json.dumps(jsonable(
        {'readout': 'differential', 'rule': rule, 'n_splits': n_splits, 'rows': rows,
         'emergence_epoch': emerg, 'strong_edges': strong_names}), indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (a0, a1, a2) = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
    for key, lab in (('predictor_f1', 'predictor F1'), ('predictor_auroc', 'predictor AUROC'),
                     ('encoder_f1', 'encoder F1'), ('spearman_data_predictor', 'Spearman vs data')):
        a0.plot(ep, [w[key] for w in rows], marker='o', ms=3, label=lab)
    a0.set_ylim(-0.05, 1.05); a0.set_xlabel('epoch'); a0.legend(fontsize=8)
    a0.set_title(f'edge recovery (strong graph, rule {rule})')
    for j, t in enumerate(TARGET_NAMES):
        line, = a1.plot(ep, [w['change_R2_predictor'][j] for w in rows], marker='o', ms=3, label=t)
        a1.plot(ep, [w['change_R2_encoder'][j] for w in rows], ls='--', c=line.get_color(), lw=1)
    a1.axhline(fit_probes.TESTABLE_R2, ls=':', c='gray', lw=1)
    a1.set_ylim(-0.25, 1.05); a1.set_xlabel('epoch'); a1.legend(fontsize=8)
    a1.set_title('change R² (solid predictor, dashed encoder)')
    W = np.array([w['strong_edge_weight_predictor'] for w in rows]).T
    a2.imshow(W, aspect='auto', cmap='Blues', vmin=0)
    yi, xi = np.where(np.array([w['strong_edge_kept_predictor'] for w in rows]).T.astype(bool))
    a2.scatter(xi, yi, s=10, c='#d1495b')
    a2.set_yticks(range(len(strong_names)), strong_names, fontsize=7)
    a2.set_xticks(range(len(ep))[::2], ep[::2], fontsize=7)
    a2.set_title('strong-edge LOCO weight (dot = kept)')
    fig.savefig(out_root / 'curve.png', dpi=110)
    print(f'[curve] {len(cks)} checkpoints -> {out_root / "curve.png"}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--run-dir', default=None)
    p.add_argument('--curve', action='store_true')
    p.add_argument('--data-baseline', action='store_true')
    p.add_argument('--splits', type=int, default=50)
    p.add_argument('--prune', choices=RULES, default='margin', help='pruning rule, see discovery/prune.py')
    p.add_argument('--seeds', type=int, default=1, help='probe seed repeats (with --probe-only)')
    p.add_argument('--probe-only', action='store_true',
                   help='only extract and report probe diagnostics (fast, no edge scoring)')
    a = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    run_name = Path(a.run_dir or Path(a.ckpt).parent.parent).name if (a.run_dir or a.ckpt) else None
    data_tag = '' if Path(a.data).name == 'cc' else f'_{Path(a.data).name}'    # e.g. _cc_phys
    out_root = RESULTS / f'{run_name}{data_tag}_diff_{a.prune}' if run_name else None
    if a.probe_only:
        assert a.ckpt, '--probe-only needs --ckpt'
        run_checkpoint(a.ckpt, a.data, a.splits, None, out_root, a.prune, probe_only=True, seeds=a.seeds)
        return
    bpath = RESULTS / f'data_baseline{data_tag}_{a.prune}.json'
    if a.data_baseline or not bpath.exists():
        base = data_baseline(a.data, a.splits, a.prune)
        print(json.dumps(jsonable(base['vs_expert']), indent=2))
    else:
        base = json.loads(bpath.read_text())
    if a.ckpt:
        run_checkpoint(a.ckpt, a.data, a.splits, base, out_root, a.prune)
    if a.curve:
        curve(a.run_dir, a.data, a.splits, base, out_root, a.prune)


if __name__ == '__main__':
    main()
