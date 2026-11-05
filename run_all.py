"""End-to-end driver (Steps 4-7).

  python run_all.py --data datasets/cc --data-baseline           # discovery on true labels only
  python run_all.py --data datasets/cc --ckpt runs/jepa_cc/checkpoints/checkpoint_epoch0200.pt
  python run_all.py --data datasets/cc --run-dir runs/jepa_cc --curve --splits 20

Outputs go to results/ (data baseline) and results/<run>/epoch####/ (per checkpoint).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cc_common import INPUT_NAMES, RESULTS, STATE_NAMES, TARGET_NAMES, load_split_episodes, stack_episodes
from discovery import edge_scores
from discovery.prune import prune
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
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def discover(X, Y, episode, n_splits, label, cross_check=True):
    t0 = time.time()
    s = edge_scores.score_graph(X, Y, episode, n_splits=n_splits)
    s['kept'] = prune(s)
    if cross_check:
        gl = edge_scores.group_lasso_weights(X, Y, episode)
        s['group_lasso_weight'] = gl['weight']
        s['group_lasso_lambda'] = gl['lambda']
        s['group_lasso_rank_agreement'] = edge_scores.rank_agreement(s['weight'], gl['weight'])
    print(f'  [{label}] held-out R2 {np.round(s["heldout_R2"], 4)}  kept {int(s["kept"].sum())} edges'
          f'  ({time.time() - t0:.0f}s)')
    return s


def data_baseline(data_dir, n_splits):
    out = RESULTS / 'data_baseline.json'
    V = stack_episodes(load_split_episodes(data_dir, 'train'))
    print(f'[data baseline] {len(V["t"])} transitions')
    g = discover(V['inputs'], V['targets'], V['episode'], n_splits, 'data')
    A, tiers, effect = compare.load_expert()
    g['vs_expert'] = compare.vs_expert(g['kept'], g['weight'], A, tiers)
    out.write_text(json.dumps(jsonable(g), indent=2))
    return g


def action_effects(E, d):
    """Model's own sensitivity of the predicted change to u_t, u_{t-1}, u_{t-2},
    in the same normalised units as the expert effect sizes."""
    P = fit_probes.Ridge().fit(E['z'], E['inputs'][:, :4])
    sd_u = E['inputs'][:, 4:7].std(0)
    sd_t = d['dd_pred'].std(0)
    eff = np.zeros((3, 4))
    for a in range(3):
        g = P.linear(E['dzhat_du'][:, :, a])               # (n, 4) change per newton
        eff[a] = np.abs(g).mean(0) * sd_u[a] / sd_t
    return eff


def run_checkpoint(ckpt, data_dir, n_splits, base, out_root):
    ckpt = Path(ckpt)
    t0 = time.time()
    E = ex.extract(str(ckpt), data_dir, 'train')
    epoch = int(E['epoch'])
    out = out_root / f'epoch{epoch:04d}'
    out.mkdir(parents=True, exist_ok=True)
    print(f'[epoch {epoch}] extracted {len(E["t"])} transitions ({time.time() - t0:.0f}s)')
    d, rep = fit_probes.fit_all(E)
    print('  probe R2 ' + ', '.join(f'{n}={v:.3f}' for n, v in zip(STATE_NAMES, rep['R2_state(x,vbar,theta,wbar)'])))
    U = E['inputs'][:, 4:7]
    Xm = np.column_stack([d['d_hat'], U])
    graphs = {'data': base}
    graphs['encoder'] = discover(Xm, d['dd_enc'], E['episode'], n_splits, 'encoder', cross_check=False)
    graphs['predictor'] = discover(Xm, d['dd_pred'], E['episode'], n_splits, 'predictor')
    A, tiers, effect = compare.load_expert()
    res = {
        'epoch': epoch, 'checkpoint': str(ckpt), 'probe': rep,
        'vs_expert': {n: compare.vs_expert(g['kept'], g['weight'], A, tiers) for n, g in graphs.items()},
        'attribution': {
            'data_vs_encoder': compare.between(graphs['data'], graphs['encoder']),
            'encoder_vs_predictor': compare.between(graphs['encoder'], graphs['predictor']),
            'data_vs_predictor': compare.between(graphs['data'], graphs['predictor']),
        },
        'action_effect_model': {'rows': ['u_t', 'u_tm1', 'u_tm2'], 'cols': TARGET_NAMES,
                                'values': action_effects(E, d)},
        'action_effect_expert': effect[4:7],
        'history_sensitivity_mean(z_t..z_t-4)': E['hist_sens'].mean(0),
        'graphs': {n: g for n, g in graphs.items() if n != 'data'},
    }
    (out / 'results.json').write_text(json.dumps(jsonable(res), indent=2))
    (out / 'edges.md').write_text(compare.edge_table(graphs, A, tiers, effect))
    compare.heatmaps(graphs, A, tiers, out / 'graphs.png')
    h = res['vs_expert']['predictor']['strong_graph']
    print(f'  predictor vs expert (strong graph): F1={h["f1"]:.3f} AUROC={h["auroc"]:.3f} SHD={h["shd"]}'
          f'  -> {out}  ({time.time() - t0:.0f}s)')
    return res


def curve(run_dir, data_dir, n_splits, base, out_root):
    cks = sorted(Path(run_dir, 'checkpoints').glob('checkpoint_epoch*.pt'))
    rows = []
    for ck in cks:
        out = out_root / f'epoch{int(ck.stem[-4:]):04d}' / 'results.json'
        r = json.loads(out.read_text()) if out.exists() else run_checkpoint(ck, data_dir, n_splits, base, out_root)
        r = jsonable(r)
        rows.append({'epoch': r['epoch'],
                     'probe_R2': r['probe']['R2_state(x,vbar,theta,wbar)'],
                     **{f'{g}_{k}': r['vs_expert'][g]['strong_graph'][k]
                        for g in ('encoder', 'predictor') for k in ('f1', 'auroc', 'shd')},
                     'predictor_full_auroc': r['vs_expert']['predictor']['full_graph']['auroc'],
                     'spearman_data_predictor': r['attribution']['data_vs_predictor']['weight_spearman']})
    (out_root / 'curve.json').write_text(json.dumps(rows, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ep = [r['epoch'] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
    for key, lab in (('predictor_f1', 'predictor F1'), ('predictor_auroc', 'predictor AUROC'),
                     ('encoder_f1', 'encoder F1'), ('spearman_data_predictor', 'weight Spearman vs data')):
        ax[0].plot(ep, [r[key] for r in rows], marker='o', ms=3, label=lab)
    ax[0].set_xlabel('epoch'); ax[0].set_ylim(-0.05, 1.05); ax[0].legend(fontsize=8)
    ax[0].set_title('edge recovery (strong graph)')
    for i, n in enumerate(STATE_NAMES):
        ax[1].plot(ep, [r['probe_R2'][i] for r in rows], marker='o', ms=3, label=n)
    ax[1].axhline(0.8, ls='--', c='gray', lw=1)
    ax[1].set_xlabel('epoch'); ax[1].set_ylim(-0.05, 1.05); ax[1].legend(fontsize=8)
    ax[1].set_title('probe R² (single latent)')
    fig.savefig(out_root / 'curve.png', dpi=110)
    print(f'[curve] {len(rows)} checkpoints -> {out_root / "curve.png"}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--run-dir', default=None)
    p.add_argument('--curve', action='store_true')
    p.add_argument('--data-baseline', action='store_true')
    p.add_argument('--splits', type=int, default=50)
    a = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    bpath = RESULTS / 'data_baseline.json'
    if a.data_baseline or not bpath.exists():
        base = data_baseline(a.data, a.splits)
        print(json.dumps(jsonable(base['vs_expert']), indent=2))
    else:
        base = json.loads(bpath.read_text())
    run_name = Path(a.run_dir or Path(a.ckpt).parent.parent).name if (a.run_dir or a.ckpt) else None
    out_root = RESULTS / run_name if run_name else None
    if a.ckpt:
        run_checkpoint(a.ckpt, a.data, a.splits, base, out_root)
    if a.curve:
        curve(a.run_dir, a.data, a.splits, base, out_root)


if __name__ == '__main__':
    main()
