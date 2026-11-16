"""Compare arms across seeds: mean, spread and paired differences against a reference arm.

Each --group is label=run_dir[,run_dir...], one run per seed. Runs are paired across groups by their
training seed (training.seed in config.yaml), so every difference is "same seed, other arm".

  python -m evaluate.compare_seeds --data datasets/cc_phys \
      --group standard=runs/jepa_cc_phys_e100,runs/jepa_cc_phys_e100_s43,runs/jepa_cc_phys_e100_s44 \
      --group per-output=runs/jepa_cc_phys_peroutput_e100,runs/jepa_cc_phys_peroutput_e100_s43,runs/jepa_cc_phys_peroutput_e100_s44 \
      --group dense=...

Metrics per run: final / best val rollout loss, generalisation gap, epochs and hours to the target
(target_mult x the reference group's mean final val rollout loss), mean epochs to ridge R² >= 0.9,
final non-parent change sensitivity (runs that log it), and open-loop errors from the final checkpoint
(evaluate/rollout_eval.rollout_errors: latent relative MSE and mean state nRMSE; any environment).
A group may hold a single run (n = 1): then it is a plain multi-arm comparison, which is how the
Acrobot arms are compared (evaluate/compare_training.py assumes cart-pole's 4 state columns).
Per group: mean, sd, min, max. Per group vs reference: mean paired difference, the number of seeds
where the arm is better, and a two-sided Wilcoxon signed-rank p (scipy). With 3 seeds the smallest
possible p is 0.25, so read the wins count and the spread, not p alone.

Output: results/studies/graph_prior_training/seeds/{seeds.md, seeds.json, curves.png}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import paths

COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
LOWER_IS_BETTER = {'final_val_rollout', 'best_val_rollout', 'final_gap', 'epochs_to_target', 'hours_to_target',
                   'epochs_to_r2', 'final_nonparent_sens'}


def seed_of(run):
    p = Path(run) / 'config.yaml'
    return int((yaml.safe_load(p.read_text()) or {}).get('training', {}).get('seed', 42)) if p.exists() else 42


def first_reach(x, v, target, higher=False):
    idx = np.flatnonzero(v >= target if higher else v <= target)
    return None if len(idx) == 0 else float(x[idx[0]])


def run_metrics(run, target, r2_target):
    h = json.loads((Path(run) / 'history.json').read_text())
    ep = np.array([r['epoch'] for r in h])
    vr = np.array([r['val']['rollout'] for r in h])
    tr = np.array([r['train']['rollout'] for r in h])
    wall = np.array([r.get('wall_s', np.nan) for r in h])
    m = {'final_val_rollout': float(vr[-1]), 'best_val_rollout': float(vr.min()), 'final_gap': float(vr[-1] - tr[-1]),
         'epochs_to_target': first_reach(ep, vr, target),
         'hours_to_target': (None if not np.isfinite(wall).any() or first_reach(wall, vr, target) is None
                             else first_reach(wall, vr, target) / 3600)}
    pr = [(r['epoch'], r['probe_r2_x_xdot_theta_thetadot']) for r in h if 'probe_r2_x_xdot_theta_thetadot' in r]
    if pr:
        pe, pv = np.array([e for e, _ in pr]), np.array([v for _, v in pr])
        reach = [first_reach(pe, pv[:, k], r2_target, higher=True) for k in range(pv.shape[1])]
        m['epochs_to_r2'] = None if any(x is None for x in reach) else float(np.mean(reach))
    sens = [r['graph_sensitivity_val']['nonparent_mean'] for r in h if 'graph_sensitivity_val' in r]
    if sens:
        m['final_nonparent_sens'] = float(sens[-1])
    return m, {'epoch': ep, 'val_rollout': vr}


def summarise(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return None
    return {'n': int(len(v)), 'mean': float(v.mean()), 'sd': float(v.std(ddof=1)) if len(v) > 1 else 0.,
            'min': float(v.min()), 'max': float(v.max())}


def paired(ref, arm, metric):
    """ref, arm: {seed: value}. Mean difference arm - ref over common seeds, wins, Wilcoxon p."""
    seeds = sorted(s for s in ref if s in arm and ref[s] is not None and arm[s] is not None)
    if not seeds:
        return None
    d = np.array([arm[s] - ref[s] for s in seeds])
    better = int((d < 0).sum() if metric in LOWER_IS_BETTER else (d > 0).sum())
    p = None
    if len(d) >= 2 and np.any(d != 0):
        try:
            from scipy.stats import wilcoxon
            p = float(wilcoxon(d).pvalue)
        except Exception:   # noqa: BLE001
            p = None
    return {'seeds': seeds, 'mean_diff': float(d.mean()), 'wins': better, 'n': len(seeds), 'wilcoxon_p': p}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--group', action='append', required=True, help='label=run1,run2,...')
    p.add_argument('--reference', default=None, help='label of the reference group (default: first)')
    p.add_argument('--data', default='datasets/cc_phys')
    p.add_argument('--target-mult', type=float, default=1.1)
    p.add_argument('--r2-target', type=float, default=0.9)
    p.add_argument('--rollout-steps', type=int, default=20, help='0 skips the checkpoint evaluation')
    p.add_argument('--report-steps', type=int, nargs='+', default=[1, 10, 20])
    p.add_argument('--ckpt', default='checkpoint_last.pt')
    p.add_argument('--max-batches', type=int, default=40)
    p.add_argument('--device', default='cuda')
    p.add_argument('--out', default=str(paths.STUDIES / 'graph_prior_training' / 'seeds'))
    a = p.parse_args()
    groups = {}
    for g in a.group:
        lab, _, runs = g.partition('=')
        groups[lab] = [r for r in runs.split(',') if r]
    assert len(groups) <= len(COLORS), f'at most {len(COLORS)} groups'
    ref = a.reference or next(iter(groups))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ref_final = [json.loads((Path(r) / 'history.json').read_text())[-1]['val']['rollout'] for r in groups[ref]]
    target = a.target_mult * float(np.mean(ref_final))
    per_run, curves = {}, {}
    if a.rollout_steps > 0:
        import torch
        from evaluate.rollout_eval import rollout_errors      # any number of state columns (Acrobot too)
        device = torch.device(a.device if torch.cuda.is_available() else 'cpu')
    for lab, runs in groups.items():
        per_run[lab], curves[lab] = {}, []
        for run in runs:
            m, c = run_metrics(run, target, a.r2_target)
            if a.rollout_steps > 0:
                ro = rollout_errors(run, a.data, a.ckpt, a.rollout_steps, 64, a.max_batches, device)
                for k in a.report_steps:
                    if k <= a.rollout_steps:
                        m[f'rollout{k}_latent'] = float(ro['latent_rel_mse'][k - 1])
                        m[f'rollout{k}_state'] = float(np.mean(ro['state_nrmse'][k - 1]))
                        LOWER_IS_BETTER.update({f'rollout{k}_latent', f'rollout{k}_state'})
            per_run[lab][seed_of(run)] = m
            curves[lab].append(c)

    metrics = sorted({k for g in per_run.values() for m in g.values() for k in m})
    summary = {lab: {k: summarise([m.get(k) for m in g.values()]) for k in metrics} for lab, g in per_run.items()}
    vs_ref = {lab: {k: paired({s: m.get(k) for s, m in per_run[ref].items()}, {s: m.get(k) for s, m in g.items()}, k)
                    for k in metrics} for lab, g in per_run.items() if lab != ref}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    for k, (lab, cs) in enumerate(curves.items()):
        L = min(len(c['epoch']) for c in cs)
        V = np.array([c['val_rollout'][:L] for c in cs])
        e = cs[0]['epoch'][:L]
        ax.fill_between(e, V.min(0), V.max(0), color=COLORS[k], alpha=0.15, lw=0)
        ax.plot(e, V.mean(0), c=COLORS[k], lw=2, label=f'{lab} (n={len(cs)})')
    ax.axhline(target, c=MUTED, ls=':', lw=1)
    ax.set_yscale('log'); ax.set_xlabel('epoch', color=MUTED); ax.set_ylabel('val rollout loss (log)', color=MUTED)
    ax.set_title('Val rollout loss: seed mean, band = min–max over seeds', loc='left', fontsize=10, color=INK)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False)
    fig.savefig(out / 'curves.png', dpi=130); plt.close(fig)

    fmt = lambda s: '—' if s is None else (f'{s["mean"]:.4g} ± {s["sd"]:.2g} (n={s["n"]})')
    md = [f'# Seed comparison (reference: {ref}; target val rollout {target:.4f})', '',
          '| metric | ' + ' | '.join(groups) + ' |', '|---|' + '---|' * len(groups)]
    for k in metrics:
        md.append(f'| {k} | ' + ' | '.join(fmt(summary[lab][k]) for lab in groups) + ' |')
    md += ['', f'Paired against **{ref}** (same seed): mean difference, seeds where the arm is better, Wilcoxon p', '',
           '| metric | ' + ' | '.join(l for l in groups if l != ref) + ' |', '|---|' + '---|' * (len(groups) - 1)]
    for k in metrics:
        cells = []
        for lab in groups:
            if lab == ref:
                continue
            r = vs_ref[lab][k]
            cells.append('—' if r is None else f'{r["mean_diff"]:+.3g}, {r["wins"]}/{r["n"]} better'
                         + ('' if r['wilcoxon_p'] is None else f', p={r["wilcoxon_p"]:.2f}'))
        md.append(f'| {k} | ' + ' | '.join(cells) + ' |')
    md += ['', 'Lower is better for losses, gaps, times, sensitivities and rollout errors. With 3 seeds the '
           'smallest two-sided Wilcoxon p is 0.25: treat "3/3 better" plus non-overlapping spreads as the evidence.']
    (out / 'seeds.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (out / 'seeds.json').write_text(json.dumps({'reference': ref, 'target_val_rollout': target, 'per_run': per_run,
                                                'summary': summary, 'vs_reference': vs_ref}, indent=2, default=str))
    print('\n'.join(md))
    print(f'[done] -> {out}')


if __name__ == '__main__':
    main()
