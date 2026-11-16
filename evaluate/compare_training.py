"""Compare JEPA training runs: standard JEPA vs the graph-prior arms (masked attention, Jacobian penalty).

Reads each run's history.json (and, if present, the run_all.py --curve output), and optionally
evaluates the final checkpoints on long open-loop rollouts over the validation split.

  python -m evaluate.compare_training --data datasets/cc_phys \
      --runs runs/jepa_cc_phys_e100 runs/jepa_cc_phys_gmask_e100 runs/jepa_cc_phys_gjac_e100 \
      --labels standard masked jacobian

Outputs (results/studies/graph_prior_training/ by default):
  training.png    val rollout loss against epoch and against optimisation time, train/val gap
  probe_r2.png    ridge R² of the latent for x, xdot, theta, thetadot against epoch
  rollout.png     open-loop error against rollout step (--rollout-steps > 0)
  graph.png       edge recovery (predictor F1 / AUROC) from run_all.py --curve, and the slot
                  sensitivity diagnostic of runs with a slot layout
  summary.md, summary.json

Efficiency numbers in the table:
  epochs / time to target  first epoch whose val rollout loss is <= target_mult x the FIRST run's
                           final val rollout loss (the first run should be the standard JEPA)
  epochs to R² >= 0.9      per variable, from the probe epochs
Rollout errors are reported two ways, because raw latent MSE is not comparable across models:
  latent rel. MSE   ||zhat - z||^2 / ||z - mean z||^2 per step
  state nRMSE       a ridge readout fitted on the model's own encoder latents (val windows, real
                    latents only, never predictions) maps predictions to state; RMSE / state std
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import paths

STATE_LABELS = ['x', 'xdot', 'theta', 'thetadot']
# validated categorical order (dataviz reference palette, light mode): blue, orange, aqua, yellow, magenta
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def load_history(run):
    h = json.loads((Path(run) / 'history.json').read_text())
    ep = np.array([r['epoch'] for r in h])
    get = lambda split, k: np.array([r[split].get(k, np.nan) for r in h])
    probe = [(r['epoch'], r['probe_r2_x_xdot_theta_thetadot']) for r in h if 'probe_r2_x_xdot_theta_thetadot' in r]
    sens = [(r['epoch'], r['graph_sensitivity_val']) for r in h if 'graph_sensitivity_val' in r]
    wall = np.array([r.get('wall_s', np.nan) for r in h])
    return {'epoch': ep, 'val_rollout': get('val', 'rollout'), 'train_rollout': get('train', 'rollout'),
            'val_loss': get('val', 'loss'), 'wall_s': wall,
            'train_s': np.array([r.get('train_s', np.nan) for r in h]),
            'probe_epoch': np.array([e for e, _ in probe]), 'probe_r2': np.array([v for _, v in probe]),
            'sens_epoch': np.array([e for e, _ in sens]), 'sens': [v for _, v in sens]}


def first_reach(x, values, target, higher=False):
    ok = values >= target if higher else values <= target
    idx = np.flatnonzero(ok)
    return None if len(idx) == 0 else float(x[idx[0]])


def find_curve(run, data, rule):
    tag = '' if Path(data).name == 'cc' else f'_{Path(data).name}'
    p = paths.model_results(run) / f'{Path(run).name}{tag}_diff_{rule}' / 'curve.json'
    return json.loads(p.read_text()) if p.exists() else None


def ridge_fit(z, y, ridge=1e-3):
    zm, zs = z.mean(0), np.maximum(z.std(0), 1e-6)
    x = (z - zm) / zs
    w = np.linalg.solve(x.T @ x + ridge * len(x) * np.eye(x.shape[1]), x.T @ (y - y.mean(0)))
    return lambda q: ((q - zm) / zs) @ w + y.mean(0)


@torch.no_grad()
def rollout_errors(run, data, ckpt_name, steps, batch_size, max_batches, device):
    from torch.utils.data import DataLoader
    from datagen.windows import WindowDataset, load_meta
    from models.jepa.train import action_contexts, encode_window
    from probing.extract_jepa import load_model
    model, scale, epoch = load_model(Path(run) / 'checkpoints' / ckpt_name, device, data)
    hist = model.history_size
    ds = WindowDataset(Path(data) / 'val.hdf5', hist - 1 + steps, model.action_context_dim,
                       float(load_meta(data)['action_scale']))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(0))
    P, Y, S, ZH, SH = [], [], [], [], []
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        z = encode_window(model, batch, device)
        a = action_contexts(model, batch, device)
        zw, preds = z[:, :hist], []
        for k in range(steps):
            p = model.predict(zw, a[:, k:k + hist])[:, -1]
            preds.append(p)
            zw = torch.cat([zw[:, 1:], p[:, None]], dim=1)
        P.append(torch.stack(preds, 1).cpu().numpy()); Y.append(z[:, hist:].cpu().numpy())
        S.append(batch['states'][:, hist:].numpy())
        ZH.append(z[:, :hist].reshape(-1, z.shape[-1]).cpu().numpy())
        SH.append(batch['states'][:, :hist].reshape(-1, 4).numpy())
    P, Y, S = np.concatenate(P), np.concatenate(Y), np.concatenate(S)        # (n, steps, d|4)
    zf = Y.reshape(-1, Y.shape[-1])
    lat = ((P - Y) ** 2).sum(-1).mean(0) / ((zf - zf.mean(0)) ** 2).sum(-1).mean()
    read = ridge_fit(np.concatenate(ZH), np.concatenate(SH))
    sp = read(P.reshape(-1, P.shape[-1])).reshape(S.shape)
    sstd = np.concatenate(SH).std(0)
    nrmse = np.sqrt(((sp - S) ** 2).mean(0)) / sstd                          # (steps, 4)
    return {'epoch': epoch, 'n_windows': int(len(P)), 'latent_rel_mse': lat.tolist(),
            'state_nrmse': nrmse.tolist()}


def style(ax, title, xlabel, ylabel=None):
    ax.set_title(title, fontsize=10, color=INK, loc='left')
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True, help='run dirs; the first is the reference (standard JEPA)')
    p.add_argument('--labels', nargs='+')
    p.add_argument('--data', default='datasets/cc_phys')
    p.add_argument('--out', default=str(paths.STUDIES / 'graph_prior_training'))
    p.add_argument('--target-mult', type=float, default=1.1)
    p.add_argument('--r2-target', type=float, default=0.9)
    p.add_argument('--rollout-steps', type=int, default=20, help='0 skips the checkpoint evaluation')
    p.add_argument('--report-steps', type=int, nargs='+', default=[1, 3, 10, 20])
    p.add_argument('--ckpt', default='checkpoint_last.pt')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--max-batches', type=int, default=40)
    p.add_argument('--curve-rule', default='null_cal_fwer')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    labels = a.labels or [Path(r).name for r in a.runs]
    assert len(labels) == len(a.runs), '--labels must match --runs'
    assert len(a.runs) <= len(COLORS), f'at most {len(COLORS)} runs per comparison'
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(a.device if torch.cuda.is_available() else 'cpu')

    H = [load_history(r) for r in a.runs]
    target = a.target_mult * float(H[0]['val_rollout'][-1])
    rows = []
    for lab, run, h in zip(labels, a.runs, H):
        r = {'label': lab, 'run': str(run), 'epochs': int(h['epoch'][-1]),
             'final_val_rollout': float(h['val_rollout'][-1]),
             'best_val_rollout': float(np.nanmin(h['val_rollout'])),
             'final_gap': float(h['val_rollout'][-1] - h['train_rollout'][-1]),
             's_per_epoch': float(np.nanmedian(h['train_s'])) if np.isfinite(h['train_s']).any() else None,
             'epochs_to_target': first_reach(h['epoch'], h['val_rollout'], target),
             'time_to_target_s': first_reach(h['wall_s'], h['val_rollout'], target)
             if np.isfinite(h['wall_s']).any() else None}
        if len(h['probe_r2']):
            r['final_r2'] = dict(zip(STATE_LABELS, h['probe_r2'][-1].tolist()))
            r['epochs_to_r2'] = {s: first_reach(h['probe_epoch'], h['probe_r2'][:, k], a.r2_target, higher=True)
                                 for k, s in enumerate(STATE_LABELS)}
        cv = find_curve(run, a.data, a.curve_rule)
        if cv:
            last = cv['rows'][-1]
            r['curve'] = {'epoch': [w['epoch'] for w in cv['rows']],
                          'predictor_f1': [w['predictor_f1'] for w in cv['rows']],
                          'predictor_auroc': [w['predictor_auroc'] for w in cv['rows']],
                          'predictor_false_pos': [w['predictor_false_pos'] for w in cv['rows']]}
            r['final_predictor_f1'], r['final_predictor_auroc'] = last['predictor_f1'], last['predictor_auroc']
        if a.rollout_steps > 0:
            print(f'[rollout] {lab}: {a.rollout_steps}-step open loop on val')
            r['rollout'] = rollout_errors(run, a.data, a.ckpt, a.rollout_steps, a.batch_size, a.max_batches, device)
        rows.append(r)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'legend.frameon': False})
    lw = 2

    fig, ax = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for k, (lab, h) in enumerate(zip(labels, H)):
        ax[0].plot(h['epoch'], h['val_rollout'], c=COLORS[k], lw=lw, label=lab)
        ax[1].plot(h['wall_s'] / 3600, h['val_rollout'], c=COLORS[k], lw=lw, label=lab)
        ax[2].plot(h['epoch'], h['val_rollout'] - h['train_rollout'], c=COLORS[k], lw=lw, label=lab)
    for x in ax[:2]:
        x.axhline(target, c=MUTED, ls=':', lw=1)
        x.set_yscale('log')
    style(ax[0], 'Val rollout loss', 'epoch', 'latent MSE (log)')
    style(ax[1], 'Val rollout loss against optimisation time', 'hours', 'latent MSE (log)')
    style(ax[2], 'Generalisation gap (val − train rollout)', 'epoch')
    ax[0].legend(fontsize=8)
    fig.savefig(out / 'training.png', dpi=130); plt.close(fig)

    if any(len(h['probe_r2']) for h in H):
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True, sharey=True)
        for k, (lab, h) in enumerate(zip(labels, H)):
            if len(h['probe_r2']):
                for j in range(4):
                    ax[j].plot(h['probe_epoch'], h['probe_r2'][:, j], c=COLORS[k], lw=lw, marker='o', ms=4, label=lab)
        for j, s in enumerate(STATE_LABELS):
            ax[j].axhline(a.r2_target, c=MUTED, ls=':', lw=1)
            style(ax[j], f'Ridge R² of latent: {s}', 'epoch')
        ax[0].legend(fontsize=8)
        fig.savefig(out / 'probe_r2.png', dpi=130); plt.close(fig)

    if a.rollout_steps > 0:
        fig, ax = plt.subplots(1, 5, figsize=(19, 3.6), constrained_layout=True)
        st = np.arange(1, a.rollout_steps + 1)
        for k, (lab, r) in enumerate(zip(labels, rows)):
            ax[0].plot(st, r['rollout']['latent_rel_mse'], c=COLORS[k], lw=lw, label=lab)
            nr = np.array(r['rollout']['state_nrmse'])
            for j in range(4):
                ax[j + 1].plot(st, nr[:, j], c=COLORS[k], lw=lw, label=lab)
        style(ax[0], 'Open-loop latent rel. MSE', 'rollout step')
        for j, s in enumerate(STATE_LABELS):
            style(ax[j + 1], f'Open-loop state nRMSE: {s}', 'rollout step')
        ax[0].legend(fontsize=8)
        fig.savefig(out / 'rollout.png', dpi=130); plt.close(fig)

    has_curve = any('curve' in r for r in rows)
    has_sens = any(len(h['sens']) for h in H)
    if has_curve or has_sens:
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), constrained_layout=True)
        for k, (lab, r, h) in enumerate(zip(labels, rows, H)):
            if 'curve' in r:
                ax[0].plot(r['curve']['epoch'], r['curve']['predictor_f1'], c=COLORS[k], lw=lw, marker='o', ms=4, label=lab)
                ax[1].plot(r['curve']['epoch'], r['curve']['predictor_auroc'], c=COLORS[k], lw=lw, marker='o', ms=4, label=lab)
            if len(h['sens']):
                ax[2].plot(h['sens_epoch'], [s['nonparent_mean'] for s in h['sens']], c=COLORS[k], lw=lw,
                           label=f'{lab}: non-parent')
                ax[2].plot(h['sens_epoch'], [s['parent_mean'] for s in h['sens']], c=COLORS[k], lw=lw, ls='--',
                           label=f'{lab}: parent')
        style(ax[0], f'Predictor strong-graph F1 ({a.curve_rule})', 'epoch')
        style(ax[1], 'Predictor strong-graph AUROC', 'epoch')
        style(ax[2], 'Slot sensitivity on val (slot-layout runs)', 'epoch')
        ax[0].set_ylim(-0.05, 1.05); ax[1].set_ylim(0.3, 1.02)
        if has_sens:
            ax[2].set_yscale('symlog', linthresh=1e-6)
            ax[2].legend(fontsize=7)
        if has_curve:
            ax[0].legend(fontsize=8)
        fig.savefig(out / 'graph.png', dpi=130); plt.close(fig)

    def f(v, fmt='{:.4f}'):
        return '—' if v is None else fmt.format(v)

    md = ['# Graph-prior JEPA training comparison', '',
          f'Reference run: **{labels[0]}**. Target val rollout loss = {a.target_mult} × its final value '
          f'= {target:.4f}. Times are optimisation seconds only (no evaluation).', '',
          '| run | epochs | s / epoch | epochs to target | hours to target | best val rollout | final val rollout '
          '| final gap | final F1 | final AUROC |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        md.append(f'| {r["label"]} | {r["epochs"]} | {f(r["s_per_epoch"], "{:.0f}")} | '
                  f'{f(r["epochs_to_target"], "{:.0f}")} | '
                  f'{f(None if r["time_to_target_s"] is None else r["time_to_target_s"] / 3600, "{:.2f}")} | '
                  f'{r["best_val_rollout"]:.4f} | {r["final_val_rollout"]:.4f} | {r["final_gap"]:.4f} | '
                  f'{f(r.get("final_predictor_f1"), "{:.3f}")} | {f(r.get("final_predictor_auroc"), "{:.3f}")} |')
    if any('epochs_to_r2' in r for r in rows):
        md += ['', f'Epochs to ridge R² ≥ {a.r2_target} (final R² in brackets):', '',
               '| run | ' + ' | '.join(STATE_LABELS) + ' |', '|---' * 5 + '|']
        for r in rows:
            if 'epochs_to_r2' in r:
                md.append(f'| {r["label"]} | ' + ' | '.join(
                    f'{f(r["epochs_to_r2"][s], "{:.0f}")} ({r["final_r2"][s]:.3f})' for s in STATE_LABELS) + ' |')
    if a.rollout_steps > 0:
        ks = [k for k in a.report_steps if k <= a.rollout_steps]
        md += ['', f'Open-loop rollout on val ({rows[0]["rollout"]["n_windows"]} windows, checkpoint `{a.ckpt}`): '
               'latent rel. MSE / mean state nRMSE', '',
               '| run | ' + ' | '.join(f'{k} steps' for k in ks) + ' |', '|---' * (len(ks) + 1) + '|']
        for r in rows:
            ro = r['rollout']
            md.append(f'| {r["label"]} | ' + ' | '.join(
                f'{ro["latent_rel_mse"][k - 1]:.3f} / {np.mean(ro["state_nrmse"][k - 1]):.3f}' for k in ks) + ' |')
    (out / 'summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    (out / 'summary.json').write_text(json.dumps({'target_val_rollout': target, 'runs': rows}, indent=2))
    print('\n'.join(md))
    print(f'[done] -> {out}')


if __name__ == '__main__':
    main()
