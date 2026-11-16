"""Do the slots hold their variables, and does ctx hold physics rather than state? (any JEPA run)

For every slot k of the latent and every target v (each state variable, and each hidden physics
parameter), fit a LINEAR ridge readout slot_k -> v on half of the episodes and score R² on the other
half (averaged over --splits random episode halves). Linear readouts only, so a strong probe cannot
manufacture alignment that is not there. Then per state variable v:
  own R²       R²(slot_v -> v)
  selectivity  own R² - best R² of any OTHER state slot          (> 0: v lives in its own slot)
  share        own R² / R²(all slots -> v)
  unique       R²(all slots) - R²(all slots except slot_v)      (> 0: only slot_v has it)
and for ctx: R²(ctx -> each state variable) (leak, low is good) and R²(ctx -> each physics parameter)
(high is good), next to the best state slot's physics R².
Wrapped angles (Acrobot th1, th2) are read as (sin, cos); their R² is the mean of the two.

Slot layout: the run's own (per-output arm, masked/Jacobian arms); a standard JEPA has no slots, so
it is cut into the default layout as a NULL reference (its numbers show what chance alignment looks like).

  python -m evaluate.slot_selectivity --data datasets/cc_phys --split val \
      --runs runs/jepa_cc_phys_e100 runs/jepa_cc_phys_peroutput_e100 runs/jepa_cc_phys_peroutput_dense_e100

Uses run_all.load_or_extract (cached in runs/<run>/extract/). Output:
results/studies/graph_prior_training/selectivity/{selectivity.md, selectivity.json, <label>.png}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import envs
import paths
from models.jepa_peroutput.graph import CTX, default_layout

INK, MUTED = '#0b0b0b', '#52514e'


def ridge_r2(Xtr, Ytr, Xte, Yte, ridge=1e-3):
    xm, xs = Xtr.mean(0), np.maximum(Xtr.std(0), 1e-6)
    ym = Ytr.mean(0)
    A, B = (Xtr - xm) / xs, (Xte - xm) / xs
    W = np.linalg.solve(A.T @ A + ridge * len(A) * np.eye(A.shape[1]), A.T @ (Ytr - ym))
    P = B @ W + ym
    return 1 - ((P - Yte) ** 2).sum(0) / np.maximum(((Yte - Yte.mean(0)) ** 2).sum(0), 1e-12)


def layout_of(ckpt, env):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    mc = ck.get('model_config', {})
    d = int(mc.get('latent_dim', 192))
    if 'slot_sizes' in mc:
        names, sizes, kind = list(mc.get('slot_names') or default_layout(env.STATE_NAMES, d)[0]), \
            [int(s) for s in mc['slot_sizes']], 'own'
    else:
        names, sizes = default_layout(env.STATE_NAMES, d)
        kind = 'null (no slots: default cut)'
    starts = np.cumsum([0] + sizes)
    return names, [slice(int(a), int(b)) for a, b in zip(starts[:-1], starts[1:])], kind


def targets(E, env):
    """[(name, (n, c) array)] for each state variable (sin/cos for wrapped angles) and physics parameter."""
    pos, ang = list(getattr(env, 'POS', [0, 2])), list(getattr(env, 'ANGLE', []))
    angles = {env.STATE_NAMES[2 * k] for k, a in enumerate(ang) if a} if ang else set()
    X = E['inputs']
    out = []
    for k, name in enumerate(env.STATE_NAMES):
        x = X[:, k]
        out.append((name, np.column_stack([np.sin(x), np.cos(x)]) if name in angles else x[:, None]))
    phys = []
    if 'physics' in E:
        for k, name in enumerate(env.PHYS_KEYS):
            col = E['physics'][:, k]
            if np.std(col) > 1e-9:                 # fixed physics: nothing to read
                phys.append((name, col[:, None]))
    return out, phys


def score(z, Y, episode, splits, seed=0):
    rng = np.random.default_rng(seed)
    eps = np.unique(episode)
    r = []
    for _ in range(splits):
        perm = rng.permutation(eps)
        tr = np.isin(episode, perm[:len(eps) // 2])
        r.append(ridge_r2(z[tr], Y[tr], z[~tr], Y[~tr]))
    return float(np.mean(np.mean(r, axis=0)))


def analyse(ckpt, data, split, splits):
    from run_all import load_or_extract
    env = envs.use_data(data)
    E = load_or_extract(ckpt, data, split)
    names, slices, kind = layout_of(ckpt, env)
    z, ep = E['z'].astype(np.float64), E['episode']
    states, phys = targets(E, env)
    tg = states + phys
    R = np.array([[score(z[:, sl], Y, ep, splits) for _, Y in tg] for sl in slices])     # (slots, targets)
    allr = np.array([score(z, Y, ep, splits) for _, Y in tg])
    rows = {}
    ci = names.index(CTX) if CTX in names else None
    state_slots = [k for k, n in enumerate(names) if n != CTX]
    for v, (vn, Y) in enumerate(states):
        own = names.index(vn)
        others = [k for k in state_slots if k != own]
        keep = np.concatenate([np.arange(sl.start, sl.stop) for k, sl in enumerate(slices) if k != own])
        drop = score(z[:, keep], Y, ep, splits)
        rows[vn] = {'own_r2': float(R[own, v]), 'best_other_r2': float(R[others, v].max()),
                    'best_other_slot': names[others[int(R[others, v].argmax())]],
                    'selectivity': float(R[own, v] - R[others, v].max()),
                    'all_r2': float(allr[v]), 'share': float(R[own, v] / max(allr[v], 1e-6)),
                    'unique': float(allr[v] - drop)}
    ns = len(states)
    ctx = None
    if ci is not None:
        ctx = {'state_leak_r2': {vn: float(R[ci, v]) for v, (vn, _) in enumerate(states)},
               'physics_r2': {pn: float(R[ci, ns + p]) for p, (pn, _) in enumerate(phys)},
               'best_state_slot_physics_r2': {pn: float(R[state_slots, ns + p].max()) for p, (pn, _) in enumerate(phys)},
               'all_slots_physics_r2': {pn: float(allr[ns + p]) for p, (pn, _) in enumerate(phys)}}
    return {'ckpt': str(ckpt), 'split': split, 'layout': kind, 'slots': names,
            'targets': [n for n, _ in tg], 'R2': R.tolist(), 'all_slots_R2': allr.tolist(),
            'state': rows, 'ctx': ctx,
            'summary': {'mean_selectivity': float(np.mean([r['selectivity'] for r in rows.values()])),
                        'mean_unique': float(np.mean([r['unique'] for r in rows.values()])),
                        'ctx_max_state_leak': (max(ctx['state_leak_r2'].values()) if ctx else None),
                        'ctx_mean_physics_r2': (float(np.mean(list(ctx['physics_r2'].values())))
                                                if ctx and ctx['physics_r2'] else None)}}


def heatmap(res, label, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    R = np.clip(np.array(res['R2']), 0, 1)
    fig, ax = plt.subplots(figsize=(1.0 + 0.9 * R.shape[1], 0.9 + 0.55 * R.shape[0]), constrained_layout=True)
    ax.imshow(R, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            ax.text(j, i, f'{R[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='#ffffff' if R[i, j] > 0.6 else INK)
    ax.set_xticks(range(R.shape[1]), res['targets'], fontsize=8, color=MUTED)
    ax.set_yticks(range(R.shape[0]), [f'slot {n}' for n in res['slots']], fontsize=8, color=MUTED)
    ax.set_title(f'{label}: linear R², slot → variable ({res["split"]}, {res["layout"]})', fontsize=9, loc='left', color=INK)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--labels', nargs='+')
    p.add_argument('--ckpt', default='checkpoint_last.pt')
    p.add_argument('--split', default='val', choices=('train', 'val', 'test'))
    p.add_argument('--splits', type=int, default=5, help='random episode halves to average over')
    p.add_argument('--out', default=str(paths.STUDIES / 'graph_prior_training' / 'selectivity'))
    a = p.parse_args()
    labels = a.labels or [Path(r).name for r in a.runs]
    assert len(labels) == len(a.runs)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    allres = {}
    for lab, run in zip(labels, a.runs):
        print(f'[selectivity] {lab}')
        res = analyse(Path(run) / 'checkpoints' / a.ckpt, a.data, a.split, a.splits)
        allres[lab] = res
        heatmap(res, lab, out / f'{lab}.png')
    (out / 'selectivity.json').write_text(json.dumps(allres, indent=2))
    first = next(iter(allres.values()))
    svars = list(first['state'])
    md = [f'# Slot selectivity ({a.split} split, linear ridge, {a.splits} episode halves)', '',
          'Selectivity = R²(own slot) − best R² of another state slot; unique = R²(all) − R²(all but own slot). '
          'Runs without slots are cut into the default layout as a null reference.', '',
          '| run | layout | ' + ' | '.join(f'{v} sel / uniq' for v in svars) + ' | mean sel | ctx max state leak | ctx physics R² |',
          '|---|---|' + '---|' * (len(svars) + 3)]
    for lab, r in allres.items():
        s = r['summary']
        md.append(f'| {lab} | {r["layout"]} | ' + ' | '.join(
            f'{r["state"][v]["selectivity"]:+.2f} / {r["state"][v]["unique"]:+.2f}' for v in svars)
            + f' | {s["mean_selectivity"]:+.2f} | '
            + ('—' if s['ctx_max_state_leak'] is None else f'{s["ctx_max_state_leak"]:.2f}') + ' | '
            + ('—' if s['ctx_mean_physics_r2'] is None else f'{s["ctx_mean_physics_r2"]:.2f}') + ' |')
    md += ['', 'Reading: selectivity well above the null run means the slots carry their variables, so the '
           'per-output masks mean what the graph says. A high ctx state leak means ctx carries state and '
           'the masks can be bypassed through it.']
    (out / 'selectivity.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))
    print(f'[done] -> {out}')


if __name__ == '__main__':
    main()
