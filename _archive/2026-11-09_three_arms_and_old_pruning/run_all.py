"""End-to-end driver (Steps 4-7).

  python run_all.py --data datasets/cc --data-baseline           # discovery on true labels only
  python run_all.py --data datasets/cc --ckpt runs/jepa_cc/checkpoints/checkpoint_epoch0200.pt
  python run_all.py --data datasets/cc --run-dir runs/jepa_cc --curve --splits 20
  python run_all.py --data datasets/cc --ckpt <ckpt> --probe-only --seeds 3   # probe diagnostics only
  python run_all.py --data datasets/cc --ckpt <ckpt> --arms normal,hybrid      # a subset of the arms
  python run_all.py --data datasets/cc --ckpt <ckpt> --readout delta           # LEGACY linear -> results/jepa_cc_delta/

Default: the three MLP readout arms (normal, differential, hybrid; probing/fit_probes.fit_arms)
with true-state inputs -> results/<run>_arms/epoch####/<arm>/ and epoch####/summary.md.
The curve goes to results/<run>_arms/curve_s<splits>/ plus curve_<arm>.json and curve.png.
Runs are resumable: finished (checkpoint, arm) pairs and cached extractions (runs/<run>/extract/)
are reused, so after a GPU loss just rerun the same command.
Data baseline: results/data_baseline.json.
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


def fmt(names, vals):
    return ', '.join(f'{n}={v:.3f}' for n, v in zip(names, vals))


def run_checkpoint(ckpt, data_dir, n_splits, base, out_root, readout='state', probe_only=False):
    ckpt = Path(ckpt)
    t0 = time.time()
    E = ex.extract(str(ckpt), data_dir, 'train')
    epoch = int(E['epoch'])
    out = out_root / f'epoch{epoch:04d}'
    out.mkdir(parents=True, exist_ok=True)
    print(f'[epoch {epoch}] extracted {len(E["t"])} transitions ({time.time() - t0:.0f}s)')
    d, rep = fit_probes.fit_all(E, readout=readout)
    print('  probe R2 linear ' + fmt(STATE_NAMES, rep['R2_state(x,vbar,theta,wbar)']))
    print('  probe R2 MLP    ' + fmt(STATE_NAMES, rep['R2_state_mlp(x,vbar,theta,wbar)']))
    print('  change R2, state probe: predictor ' + fmt(TARGET_NAMES, rep['R2_change_predictor_vs_true'])
          + ' | encoder ' + fmt(TARGET_NAMES, rep['R2_change_encoder_vs_true']))
    print('  change R2, delta probe: predictor ' + fmt(TARGET_NAMES, rep['R2_change_delta_probe_predictor_vs_true'])
          + ' | encoder ' + fmt(TARGET_NAMES, rep['R2_change_delta_probe_encoder_vs_true']))
    if probe_only:
        (out / 'probe.json').write_text(json.dumps(jsonable(rep), indent=2))
        print(f'  -> {out / "probe.json"}  ({time.time() - t0:.0f}s)')
        return rep
    U = E['inputs'][:, 4:7]
    Xm = np.column_stack([d['d_hat'], U])
    graphs = {'data': base}
    graphs['encoder'] = discover(Xm, d['dd_enc'], E['episode'], n_splits, 'encoder', cross_check=False)
    graphs['predictor'] = discover(Xm, d['dd_pred'], E['episode'], n_splits, 'predictor')
    A, tiers, effect = compare.load_expert()
    res = {
        'epoch': epoch, 'checkpoint': str(ckpt), 'readout': readout, 'probe': rep,
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


def curve(run_dir, data_dir, n_splits, base, out_root, readout='state'):
    cks = sorted(Path(run_dir, 'checkpoints').glob('checkpoint_epoch*.pt'))
    rows = []
    for ck in cks:
        out = out_root / f'epoch{int(ck.stem[-4:]):04d}' / 'results.json'
        r = (json.loads(out.read_text()) if out.exists()
             else run_checkpoint(ck, data_dir, n_splits, base, out_root, readout))
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


# ---------------------------------------------------------------------------
# Three readout arms (normal, differential, hybrid): see probing/fit_probes.fit_arms.
# ---------------------------------------------------------------------------

def ckpt_epoch(ckpt):
    stem = Path(ckpt).stem
    if stem[-4:].isdigit():
        return int(stem[-4:])
    import torch
    return int(torch.load(ckpt, map_location='cpu', weights_only=False).get('epoch', -1))


def load_or_extract(ckpt, data_dir):
    """Extraction is the GPU-heavy step: cache it next to the run so a GPU loss never repeats it."""
    ckpt = Path(ckpt)
    cache = ckpt.parent.parent / 'extract' / f'{Path(data_dir).name}_{ckpt.stem}.npz'
    if cache.exists():
        with np.load(cache) as f:
            return {k: f[k] for k in f.files}
    t0 = time.time()
    E = ex.extract(str(ckpt), data_dir, 'train', out=str(cache))
    print(f'  extracted {len(E["t"])} transitions -> {cache} ({time.time() - t0:.0f}s)')
    return E


def fmt_rep(rep):
    s = ('change R2 enc ' + fmt(TARGET_NAMES, rep['R2_change_encoder_vs_true'])
         + ' | pred ' + fmt(TARGET_NAMES, rep['R2_change_predictor_vs_true']))
    shuf = {k: np.round(v, 3).tolist() for k, v in rep['R2_shuffled_label_control'].items() if v is not None}
    return s + (f' | shuffled {shuf}' if shuf else '')


def run_checkpoint_arms(ckpt, data_dir, n_splits, base, out_root, arms=fit_probes.ARMS, inputs='true',
                        probe_only=False, seeds=1):
    """Per checkpoint: one extraction, then each arm's own probes, encoder and predictor graphs.
    Resumable: an arm whose results.json already exists is not recomputed."""
    ckpt, t0 = Path(ckpt), time.time()
    epoch = ckpt_epoch(ckpt)
    out = out_root / f'epoch{epoch:04d}'
    todo = [a for a in arms if probe_only or not (out / a / 'results.json').exists()]
    if todo:
        print(f'[epoch {epoch}] arms {todo}')
        E = load_or_extract(ckpt, data_dir)
        ep, U = E['episode'], E['inputs'][:, 4:7]
        if inputs == 'true':
            X = E['inputs']
        else:                                   # continuity: linear-probed state inputs
            X = np.column_stack([fit_probes.cross_fit(E['z'], E['inputs'][:, :4], ep)[0], U])
        P = fit_probes.fit_arms(E, todo, seed=0)
        for arm in todo:
            print(f'  [{arm}] ' + fmt_rep(P[arm]['report']))
        if probe_only:
            reps = {a: [P[a]['report']] for a in todo}
            for s in range(1, seeds):
                Ps = fit_probes.fit_arms(E, todo, seed=s, shuffle_control=False)
                for a in todo:
                    reps[a].append(Ps[a]['report'])
                    print(f'  [{a} seed {s}] ' + fmt_rep(Ps[a]['report']))
            for a in todo:
                (out / a).mkdir(parents=True, exist_ok=True)
                keys = ('R2_change_encoder_vs_true', 'R2_change_predictor_vs_true')
                spread = {k: {'mean': np.mean([r[k] for r in reps[a]], 0), 'sd': np.std([r[k] for r in reps[a]], 0)}
                          for k in keys}
                (out / a / 'probe.json').write_text(json.dumps(jsonable({'seeds': reps[a], 'spread': spread}), indent=2))
            print(f'  -> {out}/<arm>/probe.json ({time.time() - t0:.0f}s)')
            return reps
        A, tiers, effect = compare.load_expert()
        for arm in todo:
            r = P[arm]
            graphs = {'data': base,
                      'encoder': discover(X, r['dd_enc'], ep, n_splits, f'{arm} encoder', cross_check=False),
                      'predictor': discover(X, r['dd_pred'], ep, n_splits, f'{arm} predictor')}
            testable = r['report']['testable(dx,dvbar,dtheta,dwbar)']
            res = {
                'epoch': epoch, 'checkpoint': str(ckpt), 'arm': arm, 'inputs': inputs, 'n_splits': n_splits,
                'probe': r['report'],
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
                                        'values': r['action_effect']},
                'action_effect_expert': effect[4:7],
                'history_sensitivity_mean(z_t..z_t-4)': E['hist_sens'].mean(0),
                'graphs': {n: g for n, g in graphs.items() if n != 'data'},
            }
            (out / arm).mkdir(parents=True, exist_ok=True)
            (out / arm / 'results.json').write_text(json.dumps(jsonable(res), indent=2))
            (out / arm / 'edges.md').write_text(compare.edge_table(graphs, A, tiers, effect))
            compare.heatmaps(graphs, A, tiers, out / arm / 'graphs.png')
            h = res['vs_expert']['predictor']['strong_graph']
            print(f'  [{arm}] predictor strong graph: F1={h["f1"]:.3f} AUROC={h["auroc"]:.3f} SHD={h["shd"]}'
                  f'  ({time.time() - t0:.0f}s)')
    done = {a: json.loads((out / a / 'results.json').read_text())
            for a in fit_probes.ARMS if (out / a / 'results.json').exists()}
    if done:
        (out / 'summary.md').write_text(f'# Epoch {epoch}: readout arms side by side\n\n'
                                        f'Inputs: {inputs} states. Splits: {n_splits}.\n\n'
                                        + compare.arm_summary(done))
    return done


def curve_arms(run_dir, data_dir, n_splits, base, out_root, arms=fit_probes.ARMS, inputs='true'):
    croot = out_root / f'curve_s{n_splits}'
    cks = sorted(Path(run_dir, 'checkpoints').glob('checkpoint_epoch*.pt'))
    for ck in cks:
        run_checkpoint_arms(ck, data_dir, n_splits, base, croot, arms, inputs)
    A, tiers, effect = compare.load_expert()
    strong = [(i, j) for i, j in zip(*np.where(tiers == 'strong'))]
    strong_names = [f'{INPUT_NAMES[i]}→{TARGET_NAMES[j]}' for i, j in strong]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(len(arms), 3, figsize=(15, 3.8 * len(arms)), constrained_layout=True, squeeze=False)
    for r_i, arm in enumerate(arms):
        R = [json.loads((croot / f'epoch{ckpt_epoch(ck):04d}' / arm / 'results.json').read_text()) for ck in cks]
        ep = [r['epoch'] for r in R]
        rows, emerg = [], {}
        for g in ('encoder', 'predictor'):
            K = np.array([r['graphs'][g]['kept'] for r in R])
            em = compare.emergence(ep, K)
            emerg[g] = {f'{INPUT_NAMES[i]}->{TARGET_NAMES[j]}': em[i, j]
                        for i in range(len(INPUT_NAMES)) for j in range(len(TARGET_NAMES)) if em[i, j] is not None}
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
        (out_root / f'curve_{arm}.json').write_text(json.dumps(jsonable(
            {'arm': arm, 'inputs': inputs, 'n_splits': n_splits, 'rows': rows, 'emergence_epoch': emerg,
             'strong_edges': strong_names}), indent=2))
        a0, a1, a2 = ax[r_i]
        for key, lab in (('predictor_f1', 'predictor F1'), ('predictor_auroc', 'predictor AUROC'),
                         ('encoder_f1', 'encoder F1'), ('spearman_data_predictor', 'Spearman vs data')):
            a0.plot(ep, [w[key] for w in rows], marker='o', ms=3, label=lab)
        a0.set_ylim(-0.05, 1.05); a0.set_xlabel('epoch'); a0.legend(fontsize=8)
        a0.set_title(f'{arm}: edge recovery (strong graph)')
        for j, t in enumerate(TARGET_NAMES):
            l, = a1.plot(ep, [w['change_R2_predictor'][j] for w in rows], marker='o', ms=3, label=t)
            a1.plot(ep, [w['change_R2_encoder'][j] for w in rows], ls='--', c=l.get_color(), lw=1)
        a1.axhline(fit_probes.TESTABLE_R2, ls=':', c='gray', lw=1)
        a1.set_ylim(-0.25, 1.05); a1.set_xlabel('epoch'); a1.legend(fontsize=8)
        a1.set_title(f'{arm}: change R² (solid predictor, dashed encoder)')
        W = np.array([w['strong_edge_weight_predictor'] for w in rows]).T
        a2.imshow(W, aspect='auto', cmap='Blues', vmin=0)
        kk = np.array([w['strong_edge_kept_predictor'] for w in rows]).T
        yi, xi = np.where(kk.astype(bool))
        a2.scatter(xi, yi, s=10, c='#d1495b')
        a2.set_yticks(range(len(strong_names)), strong_names, fontsize=7)
        a2.set_xticks(range(len(ep))[::2], ep[::2], fontsize=7)
        a2.set_title(f'{arm}: strong-edge LOCO weight (dot = kept)')
    fig.savefig(out_root / 'curve.png', dpi=110)
    print(f'[curve] {len(cks)} checkpoints x {len(arms)} arms -> {out_root / "curve.png"}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--run-dir', default=None)
    p.add_argument('--curve', action='store_true')
    p.add_argument('--data-baseline', action='store_true')
    p.add_argument('--splits', type=int, default=50)
    p.add_argument('--readout', choices=fit_probes.READOUTS, default=None,
                   help="LEGACY linear readouts: 'state' = state probe W (original), 'delta' = linear delta "
                        "probe. Omit to run the three MLP readout arms instead.")
    p.add_argument('--arms', default=','.join(fit_probes.ARMS),
                   help='comma-separated readout arms: normal, differential, hybrid')
    p.add_argument('--inputs', choices=('true', 'linear'), default='true',
                   help="edge-scoring inputs for the arms: 'true' simulator states, or 'linear'-probed (continuity)")
    p.add_argument('--seeds', type=int, default=1, help='probe seed repeats (with --probe-only)')
    p.add_argument('--probe-only', action='store_true',
                   help='only extract and report probe diagnostics (fast, no edge scoring)')
    a = p.parse_args()
    RESULTS.mkdir(exist_ok=True)
    arms = [s.strip() for s in a.arms.split(',') if s.strip()]
    assert set(arms) <= set(fit_probes.ARMS), arms
    legacy = a.readout is not None
    run_name = Path(a.run_dir or Path(a.ckpt).parent.parent).name if (a.run_dir or a.ckpt) else None
    if run_name:
        if legacy:
            run_name += '' if a.readout == 'state' else f'_{a.readout}'    # results/jepa_cc[_delta]/
        else:
            run_name += '_arms' + ('' if a.inputs == 'true' else f'_{a.inputs}in')
    out_root = RESULTS / run_name if run_name else None
    if a.probe_only:
        assert a.ckpt, '--probe-only needs --ckpt'
        if legacy:
            run_checkpoint(a.ckpt, a.data, a.splits, None, out_root, a.readout, probe_only=True)
        else:
            run_checkpoint_arms(a.ckpt, a.data, a.splits, None, out_root, arms, a.inputs,
                                probe_only=True, seeds=a.seeds)
        return
    bpath = RESULTS / 'data_baseline.json'
    if a.data_baseline or not bpath.exists():
        base = data_baseline(a.data, a.splits)
        print(json.dumps(jsonable(base['vs_expert']), indent=2))
    else:
        base = json.loads(bpath.read_text())
    if a.ckpt:
        if legacy:
            run_checkpoint(a.ckpt, a.data, a.splits, base, out_root, a.readout)
        else:
            run_checkpoint_arms(a.ckpt, a.data, a.splits, base, out_root, arms, a.inputs)
    if a.curve:
        if legacy:
            curve(a.run_dir, a.data, a.splits, base, out_root, a.readout)
        else:
            curve_arms(a.run_dir, a.data, a.splits, base, out_root, arms, a.inputs)


if __name__ == '__main__':
    main()
