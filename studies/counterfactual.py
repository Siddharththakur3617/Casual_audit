"""Counterfactual evaluation of the recovered graph (docs/counterfactual_study.md).

  python -m studies.counterfactual --data datasets/cc_phys --ckpt runs/jepa_cc_phys/checkpoints/checkpoint_epoch0200.pt
  python -m studies.counterfactual --data datasets/ac_phys --ckpt runs/jepa_ac_phys/checkpoints/checkpoint_epoch0200.pt
  add --kappa 0.5 (sensitivity) or --rollout (multi-step phase, horizon 3, u_t and u_{t-1} only)

For each query window of the test split and each intervention (counterfactual/queries.py), the change at t
(or the state change up to t+k, --rollout) is answered by:
  truth      the simulator at the episode physics
  ceiling    the encoder on the true counterfactual frames, read by the differential probe
  model      the JEPA predictor on the counterfactual history and actions, read by the probe
  SCMs       graph-restricted MLPs (counterfactual/scm.py): expert graph on true changes, recovered graph on
             the model's read-out changes, and diagnostics (see SCM_VARIANTS)
Pre-declared rules (not tuned): kappa 0.25 main / 0.5 sensitivity, test split, at most 3000 windows (seed 0),
testable targets = train-probe encoder R2 >= 0.5, model-interventional edge = CE lower 95% bound > readout floor
(max over expert-absent testable cells of the ceiling's upper bound), 1000 episode-bootstrap resamples, 3 SCM seeds.

Caches (resumable): runs/<run>/counterfactual/ (probe, queries, per-chunk latents).
Outputs: results/studies/counterfactual_<data>_<run>/epoch####_k###[_rollout]/ (results/_smoke/studies/ for smoke runs).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import envs
import paths
from counterfactual import metrics as M
from counterfactual import queries as q
from counterfactual.model_jepa import JEPARunner
from counterfactual.scm import SCMBank
from evaluate import compare
from probing import fit_probes
from run_all import ckpt_epoch, jsonable, load_or_extract

SCM_VARIANTS = {   # name: (parents, data source, physics as known input)
    'expert_true': ('expert', 'true', False),              # the expert graph's answer
    'recovered_model': ('recovered', 'model', False),      # the recovered graph's answer (describes the model)
    'recovered_true': ('recovered', 'true', False),        # recovered parents on true data: cost of graph errors alone
    'full_model': ('full', 'model', False),                # no pruning, on the model: cost of pruning
    'full_true': ('full', 'true', False),                  # observational fit, physics hidden: confounded
    'expert_true_phys': ('expert', 'true', True),
    'recovered_model_phys': ('recovered', 'model', True),
    'full_true_phys': ('full', 'true', True),
}
ROLLOUT_K = 3


# ---------------------------------------------------------------- inputs
def find_graph(run_dir, data_dir, epoch, rule='null_cal_fwer'):
    name = Path(data_dir).name
    tag = '' if name == 'cc' else f'_{name}'
    root = paths.model_results(run_dir) / f'{Path(run_dir).name}{tag}_diff_{rule}'
    for p in (root / f'epoch{epoch:04d}' / 'results.json', root / 'curve_s20' / f'epoch{epoch:04d}' / 'results.json'):
        if p.exists():
            return p
    return None


def out_dir(run_dir, data_dir, epoch, kappa, rollout, split):
    base = (paths.SMOKE_RESULTS / 'studies' if Path(run_dir).resolve().parent == paths.SMOKE_RUNS.resolve()
            else paths.STUDIES)
    tag = f'epoch{epoch:04d}_k{int(round(kappa * 100)):03d}' + ('_rollout' if rollout else '') + \
          ('' if split == 'test' else f'_{split}')
    return base / f'counterfactual_{Path(data_dir).name}_{Path(run_dir).name}' / tag


def probe_for(ckpt, data_dir, cache):
    """Differential probe of the train extraction (seed 0, as in the graph run), with its fold nets."""
    if cache.exists():
        return torch.load(cache, weights_only=False)
    E = load_or_extract(ckpt, data_dir, 'train')
    t0 = time.time()
    P = fit_probes.fit_differential(E, seed=0, shuffle_control=False, return_nets=True)
    keep = {'probe': P['probe'], 'dd_pred': P['dd_pred'], 'report': P['report'], 'action_effect': P['action_effect'],
            'inputs': E['inputs'], 'targets': E['targets'], 'physics': E['physics'], 'episode': E['episode']}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(keep, cache)
    print(f'  [probe] fitted ({time.time() - t0:.0f}s) -> {cache}', flush=True)
    return keep


# ---------------------------------------------------------------- latents (render + encode, chunked)
def latents(Q, runner, renderer, cdir, data_dir, split, K, chunk=200):
    """Per variant: zenc (n_var, n, K+1, d) = z_t..z_{t+K} from rendered frames, zhat (n_var, n, K, d)."""
    n_var, n = Q['valid'].shape
    cdir.mkdir(parents=True, exist_ok=True)
    verify = None
    ZE, ZH = [], []
    for c0 in range(0, n, chunk):
        f = cdir / f'chunk_{c0:05d}.npz'
        if f.exists():
            with np.load(f) as d:
                ZE.append(d['zenc']); ZH.append(d['zhat'])
                if 'verify' in d.files:
                    verify = json.loads(str(d['verify']))
            continue
        t0 = time.time()
        idx = np.arange(c0, min(n, c0 + chunk))
        Sf, Pf = Q['S'][0, idx], Q['phys'][0, idx]
        ref = renderer.windows(Sf, Pf)
        ze, zh = [], []
        for v in range(n_var):
            S, P = Q['S'][v, idx].copy(), Q['phys'][v, idx].copy()
            bad = ~Q['valid'][v, idx]
            S[bad], P[bad] = Sf[bad], Pf[bad]                       # invalid rows: not rendered, ignored later
            frames = ref if v == 0 else renderer.windows(S, P, Sf, Pf, ref)
            Z = runner.encode_windows(frames)
            ze.append(Z[:, q.HIST:])
            zh.append(runner.rollout(Z, Q['U'][v, idx], K))
        extra = {}
        if c0 == 0:                                                 # factual re-render vs the stored frames
            obs = q.load_obs(data_dir, split, Q['episode'][idx], Q['t'][idx], horizon=K)
            dpx = np.abs(obs.astype(int) - ref.astype(int))
            Zs = runner.encode_windows(obs)
            # counterfactual frames: the renderer used vs the sim's own renderer, on a random sample
            rng = np.random.default_rng(0)
            vv = rng.integers(1, n_var, 64)
            ww = rng.integers(0, len(idx), 64)
            ff = rng.integers(0, Q['S'].shape[2], 64)
            st, ph = Q['S'][vv, idx[ww], ff], Q['phys'][vv, idx[ww]]
            okv = Q['valid'][vv, idx[ww]]
            q._init_render(envs.current().NAME, renderer.size)
            sim_frames = q._render_block((st[okv], ph[okv]))
            verify = {'pixel_max_abs_diff': int(dpx.max()), 'pixel_mean_abs_diff': float(dpx.mean()),
                      'latent_max_abs_diff': float(np.nanmax(np.abs(Zs[:, 1:] - runner.encode_windows(ref)[:, 1:]))),
                      'latent_sd': float(np.nanstd(Zs[:, 1:])), 'windows': int(len(idx)),
                      'cf_frames_vs_sim_renderer_max_abs_diff':
                          int(np.abs(renderer(st[okv], ph[okv]).astype(int) - sim_frames.astype(int)).max())
                          if okv.any() else None}
            extra['verify'] = json.dumps(verify)
        ze, zh = np.stack(ze), np.stack(zh)
        np.savez(f, zenc=ze, zhat=zh, **extra)
        ZE.append(ze); ZH.append(zh)
        print(f'    [latents] windows {c0}-{idx[-1]} of {n} ({time.time() - t0:.0f}s)', flush=True)
    return np.concatenate(ZE, 1), np.concatenate(ZH, 1), verify


def readouts(probe, zenc, zhat):
    """Probe-read changes: ceiling (n_var, n, K, n_tg) from zenc differences, model from the predicted chain."""
    n_var, n, K = zhat.shape[:3]
    prev = np.concatenate([zenc[:, :, :1], zhat[:, :, :-1]], axis=2)
    dz_model = (zhat - prev).reshape(-1, zhat.shape[-1])
    dz_ceil = (zenc[:, :, 1:] - zenc[:, :, :-1]).reshape(-1, zenc.shape[-1])
    r = lambda dz: probe(dz).reshape(n_var, n, K, -1)  # noqa: E731
    return r(dz_ceil), r(dz_model)


def scm_answers(bank, Q, K, env):
    """{name: (n_var, n, K, n_tg)} SCM changes; K > 1 rolls the SCM forward (predicted state fed back, factual actions)."""
    am = np.zeros(len(env.STATE_NAMES), bool)
    am[:] = q.angle_mask(env)
    k = len(env.STATE_NAMES)
    n_var, n = Q['valid'].shape
    out = {name: np.zeros((n_var, n, K, len(env.TARGET_NAMES))) for name in bank.names}
    for v in range(n_var):
        o = Q['inputs'][v, :, 0]
        U = Q['U'][v]
        first = bank.predict(o, Q['phys'][v])                     # step 0: every SCM sees the same inputs
        state = {}
        for name in bank.names:
            out[name][v, :, 0] = first[name]
            state[name] = o[:, :k] + first[name]
        for step in range(1, K):
            acts = np.column_stack([U[:, q.HIST + step], U[:, q.HIST + step - 1], U[:, q.HIST + step - 2]])
            for name in bank.names:
                state[name][:, am] = (state[name][:, am] + np.pi) % (2 * np.pi) - np.pi
                x = np.column_stack([state[name], acts])
                d = bank.predict(x, Q['phys'][v])[name]
                out[name][v, :, step] = d
                state[name] = x[:, :k] + d
    return out


# ---------------------------------------------------------------- analysis
def fmt(x, nd=3):
    return '–' if x is None or (isinstance(x, float) and not np.isfinite(x)) else f'{x:.{nd}f}'


def analyse_one_step(env, Q, src, sd_t, kappa, A, tiers, effect, kept, jac, testable, action_effect_model, n_boot):
    names_in, names_tg = env.INPUT_NAMES, env.TARGET_NAMES
    groups = M.node_groups(Q)
    E = {s: M.effects(D[:, :, 0], Q, kappa, sd_t, groups) for s, D in src.items()}
    ep = Q['episode']
    nodes, phys_nodes = list(names_in), list(env.PHYS_KEYS)
    CE = {s: np.array([M.ce(E[s][nd]) for nd in nodes]) for s in E}
    CEp = {s: np.array([M.ce(E[s][nd]) for nd in phys_nodes]) for s in E}
    ACE = {s: np.array([M.ace(E[s][nd]) for nd in nodes]) for s in E}
    boot = {s: [M.bootstrap_ce(E[s][nd], ep, n_boot, seed=i) for i, nd in enumerate(nodes)]
            for s in ('truth', 'model', 'ceiling')}
    LO = {s: np.array([b[0] for b in boot[s]]) for s in boot}
    HI = {s: np.array([b[1] for b in boot[s]]) for s in boot}
    tcols = [j for j in range(len(names_tg)) if testable[j]]
    allcols = list(range(len(names_tg)))
    res = {'nodes': nodes, 'targets': names_tg, 'testable': testable, 'kappa': kappa}

    # pipeline sanity (pre-declared): truth is exactly 0 on absent cells and ranks like the expert effect sizes
    absent, present = tiers == 'absent', tiers != 'absent'
    sp = M.spearmanr(CE['truth'][present], effect[present]).statistic
    res['sanity'] = {'truth_ce_max_on_absent': float(CE['truth'][absent].max()) if absent.any() else None,
                     'truth_vs_expert_effect_spearman_present': float(sp),
                     'truth_vs_expert_effect_max_abs_diff': float(np.abs(CE['truth'] - effect).max()),
                     'pass': bool((CE['truth'][absent].max() if absent.any() else 0) <= 1e-6 and sp >= 0.95)}

    pairs = [('expert_true', 'truth', allcols), ('model', 'truth', tcols), ('ceiling', 'truth', tcols),
             ('recovered_model', 'model', tcols), ('recovered_model', 'truth', tcols),
             ('recovered_true', 'truth', allcols), ('full_model', 'model', tcols), ('full_true', 'truth', allcols)]
    res['agreement'] = {}
    for a, b, cols in pairs:
        g = M.matrix_agreement(E[a], E[b], nodes, cols)
        res['agreement'][f'{a}_vs_{b}'] = {'ce_spearman': g['ce_spearman'], 'pooled_cf_r2': g['pooled_cf_r2'],
                                            'cells': {f'{nd}->{names_tg[j]}': v for (nd, j), v in g['cells'].items()}}

    # the model's interventional graph and the discovery scored against it
    ab_t = absent[:, tcols]
    floor = float(np.nanmax(HI['ceiling'][:, tcols][ab_t])) if ab_t.any() else float('nan')
    model_edge = LO['model'] > floor
    rows = []
    T = np.zeros_like(model_edge)
    T[:, tcols] = True
    kept_b, A_b = np.asarray(kept, bool), np.asarray(A, bool)
    for i, nd in enumerate(nodes):
        for j in tcols:
            m_, r_, x_ = bool(model_edge[i, j]), bool(kept_b[i, j]), bool(A_b[i, j])
            disc = ('agree (edge)' if m_ and r_ else 'agree (no edge)' if not m_ and not r_
                    else 'discovery miss' if m_ else 'discovery false alarm')
            rows.append({'cell': f'{nd}->{names_tg[j]}', 'tier': str(tiers[i, j]), 'recovered': r_, 'model_uses': m_,
                         'expert': x_, 'discovery': disc, 'model_vs_expert': 'ok' if m_ == x_ else
                         ('model uses an absent input' if m_ else 'model ignores a true input'),
                         'ce_truth': float(CE['truth'][i, j]), 'ce_model': float(CE['model'][i, j]),
                         'ce_model_lo': float(LO['model'][i, j]), 'ce_ceiling': float(CE['ceiling'][i, j]),
                         'ce_recovered_scm': float(CE['recovered_model'][i, j]),
                         'ce_expert_scm': float(CE['expert_true'][i, j]),
                         'loco_jacobian': None if jac is None else float(jac[i][j])})
    res['readout_floor'] = floor
    res['discovery_vs_model'] = {**M.prf(model_edge[T], kept_b[T]), 'cells': rows}
    res['model_vs_expert'] = M.prf(A_b[T], model_edge[T])
    res['recovered_vs_expert'] = M.prf(A_b[T], kept_b[T])
    res['ce'] = {s: CE[s] for s in CE}
    res['ace'] = {s: ACE[s] for s in ACE}
    res['ce_lo'], res['ce_hi'] = LO, HI
    res['expert_effect_size'] = effect
    res['physics'] = {'nodes': phys_nodes, 'ce': {s: CEp[s] for s in CEp},
                      'cf_r2_model_vs_truth': [[M.agreement(E['model'][p][:, j], E['truth'][p][:, j])['cf_r2']
                                                for j in allcols] for p in phys_nodes]}
    ia = [names_in.index(n) for n in names_in if n.startswith('u_')]
    res['action_effect_check'] = {'rows': [names_in[i] for i in ia], 'ce_model_consistent_history': CE['model'][ia],
                                  'action_effect_model_history_held': action_effect_model}
    res['valid_fraction'] = {str(v): float(Q['valid'][k].mean()) for k, v in enumerate(Q['variants'])}
    return res, E


def table(header, rows):
    return '\n'.join(['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
                     + ['| ' + ' | '.join(str(c) for c in r) + ' |' for r in rows]) + '\n'


def write_one_step(out, env, res, meta):
    names_tg = env.TARGET_NAMES
    tcols = [j for j, t in enumerate(res['testable']) if t]
    S = []
    S.append(f'# Counterfactual study: {meta["run"]} epoch {meta["epoch"]} on {meta["data"]} ({meta["split"]}), '
             f'kappa {res["kappa"]}\n\n{meta["n_windows"]} windows from {meta["n_episodes"]} episodes. '
             f'Recovered graph: `{meta["graph"]}`. Testable targets: '
             f'{", ".join(t for t, ok in zip(names_tg, res["testable"]) if ok)}.\n')
    s = res['sanity']
    S.append('## Pipeline sanity (pre-declared)\n\n'
             f'- truth CE on expert-absent cells: max {s["truth_ce_max_on_absent"]:.1e}'
             f' (must be <= 1e-6)\n- Spearman(truth CE, expert effect size) on present cells: '
             f'{fmt(s["truth_vs_expert_effect_spearman_present"])} (must be >= 0.95)\n- **{"PASS" if s["pass"] else "FAIL"}**\n'
             f'- factual re-render vs stored frames: {meta.get("verify")}\n'
             f'- test-split probe readout R2 (train probe, unseen episodes): encoder {meta["test_r2_encoder"]}, '
             f'predictor {meta["test_r2_predictor"]}\n')
    rows = []
    for k, g in res['agreement'].items():
        rows.append([k.replace('_vs_', ' vs '), fmt(g['ce_spearman']), fmt(g['pooled_cf_r2'])])
    S.append('## Agreement between sources\n\nCE Spearman = rank agreement of the mean |effect| matrices; pooled CF-R2 = '
             'unit-level agreement of the effects (1 = identical window by window).\n\n'
             + table(['comparison (source vs reference)', 'CE Spearman', 'pooled CF-R2'], rows))
    hdr = ['input'] + names_tg
    for src, title in (('truth', 'truth (simulator)'), ('expert_true', 'expert-graph SCM'),
                       ('recovered_model', 'recovered-graph SCM (on model read-outs)'), ('model', 'model (JEPA)'),
                       ('ceiling', 'ceiling (encoder on true counterfactual frames)')):
        S.append(f'### CE: {title}\n\n' + table(hdr, [[nd] + [fmt(res['ce'][src][i, j]) for j in range(len(names_tg))]
                                                       for i, nd in enumerate(res['nodes'])]))
    S.append(f'Expert effect size (expert_graph_final), for reference:\n\n' + table(
        hdr, [[nd] + [fmt(res['expert_effect_size'][i, j]) for j in range(len(names_tg))] for i, nd in enumerate(res['nodes'])]))
    report = '\n'.join(S)

    dv = res['discovery_vs_model']
    D = [f'# Discovery vs the model\'s interventional graph\n\nReadout floor (max upper 95% bound of the ceiling CE '
         f'over expert-absent testable cells): {fmt(res["readout_floor"])}. A cell is a model-interventional edge if '
         f'the lower 95% bound of the model CE is above it.\n\n'
         f'- recovered graph vs model-interventional graph: precision {fmt(dv["precision"])}, recall {fmt(dv["recall"])}, '
         f'F1 {fmt(dv["f1"])} (tp {dv["tp"]}, fp {dv["fp"]}, fn {dv["fn"]})\n'
         f'- model-interventional graph vs expert: F1 {fmt(res["model_vs_expert"]["f1"])} '
         f'(fp {res["model_vs_expert"]["fp"]}, fn {res["model_vs_expert"]["fn"]})\n'
         f'- recovered graph vs expert (testable cells): F1 {fmt(res["recovered_vs_expert"]["f1"])}\n\n']
    D.append(table(['cell', 'tier', 'recovered', 'model uses', 'discovery', 'model vs expert', 'CE truth', 'CE model',
                    'CE ceiling', 'CE recovered SCM', 'LOCO Jacobian'],
                   [[c['cell'], c['tier'], 'yes' if c['recovered'] else '', 'yes' if c['model_uses'] else '',
                     c['discovery'], c['model_vs_expert'], fmt(c['ce_truth']), fmt(c['ce_model']), fmt(c['ce_ceiling']),
                     fmt(c['ce_recovered_scm']), fmt(c['loco_jacobian'])] for c in dv['cells']]))
    ac = res['action_effect_check']
    D.append('\n## Action effects: consistent history vs history held fixed\n\nModel CE with the history re-solved '
             '(this study) vs the existing finite difference with the history latents held (run_all action_effect_model).\n\n'
             + table(['action'] + [f'{names_tg[j]} (cf / held)' for j in tcols],
                     [[r] + [f'{fmt(ac["ce_model_consistent_history"][k][j])} / {fmt(ac["action_effect_model_history_held"][k][j])}'
                             for j in tcols] for k, r in enumerate(ac['rows'])]))
    disc = '\n'.join(D)

    C = ['# Confounding: interventional truth vs observational fits\n\nPhysics is fixed within an episode, so the truth '
         'and the model CE are interventional. `full_true` is an all-input fit on true data with physics hidden (what an '
         'observational estimator sees); `full_true_phys` adds the physics as known input. Cells flagged where the '
         'observational fit or the model differs from the truth by more than 0.05.\n\n']
    rows = []
    for i, nd in enumerate(res['nodes']):
        for j in tcols:
            t, fo, fp_, m = (res['ce'][s][i, j] for s in ('truth', 'full_true', 'full_true_phys', 'model'))
            flag = ', '.join(x for x, c in (('obs. bias', abs(fo - t) > 0.05), ('model', abs(m - t) > 0.05)) if c)
            rows.append([f'{nd}->{names_tg[j]}', fmt(t), fmt(fo), fmt(fp_), fmt(res['ce']['expert_true_phys'][i, j]),
                         fmt(res['ce']['recovered_model'][i, j]), fmt(m), flag])
    C.append(table(['cell', 'truth', 'full_true (phys hidden)', 'full_true_phys', 'expert_true_phys', 'recovered SCM',
                    'model', 'flag'], rows))
    conf = '\n'.join(C)

    ph = res['physics']
    Pp = ['# Physics (confounder) interventions\n\nd_t and the actions are held fixed; only the episode physics changes. '
          'Graph SCMs without physics answer 0 by construction. Rendered parameters change the image directly; the '
          'others are visible only through the dynamics of the 5-frame history.\n']
    rendered = {'cartpole': ('l',), 'acrobot': ('l1', 'l2')}[env.NAME]
    for src in ('truth', 'model', 'ceiling', 'expert_true_phys', 'recovered_model_phys', 'full_true_phys'):
        Pp.append(f'\n### CE: {src}\n\n' + table(['physics'] + names_tg,
                                                 [[p + (' (rendered)' if p in rendered else '')] +
                                                  [fmt(ph['ce'][src][i, j]) for j in range(len(names_tg))]
                                                  for i, p in enumerate(ph['nodes'])]))
    Pp.append('\n### CF-R2, model vs truth\n\n' + table(['physics'] + names_tg,
                                                       [[p] + [fmt(v) for v in row] for p, row in
                                                        zip(ph['nodes'], ph['cf_r2_model_vs_truth'])]))
    phys = '\n'.join(Pp)

    (out / 'discovery_vs_model.md').write_text(disc, encoding='utf-8')
    (out / 'confounding.md').write_text(conf, encoding='utf-8')
    (out / 'physics.md').write_text(phys, encoding='utf-8')
    (out / 'report.md').write_text('\n\n'.join([report, disc, conf, phys]), encoding='utf-8')


def heatmaps(out, env, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    srcs = [('truth', 'truth'), ('expert_true', 'expert SCM'), ('model', 'model'), ('ceiling', 'ceiling'),
            ('recovered_model', 'recovered SCM')]
    vmax = max([float(np.nanmax(res['ce'][s])) for s, _ in srcs if np.isfinite(res['ce'][s]).any()] + [1e-6])
    fig, axes = plt.subplots(1, len(srcs), figsize=(3.1 * len(srcs), 0.42 * len(res['nodes']) + 1.6),
                             constrained_layout=True, sharey=True)
    dv = {c['cell']: c for c in res['discovery_vs_model']['cells']}
    for ax, (s, title) in zip(axes, srcs):
        C = res['ce'][s]
        im = ax.imshow(C, cmap='Blues', vmin=0, vmax=vmax, aspect='auto')
        for i in range(C.shape[0]):
            for j in range(C.shape[1]):
                if C[i, j] >= 0.05:
                    ax.text(j, i, f'{C[i, j]:.2f}', ha='center', va='center', fontsize=7,
                            color='white' if C[i, j] > 0.6 * vmax else '#222')
                if not res['testable'][j]:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, hatch='///', ec='#bbb', lw=0))
        if s == 'recovered_model':
            for key, c in dv.items():
                if c['recovered']:
                    i, j = res['nodes'].index(key.split('->')[0]), env.TARGET_NAMES.index(key.split('->')[1])
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec='#d1495b', lw=1.5))
        ax.set_xticks(range(len(env.TARGET_NAMES)), env.TARGET_NAMES, rotation=45, fontsize=8)
        ax.set_yticks(range(len(res['nodes'])), res['nodes'], fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=axes, shrink=0.8, label='mean |effect| (expert effect-size unit)')
    fig.suptitle(f'Counterfactual effects (kappa {res["kappa"]}); red box = kept by the recovered graph; hatched = untestable target',
                 fontsize=9)
    fig.savefig(out / 'ce_heatmaps.png', dpi=120)
    plt.close(fig)


def analyse_rollout(env, Q, src, kappa, testable, n_boot):
    """Effects on the cumulative state change d_{t+k} - d_t, k = 1..K, normalised by the sd of the factual true value."""
    cum = {s: np.cumsum(D, axis=2) for s, D in src.items()}
    sd_k = np.maximum(cum['truth'][0][Q['valid'][0]].std(0), 1e-12)              # (K, n_tg)
    groups = M.node_groups(Q)
    E = {s: M.effects(C, Q, kappa, sd_k, groups) for s, C in cum.items()}
    K = sd_k.shape[0]
    res = {'nodes': list(groups), 'K': K, 'targets': env.TARGET_NAMES, 'testable': testable, 'ce': {}, 'cf_r2': {}}
    for s in E:
        res['ce'][s] = {nd: M.ce(E[s][nd]) for nd in groups}                      # (K, n_tg)
    for a, b in (('model', 'truth'), ('ceiling', 'truth'), ('expert_true', 'truth'), ('recovered_model', 'model'),
                 ('recovered_model', 'truth')):
        res['cf_r2'][f'{a}_vs_{b}'] = {nd: [[M.agreement(E[a][nd][:, k, j], E[b][nd][:, k, j])['cf_r2']
                                             for j in range(len(env.TARGET_NAMES))] for k in range(K)] for nd in groups}
    return res


def write_rollout(out, env, res, meta):
    names = env.TARGET_NAMES
    L = [f'# Multi-step counterfactuals: {meta["run"]} epoch {meta["epoch"]} on {meta["data"]} ({meta["split"]}), '
         f'kappa {meta["kappa"]}\n\nEffect of do(action) on the state change d_(t+k) - d_t (k = 1..{res["K"]}, factual '
         'later actions), normalised by the sd of the true factual change at that horizon. Model: autoregressive latent '
         'rollout read by the probe step by step; SCMs: rolled forward on their own predictions. Untestable targets '
         'are marked *.\n']
    for nd in res['nodes']:
        L.append(f'\n## do({nd})\n')
        for k in range(res['K']):
            rows = []
            for s in ('truth', 'model', 'ceiling', 'expert_true', 'recovered_model'):
                rows.append([s] + [fmt(res['ce'][s][nd][k, j]) for j in range(len(names))])
            for p in ('model_vs_truth', 'recovered_model_vs_model'):
                rows.append([f'CF-R2 {p.replace("_vs_", " vs ")}'] + [fmt(v) for v in res['cf_r2'][p][nd][k]])
            L.append(f'\nk = {k + 1}\n\n' + table(['source'] + [n + ('' if t else '*') for n, t in zip(names, res['testable'])],
                                                 rows))
    (out / 'rollout.md').write_text('\n'.join(L), encoding='utf-8')


# ---------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--kappa', type=float, default=0.25)
    p.add_argument('--windows', type=int, default=3000)
    p.add_argument('--split', default='test', choices=('train', 'val', 'test'))
    p.add_argument('--rollout', action='store_true', help=f'multi-step phase: horizon {ROLLOUT_K}, u_t and u_(t-1)')
    p.add_argument('--graph', default='auto', help="'auto' (the run's null_cal_fwer results.json), 'expert' "
                                                   "(smoke tests only) or a results.json path")
    p.add_argument('--chunk', type=int, default=200)
    p.add_argument('--workers', type=int, default=None)
    p.add_argument('--scm-seeds', type=int, default=3)
    p.add_argument('--scm-steps', type=int, default=1500)
    p.add_argument('--boot', type=int, default=M.N_BOOT)
    a = p.parse_args()
    t_start = time.time()
    env = envs.use_data(a.data)
    ckpt = Path(a.ckpt)
    run_dir = ckpt.parent.parent
    epoch = ckpt_epoch(ckpt)
    K = ROLLOUT_K if a.rollout else 1
    out = out_dir(run_dir, a.data, epoch, a.kappa, a.rollout, a.split)
    out.mkdir(parents=True, exist_ok=True)
    cache = run_dir / 'counterfactual'
    print(f'[counterfactual] {run_dir.name} epoch {epoch} on {a.data} ({a.split}), kappa {a.kappa}, K {K} -> {out}', flush=True)

    # graphs
    A, tiers, effect = compare.load_expert(a.data)
    if a.graph == 'expert':
        gpath, kept, jac = 'expert graph (smoke test stand-in)', A.astype(bool), None
    else:
        gp = find_graph(run_dir, a.data, epoch) if a.graph == 'auto' else Path(a.graph)
        if gp is None or not gp.exists():
            raise SystemExit('recovered graph not found: run run_all.py --prune null_cal_fwer for this checkpoint first, '
                             'or pass --graph <results.json>')
        R = json.loads(gp.read_text())
        kept, jac = np.asarray(R['graphs']['predictor']['kept'], bool), R['graphs']['predictor'].get('jacobian_effect')
        gpath = str(gp.relative_to(paths.HERE)) if gp.resolve().is_relative_to(paths.HERE) else str(gp)

    # probe and train statistics
    P = probe_for(ckpt, a.data, cache / f'probe_{ckpt.stem}.pt')
    in_sd, sd_t = P['inputs'].std(0), P['targets'].std(0)
    testable = [bool(v) for v in P['report'][fit_probes.testable_key()]]

    # queries
    qtag = f'{Path(a.data).name}_{a.split}_k{int(round(a.kappa * 100)):03d}_h{K}_n{a.windows}'
    qf = cache / f'queries_{qtag}.npz'
    if qf.exists():
        with np.load(qf) as d:
            Q = {k: d[k] for k in d.files}
    else:
        t0 = time.time()
        W = q.load_windows(a.data, a.split, a.windows, seed=0, horizon=K)
        Q = q.build_queries(W, a.kappa, in_sd, rollout=a.rollout, env=env)
        qf.parent.mkdir(parents=True, exist_ok=True)
        np.savez(qf, **Q)
        print(f'  [queries] {Q["valid"].shape[1]} windows x {len(Q["variants"])} variants ({time.time() - t0:.0f}s)', flush=True)
    qcheck = q.check_queries(Q, in_sd, env)
    fs_err = q.check_frame_state(Q['S'][0], Q['U'][0], env)

    # model latents
    runner = JEPARunner(ckpt, a.data)
    renderer = q.make_renderer(env, workers=a.workers)
    try:
        zenc, zhat, verify = latents(Q, runner, renderer, cache / f'latents_{ckpt.stem}_{qtag}', a.data, a.split, K, a.chunk)
    finally:
        renderer.close()
    ceil, model = readouts(P['probe'], zenc, zhat)

    # SCMs
    t0 = time.time()
    full = np.ones_like(A, bool)
    par = {'expert': A.astype(bool), 'recovered': kept, 'full': full}
    ys = {'true': P['targets'], 'model': P['dd_pred']}
    bank = SCMBank({n: (par[g], ys[s], ph) for n, (g, s, ph) in SCM_VARIANTS.items()}, P['inputs'], P['physics'],
                   seeds=range(a.scm_seeds), steps=a.scm_steps)
    scm = scm_answers(bank, Q, K, env)
    print(f'  [scm] {len(SCM_VARIANTS)} variants fitted and queried ({time.time() - t0:.0f}s)', flush=True)

    src = {'truth': Q['targets'], 'ceiling': ceil, 'model': model, **scm}
    vf = Q['valid'][0]
    meta = {'run': run_dir.name, 'epoch': epoch, 'data': a.data, 'split': a.split, 'kappa': a.kappa, 'graph': gpath,
            'n_windows': int(vf.sum()), 'n_episodes': int(len(np.unique(Q['episode']))), 'verify': verify,
            'query_check_max_sd_error': max(v for v in qcheck.values() if v is not None),
            'frame_state_check': fs_err, 'scm_train_r2': bank.train_r2,
            'test_r2_encoder': np.round(fit_probes.r2(Q['targets'][0, vf, 0], ceil[0, vf, 0]), 3).tolist(),
            'test_r2_predictor': np.round(fit_probes.r2(Q['targets'][0, vf, 0], model[0, vf, 0]), 3).tolist(),
            'train_probe_report': P['report']}
    if a.rollout:
        res = analyse_rollout(env, Q, src, a.kappa, testable, a.boot)
        write_rollout(out, env, res, meta)
    else:
        res, _ = analyse_one_step(env, Q, src, sd_t, a.kappa, A, tiers, effect, kept, jac, testable,
                                  P['action_effect'], a.boot)
        write_one_step(out, env, res, meta)
        heatmaps(out, env, res)
        print(f'  sanity: {res["sanity"]}', flush=True)
        dv = res['discovery_vs_model']
        print(f'  recovered graph vs model-interventional graph: F1 {dv["f1"]:.3f} (fp {dv["fp"]}, fn {dv["fn"]}); '
              f'floor {res["readout_floor"]:.3f}', flush=True)
    meta['seconds'] = round(time.time() - t_start)
    (out / 'results.json').write_text(json.dumps(jsonable({'meta': meta, 'results': res}), indent=2))
    print(f'  -> {out} ({meta["seconds"]}s)', flush=True)


if __name__ == '__main__':
    main()
