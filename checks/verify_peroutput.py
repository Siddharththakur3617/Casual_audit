"""Check the per-output-mask JEPA (graph-prior arm P) without data: CPU, random weights, about a minute.

 1. Cart-pole change-parent mask matches checks/expert_graph (self only for theta, wbar; both actions
    parents of every state change; ctx row reads only ctx).
 2. Perturbing a slot that is not a parent of the change of j (i != j, every lag) leaves zhat_j exactly
    unchanged (with ctx_pool on, as in the configs).
 3. Perturbing x or vbar (not parents of their own change) moves zhat_x / zhat_vbar by exactly delta.
 4. Perturbing a parent changes the predicted change.
 5. Action masks: zeroing a slot's action parents makes it exactly insensitive to the actions
    (the Acrobot distractor case), while the other slots still respond.
 6. Dense control: every slot affects every output.
 7. per_output_hidden: auto matches the standard transformer predictor's parameter count (within 5%),
    and the per-output arm, dense control and corrupted controls have identical sizes.
 8. Wrong-graph controls: shuffled keeps the edge count; drop 2 / add 2 change exactly 2 edges; the
    expert reference used by the diagnostics is unchanged.
 9. Anchor targets (vbar = dx / DT, ...) and episode ids of windows are right; --set overrides parse.
10. probing/extract_jepa.load_model loads a per-output checkpoint with its exact graph, and a checkpoint
    without `model_class` still loads as JEPA.

Run: python -m checks.verify_peroutput
"""
from __future__ import annotations

import copy
import tempfile
from pathlib import Path

import numpy as np
import torch
import yaml

from models.jepa.model import JEPA
from models.jepa_peroutput import graph
from models.jepa_peroutput.model import MODEL_CLASS, JEPAPerOutput, transformer_params
from models.jepa_peroutput.train import anchor_targets, apply_overrides, episode_ids
from paths import MODEL_CONFIGS

CFG = MODEL_CONFIGS / 'graph_priors'


def model_cfg(path):
    return yaml.safe_load(Path(path).read_text())['model']


def main():
    torch.manual_seed(0)
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(('  PASS ' if cond else '  FAIL ') + msg)

    main_m = JEPAPerOutput(model_cfg(CFG / 'jepa_cartpole_peroutput.yaml')).eval()
    dense_m = JEPAPerOutput(model_cfg(CFG / 'jepa_cartpole_peroutput_dense.yaml')).eval()
    base_m = JEPA(model_cfg(MODEL_CONFIGS / 'jepa_cartpole.yaml')).eval()
    spec = main_m.spec
    names, sl = spec['names'], spec['slices']

    print('[1] change-parent mask (expert graph, strong + weak)')
    print(graph.describe(spec))
    want = {'x': {'vbar', 'theta', 'wbar', 'ctx'}, 'vbar': {'theta', 'wbar', 'ctx'},
            'theta': {'theta', 'wbar', 'ctx'}, 'wbar': {'theta', 'wbar', 'ctx'}, 'ctx': {'ctx'}}
    got = {names[j]: {names[i] for i in range(len(names)) if spec['cmask'][j, i]} for j in range(len(names))}
    check(got == want, 'parents of each change match checks/expert_graph.EXPERT')
    check(spec['amask'][:4].all() and not spec['amask'][4].any(), 'u_{t-1}, u_t parents of every state change, not of ctx')
    check([names[i] for i, j in spec['expert_non_edges'] if i == j] == ['x', 'vbar'], 'self non-parents are x and vbar')
    check(main_m.cfg.get('ctx_pool') is True, 'ctx_pool is on in the main config')

    b, H, d = 8, main_m.history_size, 192
    zh, ah = torch.randn(b, H, d), torch.randn(b, H, main_m.action_context_dim)

    def response(m, i=None, act=False):
        """(change of zhat, change of the predicted change) when slot i (or the actions) move at every lag."""
        with torch.no_grad():
            p0 = m.predict(zh, ah)[:, -1]
            z1, a1 = zh.clone(), ah.clone()
            if act:
                a1 += torch.randn_like(a1)
            else:
                z1[..., sl[i]] += torch.randn_like(z1[..., sl[i]])
            p1 = m.predict(z1, a1)[:, -1]
        dz = p1 - p0
        return dz, dz - (z1[:, -1] - zh[:, -1])

    print('[2-4] per-output predictor respects the graph')
    for i, ni in enumerate(names):
        dz, dc = response(main_m, i)
        for j, nj in enumerate(names):
            ez, ec = dz[:, sl[j]].abs().max().item(), dc[:, sl[j]].abs().max().item()
            if spec['cmask'][j, i]:
                check(ec > 1e-6, f'{ni:>5s} -> d{nj:<5s} parent:      change moves by {ec:.2e}')
            elif i != j:
                check(ez == 0.0, f'{ni:>5s} -> d{nj:<5s} not parent:  zhat moves by {ez:.2e}')
            else:
                check(ec <= 1e-5, f'{ni:>5s} -> d{nj:<5s} self, absent: change moves by {ec:.2e} (zhat by exactly delta)')

    print('[5] action masks')
    cfg5 = copy.deepcopy(main_m.cfg)
    am = np.asarray(cfg5['action_mask'])
    am[1] = 0                                        # vbar's change gets no action parent
    cfg5['action_mask'] = am.tolist()
    m5 = JEPAPerOutput(cfg5).eval()
    dz, _ = response(m5, act=True)
    check(dz[:, sl[1]].abs().max().item() == 0.0, 'slot without action parents ignores the actions exactly')
    check(dz[:, sl[0]].abs().max().item() > 1e-6, 'slot with action parents responds to the actions')

    print('[6] dense control: every slot affects every output')
    for i, ni in enumerate(names):
        _, dc = response(dense_m, i)
        check(all(dc[:, sl[j]].abs().max().item() > 1e-6 for j in range(len(names))), f'{ni} reaches every output')

    print('[7] parameter counts')
    cnt = lambda m: sum(p.numel() for p in m.predictor.parameters())
    n_main, n_dense, n_base = cnt(main_m), cnt(dense_m), cnt(base_m)
    print(f'    per-output {n_main:,} (hidden {main_m.cfg["per_output_hidden"]}) | dense {n_dense:,} | '
          f'standard transformer {n_base:,} (target {transformer_params(main_m.cfg):,})')
    check(n_main == n_dense, 'per-output arm and dense control have identical sizes')
    check(abs(n_main - n_base) / n_base <= 0.05, f'auto width within 5% of the standard predictor ({(n_main - n_base) / n_base:+.1%})')
    check(main_m.predict(zh[:, :2], ah[:, :2]).shape == (b, 2, d), 'shorter history works')
    obs, prev = torch.rand(2, 3, 128, 128), torch.rand(2, 3, 128, 128)
    with torch.no_grad():
        check(main_m.encode(obs, prev).shape == base_m.encode(obs, prev).shape, 'encode shape matches JEPA')

    print('[8] wrong-graph controls')
    base_cfg = model_cfg(CFG / 'jepa_cartpole_peroutput.yaml')
    n_true = int(np.asarray(main_m.cfg['change_mask'])[:4, :4].sum())
    cells = lambda c: np.asarray(c['change_mask'])[:4, :4].astype(bool)
    sh = graph.resolve(model_cfg(CFG / 'jepa_cartpole_peroutput_shuffled.yaml'))
    dr = graph.resolve(dict(base_cfg, graph_corrupt={'drop': 2, 'seed': 0}))
    ad = graph.resolve(dict(base_cfg, graph_corrupt={'add': 2, 'seed': 0}))
    ref = cells(main_m.cfg)
    check(cells(sh).sum() == n_true, f'shuffled keeps {n_true} state edges ({", ".join(sh["graph_log"]["shuffled"])})')
    check((ref & ~cells(dr)).sum() == 2 and (~ref & cells(dr)).sum() == 0, f'drop 2: {dr["graph_log"]["dropped"]}')
    check((~ref & cells(ad)).sum() == 2 and (ref & ~cells(ad)).sum() == 0, f'add 2: {ad["graph_log"]["added"]}')
    check(all(c['expert_change_mask'] == main_m.cfg['expert_change_mask'] for c in (sh, dr, ad)),
          'diagnostics keep the uncorrupted expert graph')
    check(sum(p.numel() for p in JEPAPerOutput(dr).predictor.parameters()) == n_main, 'corrupted control has the same size')

    print('[9] anchor targets, episode ids, --set')
    import envs
    env = envs.use('cartpole')
    st = torch.tensor([[[0.0, 0, 0.1, 0], [0.04, 0, 0.12, 0], [0.10, 0, 0.11, 0]]])   # x, xdot, theta, thetadot
    y = anchor_targets(st, env)
    exp = torch.tensor([[[0.04, 0.04 / env.DT, 0.12, 0.02 / env.DT], [0.10, 0.06 / env.DT, 0.11, -0.01 / env.DT]]])
    check(y.shape == (1, 2, 4) and torch.allclose(y, exp, atol=1e-5), 'cart-pole anchor targets x, vbar, theta, wbar')
    starts = np.array([[0, 0, 0], [1, 1, 1], [9, 8, 0], [10, 9, 1], [11, 10, 2], [20, 18, 0]])
    check(episode_ids(starts).tolist() == [0, 0, 1, 1, 1, 2], 'episode ids from window starts')
    c = apply_overrides({'model': {'graph_corrupt': None}, 'training': {'lambda_ctx': 0.1}},
                        ['training.lambda_ctx=1.0', 'model.graph_corrupt={drop: 2, seed: 0}'])
    check(c['training']['lambda_ctx'] == 1.0 and c['model']['graph_corrupt'] == {'drop': 2, 'seed': 0}, '--set overrides')

    print('[10] load_model hook')
    from probing.extract_jepa import load_model
    with tempfile.TemporaryDirectory() as tmp:
        p1, p2 = Path(tmp) / 'peroutput.pt', Path(tmp) / 'jepa.pt'
        m_dr = JEPAPerOutput(dr).eval()
        torch.save({'model_type': 'jepa', 'model_class': MODEL_CLASS, 'model': m_dr.state_dict(),
                    'model_config': m_dr.cfg, 'epoch': 0}, p1)
        torch.save({'model_type': 'jepa', 'model': base_m.state_dict(), 'model_config': base_m.cfg, 'epoch': 0}, p2)
        m1, _, _ = load_model(p1, torch.device('cpu'))
        m2, _, _ = load_model(p2, torch.device('cpu'))
        check(isinstance(m1, JEPAPerOutput), 'per-output checkpoint loads as JEPAPerOutput')
        check(np.array_equal(m1.spec['cmask'], m_dr.spec['cmask']), 'loaded model keeps the saved (corrupted) graph')
        check(type(m2) is JEPA, 'checkpoint without model_class loads as JEPA')
        with torch.no_grad():
            check(torch.allclose(m1.predict(zh, ah), m_dr.predict(zh, ah)), 'loaded weights give identical predictions')

    print('ALL PASS' if ok else 'SOME CHECKS FAILED')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
