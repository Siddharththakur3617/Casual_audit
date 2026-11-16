"""Check the graph-prior JEPA wiring without data (CPU, random weights, ~seconds).

1. The cart-pole slot mask matches checks/expert_graph.py and is transitively closed.
2. GraphMaskedPredictor: perturbing a non-parent slot (at every history position) leaves the
   predicted slot exactly unchanged; perturbing a parent or self slot changes it.
3. encode/predict shapes of the graph-masked JEPA equal those of the standard JEPA.
4. The Jacobian-penalty estimator gives ~0 on non-edges of the masked predictor and > 0 on the
   standard predictor, and its loss term back-propagates.

Run: python -m checks.verify_graph_mask
"""
from __future__ import annotations

from pathlib import Path

import torch
import yaml

from models.jepa import graph
from models.jepa.model import JEPA
from models.jepa.train import edge_sensitivity
from paths import MODEL_CONFIGS


def small(cfg):
    """Shrink a model config so the check runs fast on CPU (wiring is unchanged)."""
    cfg = dict(cfg)
    cfg.update(image_size=32, patch_size=8, vit_depth=1, predictor_depth=2, predictor_mlp_dim=128,
               predictor_dropout=0.0)
    return cfg


def main():
    torch.manual_seed(0)
    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(('  PASS ' if cond else '  FAIL ') + msg)

    print('[1] slot mask')
    spec = graph.graph_spec({'predictor_type': 'graph_masked', 'latent_dim': 192})
    print('    ' + graph.describe(spec).replace('\n', '\n    '))
    expected_absent = {('x', 'vbar'), ('x', 'theta'), ('x', 'wbar'), ('vbar', 'theta'), ('vbar', 'wbar')}
    got = {(spec['names'][i], spec['names'][j]) for i, j in spec['non_edges']}
    check(got == expected_absent, f'absent state edges = {sorted(got)}')
    check(graph.is_closed(spec['mask']), 'mask is transitively closed')

    cfgs = {k: small(yaml.safe_load(Path(MODEL_CONFIGS / f).read_text())['model'])
            for k, f in (('base', 'jepa_cartpole.yaml'), ('mask', 'jepa_cartpole_graphmask.yaml'),
                         ('jac', 'jepa_cartpole_graphjac.yaml'))}
    models = {k: JEPA(c).eval() for k, c in cfgs.items()}
    b, T, d = 8, models['mask'].history_size, 192
    zh = torch.randn(b, T, d)
    ah = torch.randn(b, T, models['mask'].action_context_dim)

    print('[2] masked predictor respects the graph')
    m = models['mask']
    with torch.no_grad():
        p0 = m.predict(zh, ah)[:, -1]
        for i, ni in enumerate(spec['names']):
            z1 = zh.clone()
            z1[..., spec['slices'][i]] += torch.randn_like(z1[..., spec['slices'][i]])
            p1 = m.predict(z1, ah)[:, -1]
            for j, nj in enumerate(spec['names']):
                diff = (p1 - p0)[:, spec['slices'][j]].abs().max().item()
                if spec['mask'][j, i]:
                    check(diff > 1e-6, f'{ni:>5s} -> {nj:<5s} allowed, changes by {diff:.2e}')
                else:
                    check(diff <= 1e-6, f'{ni:>5s} -> {nj:<5s} masked,  changes by {diff:.2e}')
        # shorter histories use a leading block of the mask
        check(m.predict(zh[:, :2], ah[:, :2]).shape == (b, 2, d), 'shorter history works')

    print('[3] shapes match the standard JEPA')
    obs, prev = torch.rand(b, 3, 32, 32), torch.rand(b, 3, 32, 32)
    with torch.no_grad():
        for k in ('mask', 'jac'):
            check(models[k].encode(obs, prev).shape == models['base'].encode(obs, prev).shape, f'{k}: encode shape')
            check(models[k].predict(zh, ah).shape == models['base'].predict(zh, ah).shape, f'{k}: predict shape')

    print('[4] Jacobian-penalty estimator')
    with torch.no_grad():
        s_mask = edge_sensitivity(models['mask'], zh, ah, spec, spec['non_edges'], 0.1)
        s_dense = edge_sensitivity(models['jac'], zh, ah, models['jac'].graph_spec, spec['non_edges'], 0.1)
    check(s_mask.abs().max().item() <= 1e-10, f'masked predictor non-edge sensitivity {s_mask.abs().max().item():.2e}')
    check(s_dense.min().item() > 0, f'dense predictor non-edge sensitivity min {s_dense.min().item():.2e}')
    mj = models['jac'].train()
    idx = torch.randint(len(spec['non_edges']), (b,))
    loss = edge_sensitivity(mj, zh, ah, mj.graph_spec, spec['non_edges'], 0.1, idx).mean()
    loss.backward()
    g = sum(p.grad.abs().sum().item() for p in mj.predictor.parameters() if p.grad is not None)
    check(g > 0, f'penalty back-propagates into the predictor (|grad| sum {g:.2e})')
    check(mj.predictor.training, 'predictor train mode restored after the penalty pass')

    print('ALL PASS' if ok else 'SOME CHECKS FAILED')
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
