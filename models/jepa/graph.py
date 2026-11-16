"""Expert-graph structure for the graph-prior JEPA arms (cart-pole).

The latent z is split into named slots, one per state variable plus a free context slot:
  z = [x | vbar | theta | wbar | ctx]      (default sizes 40, 40, 40, 40, 32 -> 192)
ctx is meant for the hidden per-episode physics (M, m, l, g): true parents of every change
that have no node in the graph.

mask[j, i] = True  <=>  predicted slot j may read slot i (i is a parent of j, i == j, or i is ctx).
ctx reads only itself. Parents come from checks/expert_graph.EXPERT: strong + weak edges by
default, because a missing true edge hurts far more than an extra one. Self is always allowed:
the predictor outputs z_{t+1} = z_t + change, so a slot must see its own past.

For cart-pole (strong + weak) the state parents are
  x <- vbar, theta, wbar     vbar <- theta, wbar     theta <- wbar     wbar <- theta
so the edges the mask removes are x -> vbar, theta, wbar and vbar -> theta, wbar. The mask is
closed under composition (parents of parents are parents), so stacking masked attention layers
adds no non-parent path within one step.

Actions u_t, u_{t-1} are parents of every change, so global action conditioning already matches
the graph; u_{t-2} is not in the action context.
"""
from __future__ import annotations

import numpy as np

DEFAULT_SLOT_NAMES = ('x', 'vbar', 'theta', 'wbar', 'ctx')
DEFAULT_SLOT_SIZES = (40, 40, 40, 40, 32)
CTX = 'ctx'
EDGE_SETS = ('strong_weak', 'strong', 'shuffled', 'full')


def expert_state_parents(edges='strong_weak'):
    """{state slot: [state parents]} from checks/expert_graph.EXPERT (actions dropped)."""
    from checks.expert_graph import EXPERT
    from envs.cartpole import STATE_NAMES
    keep = ('strong', 'weak') if edges in ('strong_weak', 'shuffled') else ('strong',)
    out = {}
    for tgt, parents in EXPERT.items():
        out[tgt[1:]] = [p for p, tier in parents.items() if p in STATE_NAMES and tier in keep]
    return out


def is_closed(mask):
    """True if every two-hop path is already a direct edge (no leakage through depth)."""
    m = mask.astype(int)
    return bool(np.all(((m @ m) > 0) <= mask))


def slot_mask(names, edges='strong_weak', seed=0):
    """(K, K) bool, mask[j, i] = slot j may read slot i."""
    assert edges in EDGE_SETS, f'graph edges must be one of {EDGE_SETS}'
    K = len(names)
    idx = {n: k for k, n in enumerate(names)}
    state = [n for n in names if n != CTX]
    mask = np.eye(K, dtype=bool)
    if CTX in idx:
        mask[:, idx[CTX]] = True                       # everyone reads ctx; ctx reads only itself
    if edges == 'full':
        for j in state:
            for i in state:
                mask[idx[j], idx[i]] = True
        return mask
    parents = expert_state_parents(edges)
    pairs = [(idx[j], idx[i]) for j in state for i in parents.get(j, []) if i != j]
    if edges == 'shuffled':                           # same number of off-diagonal state edges, random
        rng = np.random.default_rng(seed)
        cand = [(idx[j], idx[i]) for j in state for i in state if i != j]
        pairs = [cand[k] for k in rng.choice(len(cand), size=len(pairs), replace=False)]
    for j, i in pairs:
        mask[j, i] = True
    if edges == 'strong_weak':
        assert is_closed(mask), 'expert slot mask is not transitively closed: depth would leak non-parents'
    elif not is_closed(mask):
        print(f'[graph] warning: {edges!r} slot mask is not transitively closed; stacked attention '
              'layers let information pass along two-hop paths within one step')
    return mask


def non_edges(mask, names):
    """[(i, j)]: slot i is NOT allowed to influence predicted slot j (state slots only)."""
    return [(i, j) for j, nj in enumerate(names) for i, ni in enumerate(names)
            if not mask[j, i] and ni != CTX and nj != CTX]


def parent_edges(mask, names):
    """[(i, j)]: allowed cross-slot reads between state slots (no self, no ctx)."""
    return [(i, j) for j, nj in enumerate(names) for i, ni in enumerate(names)
            if mask[j, i] and i != j and ni != CTX and nj != CTX]


def graph_spec(cfg, latent_dim=None):
    """Slot layout and mask from a model config, or None when the config has no slot layout.

    Keys read: slot_names, slot_sizes, graph_edges ('strong_weak'), graph_seed (0).
    """
    if 'slot_sizes' not in cfg and cfg.get('predictor_type', 'transformer') != 'graph_masked':
        return None
    names = tuple(cfg.get('slot_names', DEFAULT_SLOT_NAMES))
    sizes = tuple(int(s) for s in cfg.get('slot_sizes', DEFAULT_SLOT_SIZES))
    d = int(latent_dim or cfg.get('latent_dim', 192))
    assert len(names) == len(sizes), 'slot_names and slot_sizes differ in length'
    assert sum(sizes) == d, f'slot sizes sum to {sum(sizes)}, latent_dim is {d}'
    edges = cfg.get('graph_edges', 'strong_weak')
    mask = slot_mask(names, edges, int(cfg.get('graph_seed', 0)))
    starts = np.cumsum((0,) + sizes)
    return {'names': names, 'sizes': sizes, 'edges': edges, 'mask': mask,
            'slices': [slice(int(a), int(b)) for a, b in zip(starts[:-1], starts[1:])],
            'non_edges': non_edges(mask, names), 'parent_edges': parent_edges(mask, names),
            'ctx': names.index(CTX) if CTX in names else None}


def describe(spec):
    n = spec['names']
    lines = [f'{n[j]:>5s} <- ' + ', '.join(n[i] for i in range(len(n)) if spec['mask'][j, i])
             for j in range(len(n))]
    lines.append('absent: ' + ', '.join(f'{n[i]}->{n[j]}' for i, j in spec['non_edges']))
    return '\n'.join(lines)
