"""Slot layout, change-parent masks and graph controls for the per-output arm (any environment).

z = [one slot per state variable | ctx]. Cart-pole: [x | vbar | theta | wbar | ctx];
Acrobot: [th1 | w1bar | th2 | w2bar | p | vpbar | ctx]. ctx is a free slot for the hidden
per-episode physics, which are true parents of every change but have no node in the graph.

cmask[j, i] = slot i is a parent of the CHANGE of slot j. The predictor computes
zhat_j(t+1) = z_j(t) + f_j(parents of the change of j); the z_j(t) term is integration, not a
causal edge, so self is a parent only when the expert graph lists it. Every change reads ctx; the
change of ctx reads only ctx.
amask[j, c] = action-context component c (oldest first: u_{t-ctx+1} .. u_t) is a parent of the
change of slot j. On cart-pole every change has both actions as parents; on Acrobot the distractor
has none, which is why actions are masked per output too.

Graph sources (model config `graph_source`):
  hand      checks/expert_graph.tiers() (cart-pole only; the graph used by the gnn-commit arms)
  measured  results/expert_graphs/expert_graph_final_<data>.json via evaluate/compare.load_expert
Edge sets (`graph_edges`): strong_weak (default), strong, full (dense control), shuffled
(the same number of state->change edges placed at random).
Corruption (`graph_corrupt: {drop: k, add: k, seed: s}`): delete k true state->change edges and/or
add k absent ones, to measure the cost of a wrong graph separately for each error type.

The resolved masks are written into the model config (`change_mask`, `action_mask`,
`expert_change_mask`, `expert_action_mask`, `slot_names`, `slot_sizes`, `anchor_dims`), so
checkpoints carry their exact graph and load without the results files.
"""
from __future__ import annotations

import numpy as np

CTX = 'ctx'
EDGE_SETS = ('strong_weak', 'strong', 'full', 'shuffled')
CTX_SIZE = 32


def default_layout(state_names, latent_dim=192, ctx_size=CTX_SIZE):
    """Slot names and sizes: the state slots share latent_dim - ctx_size as evenly as possible."""
    n = len(state_names)
    rest = latent_dim - ctx_size
    sizes = [rest // n + (1 if k < rest % n else 0) for k in range(n)]
    return list(state_names) + [CTX], sizes + [ctx_size]


def lag_name(lag):
    return 'u_t' if lag == 0 else f'u_tm{lag}'


def expert_tiers(source, data_dir=None):
    """(input names, target names, tiers (n_inputs, n_targets) of 'strong' / 'weak' / 'absent')."""
    if source == 'hand':
        from checks import expert_graph
        from envs.cartpole import INPUT_NAMES, TARGET_NAMES
        return list(INPUT_NAMES), list(TARGET_NAMES), np.asarray(expert_graph.tiers(), dtype=object)
    if source == 'measured':
        assert data_dir is not None, "graph_source 'measured' needs the dataset (--data)"
        import envs
        from evaluate import compare
        env = envs.use_data(data_dir)
        _, tiers, _ = compare.load_expert(data_dir)
        return list(env.INPUT_NAMES), list(env.TARGET_NAMES), np.asarray(tiers, dtype=object)
    raise ValueError(f"graph_source must be 'hand' or 'measured', not {source!r}")


def expert_masks(names, ctx_dim, source='hand', edges='strong_weak', data_dir=None):
    """(cmask (K, K), amask (K, ctx_dim)) from the expert graph."""
    assert edges in EDGE_SETS
    K = len(names)
    if edges == 'full':
        return np.ones((K, K), bool), np.ones((K, ctx_dim), bool)
    inputs, targets, tiers = expert_tiers(source, data_dir)
    keep = ('strong',) if edges == 'strong' else ('strong', 'weak')
    cmask, amask = np.zeros((K, K), bool), np.zeros((K, ctx_dim), bool)
    for j, nj in enumerate(names):
        if nj == CTX:
            continue
        col = targets.index('d' + nj)
        for i, ni in enumerate(names):
            if ni != CTX and ni in inputs:
                cmask[j, i] = tiers[inputs.index(ni), col] in keep
        for c in range(ctx_dim):
            a = lag_name(ctx_dim - 1 - c)
            if a in inputs:
                amask[j, c] = tiers[inputs.index(a), col] in keep
    ci = names.index(CTX)
    cmask[:, ci] = True                     # every change reads ctx
    cmask[ci, :] = False
    cmask[ci, ci] = True                    # the change of ctx reads only ctx
    amask[ci, :] = False
    return cmask, amask


def state_cells(names):
    return [(j, i) for j, nj in enumerate(names) for i, ni in enumerate(names) if CTX not in (ni, nj)]


def corrupt(cmask, names, shuffle=False, drop=0, add=0, seed=0):
    """Shuffle, delete or add state->change edges (ctx and actions untouched). Returns (mask, log)."""
    rng = np.random.default_rng(seed)
    m = cmask.copy()
    cells = state_cells(names)
    log = {}
    if shuffle:
        n_true = sum(m[c] for c in cells)
        for c in cells:
            m[c] = False
        pick = rng.choice(len(cells), size=n_true, replace=False)
        for k in pick:
            m[cells[k]] = True
        log['shuffled'] = [f'{names[i]}->d{names[j]}' for j, i in (cells[k] for k in sorted(pick))]
    true = [c for c in cells if m[c]]
    false = [c for c in cells if not m[c]]
    assert drop <= len(true) and add <= len(false), 'graph_corrupt asks for more edges than exist'
    dropped = [true[k] for k in rng.choice(len(true), size=drop, replace=False)] if drop else []
    added = [false[k] for k in rng.choice(len(false), size=add, replace=False)] if add else []
    for c in dropped:
        m[c] = False
    for c in added:
        m[c] = True
    if drop:
        log['dropped'] = [f'{names[i]}->d{names[j]}' for j, i in dropped]
    if add:
        log['added'] = [f'{names[i]}->d{names[j]}' for j, i in added]
    return m, log


def anchor_dims(env, state_names):
    """Anchor-head outputs per state slot: 2 (sin, cos) for wrapped angles, else 1."""
    pos = list(getattr(env, 'POS', [0, 2]))
    ang = list(getattr(env, 'ANGLE', [False] * len(pos)))
    dims = []
    for a in ang:
        dims += [2 if a else 1, 1]                # position (or angle), then its frame velocity
    assert len(dims) == len(state_names)
    return dims


def resolve(cfg, data_dir=None):
    """Fill the slot layout and the graph masks into a model config (in place) and return it.
    Already-resolved configs (from a checkpoint) are returned unchanged."""
    if 'change_mask' in cfg:
        return cfg
    import envs
    env = envs.use_data(data_dir) if data_dir is not None else envs.current()
    d = int(cfg.get('latent_dim', 192))
    names, sizes = default_layout(env.STATE_NAMES, d, int(cfg.get('ctx_size', CTX_SIZE)))
    names = list(cfg.get('slot_names', names))
    sizes = [int(s) for s in cfg.get('slot_sizes', sizes)]
    assert names[:-1] == list(env.STATE_NAMES) and names[-1] == CTX, \
        f'slot_names must be {list(env.STATE_NAMES) + [CTX]}'
    assert sum(sizes) == d, f'slot sizes sum to {sum(sizes)}, latent_dim is {d}'
    ctx_dim = int(cfg.get('action_context_dim', 2))
    source = cfg.get('graph_source', 'hand' if env.NAME == 'cartpole' else 'measured')
    edges = cfg.get('graph_edges', 'strong_weak')
    assert edges in EDGE_SETS, f'graph_edges must be one of {EDGE_SETS}'
    exp_c, exp_a = expert_masks(names, ctx_dim, source, 'strong_weak', data_dir)   # reference for diagnostics
    if edges in ('full', 'strong'):
        cm, am = expert_masks(names, ctx_dim, source, edges, data_dir)
    else:                                                                           # strong_weak, shuffled
        cm, am = exp_c.copy(), exp_a.copy()
    log = {}
    cor = dict(cfg.get('graph_corrupt') or {})
    if edges == 'shuffled' or cor:
        cm, log = corrupt(cm, names, shuffle=edges == 'shuffled', drop=int(cor.get('drop', 0)),
                          add=int(cor.get('add', 0)), seed=int(cor.get('seed', cfg.get('graph_seed', 0))))
    cfg.update(slot_names=names, slot_sizes=sizes, graph_source=source,
               change_mask=cm.astype(int).tolist(), action_mask=am.astype(int).tolist(),
               expert_change_mask=exp_c.astype(int).tolist(), expert_action_mask=exp_a.astype(int).tolist(),
               anchor_dims=anchor_dims(env, names[:-1]), graph_log=log)
    return cfg


def non_edges(cmask, names):
    """[(i, j)] between state slots where i is NOT a parent of the change of j, self pairs included."""
    return [(i, j) for j, i in state_cells(names) if not cmask[j, i]]


def parent_edges(cmask, names):
    return [(i, j) for j, i in state_cells(names) if cmask[j, i]]


def spec_from_cfg(cfg):
    """Slot layout and masks from a RESOLVED model config (see resolve)."""
    assert 'change_mask' in cfg, 'model config is not resolved: call models.jepa_peroutput.graph.resolve first'
    names, sizes = list(cfg['slot_names']), [int(s) for s in cfg['slot_sizes']]
    cm = np.asarray(cfg['change_mask'], bool)
    am = np.asarray(cfg['action_mask'], bool)
    ex = np.asarray(cfg['expert_change_mask'], bool)
    starts = np.cumsum([0] + sizes)
    return {'names': names, 'sizes': sizes, 'edges': cfg.get('graph_edges', 'strong_weak'),
            'cmask': cm, 'amask': am, 'expert_cmask': ex,
            'slices': [slice(int(a), int(b)) for a, b in zip(starts[:-1], starts[1:])],
            'ctx': names.index(CTX), 'state_idx': list(range(len(names) - 1)),
            'anchor_dims': list(cfg.get('anchor_dims', [1] * (len(names) - 1))),
            # diagnostics always use the uncorrupted expert graph
            'expert_non_edges': non_edges(ex, names), 'expert_parent_edges': parent_edges(ex, names),
            'graph_log': cfg.get('graph_log', {})}


def describe(spec):
    n, m, a = spec['names'], spec['cmask'], spec['amask']
    ctx = a.shape[1]
    lines = []
    for j in range(len(n)):
        acts = [lag_name(ctx - 1 - c) for c in range(ctx) if a[j, c]]
        lines.append(f'  d{n[j]:<6s} <- ' + ', '.join([n[i] for i in range(len(n)) if m[j, i]] + acts))
    absent = ', '.join(f'{n[i]}->d{n[j]}' for i, j in non_edges(m, n)) or 'none (dense)'
    lines.append(f'  absent state edges: {absent}')
    for k, v in spec.get('graph_log', {}).items():
        lines.append(f'  {k}: {", ".join(v)}')
    return '\n'.join(lines)
