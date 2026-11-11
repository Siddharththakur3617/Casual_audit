"""Step 1 for Acrobot: checks to run before building anything else (same six as for cart-pole).

(a) symbolic Euler-Lagrange equations vs the simulator code (arm), and the simulator step vs an
    independent integration (arm + distractor)
(b) energy conservation of the arm with u = 0 and a tiny step
(c) observability: the true velocities at t from the frame variables, with and without u_{t-1}
(d) Jacobian sparsity of the true transition map vs the hand-derived graph (envs/acrobot.py)
(e) effect sizes -> final strong/weak tiers
(f) Markov check: older torques / older states add nothing once the physics is known

Run: python checks/verify_acrobot.py --vary-physics   -> results/checks_ac_phys.json,
                                                         results/expert_graph_final_ac_phys.json
     (also: python checks/verify_equations.py --env acrobot --vary-physics)
Without --vary-physics the default physics is used (-> *_ac.json).

Pass bars were fixed before the first run: (a) error < 1e-9; (b) relative drift < 1e-3 at h = 1e-5;
(c) held-out R2 > 0.99 for every true velocity from frame variables + u_{t-1}; (d) exact zero pattern
equals the hand graph; (f) largest R2 gain from older history < 0.005 given the physics.
The graph has 54 edges [th1, w1bar, th2, w2bar, p, vpbar, u_t, u_{t-1}, u_{t-2}] -> 6 changes; physics is
not a node. Each transition is differentiated at its own episode's physics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import envs  # noqa: E402
from cc_common import RESULTS, stack_episodes  # noqa: E402

ENV = envs.use('acrobot')
from sim.acrobot import (C_DAMP, FRAME_SKIP, H, K_SPRING, PHYS, accelerations, energy,  # noqa: E402
                         macro_step)

INPUT_NAMES, TARGET_NAMES, PHYS_KEYS, DT, T_MIN = (ENV.INPUT_NAMES, ENV.TARGET_NAMES, ENV.PHYS_KEYS,
                                                    ENV.DT, ENV.T_MIN)
ANGLE_IN = [0, 2]                    # input columns that are wrapped angles
RNG = np.random.default_rng(0)
PRNG = np.random.default_rng(1)
REPORT = {}
VARY = False
SUFFIX = '_ac'


def physics_ranges():
    from datagen import generate
    generate.ENV = 'acrobot'
    return generate.load_cfg()['physics_ranges']


def draw_phys(n=None):
    r = physics_ranges()
    return {k: (float(PRNG.uniform(*r[k])) if n is None else PRNG.uniform(*r[k], size=n)) for k in PHYS_KEYS}


def say(name, ok, **info):
    REPORT[name] = dict(passed=bool(ok), **info)
    print(f'[{"PASS" if ok else "FAIL"}] {name}: ' +
          ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}' for k, v in info.items()))


def angle_features(M, angle_cols):
    """Replace each wrapped-angle column by (sin, cos)."""
    cols = []
    for i in range(M.shape[1]):
        cols += [np.sin(M[:, i]), np.cos(M[:, i])] if i in angle_cols else [M[:, i]]
    return np.column_stack(cols)


# ---------------------------------------------------------------- (a)
def lagrangian_accelerations():
    t = sp.symbols('t')
    m1, m2, l1, l2, g, u = sp.symbols('m1 m2 l1 l2 g u')
    T1, T2 = sp.Function('th1')(t), sp.Function('th2')(t)
    lc1, lc2 = l1 / 2, l2 / 2
    I1, I2 = m1 * l1 ** 2 / 12, m2 * l2 ** 2 / 12
    c1 = sp.Matrix([lc1 * sp.sin(T1), -lc1 * sp.cos(T1)])
    c2 = sp.Matrix([l1 * sp.sin(T1) + lc2 * sp.sin(T1 + T2), -l1 * sp.cos(T1) - lc2 * sp.cos(T1 + T2)])
    v1, v2 = c1.diff(t), c2.diff(t)
    KE = (m1 * v1.dot(v1) + I1 * T1.diff(t) ** 2 + m2 * v2.dot(v2) + I2 * (T1.diff(t) + T2.diff(t)) ** 2) / 2
    PE = m1 * g * c1[1] + m2 * g * c2[1]
    L = KE - PE
    Q = {T1: 0, T2: u}
    # Euler-Lagrange residuals; they are linear in the accelerations: M(q) a = rhs(q, qdot, u).
    # (sympy.solve on the full system overflows the stack, so the 2x2 system is solved numerically.)
    res = [L.diff(q.diff(t)).diff(t) - L.diff(q) - Q[q] for q in (T1, T2)]
    a1, a2 = sp.symbols('a1 a2')
    th1, th2, w1, w2 = sp.symbols('th1v th2v w1v w2v')
    res = [r.subs({T1.diff(t, 2): a1, T2.diff(t, 2): a2}).subs(
        {T1.diff(t): w1, T2.diff(t): w2}).subs({T1: th1, T2: th2}) for r in res]
    Mm = [[r.diff(a) for a in (a1, a2)] for r in res]
    rhs = [-r.subs({a1: 0, a2: 0}) for r in res]
    args = (th1, th2, w1, w2, u, m1, m2, l1, l2, g)
    fM = [[sp.lambdify(args, e, 'numpy') for e in row] for row in Mm]
    fr = [sp.lambdify(args, e, 'numpy') for e in rhs]

    def solve(*a):
        n = np.broadcast(*a).shape
        M = np.stack([np.stack([np.broadcast_to(f(*a), n) for f in row], -1) for row in fM], -2)
        b = np.stack([np.broadcast_to(f(*a), n) for f in fr], -1)
        return np.linalg.solve(M, b[..., None])[..., 0]

    return (lambda *a: solve(*a)[..., 0]), (lambda *a: solve(*a)[..., 1])


def check_a():
    f1, f2 = lagrangian_accelerations()
    n = 10_000
    th1, th2 = RNG.uniform(-np.pi, np.pi, n), RNG.uniform(-np.pi, np.pi, n)
    w1, w2 = RNG.uniform(-12, 12, n), RNG.uniform(-25, 25, n)
    u = RNG.uniform(-3, 3, n)
    ph = dict(m1=RNG.uniform(0.5, 2, n), m2=RNG.uniform(0.5, 2, n), l1=RNG.uniform(0.5, 1.5, n),
              l2=RNG.uniform(0.5, 1.5, n), g=RNG.uniform(4.9, 14.7, n))
    a1, a2 = accelerations(th1, th2, w1, w2, u, **ph)
    r1 = f1(th1, th2, w1, w2, u, ph['m1'], ph['m2'], ph['l1'], ph['l2'], ph['g'])
    r2_ = f2(th1, th2, w1, w2, u, ph['m1'], ph['m2'], ph['l1'], ph['l2'], ph['g'])
    scale = max(np.abs(r1).max(), np.abs(r2_).max())
    err_formula = float(max(np.abs(a1 - r1).max(), np.abs(a2 - r2_).max()) / scale)

    from sim.acrobot import Acrobot
    env = Acrobot(render=False)
    err_env = 0.0
    for _ in range(2000):
        s = np.r_[RNG.uniform(-np.pi, np.pi), RNG.uniform(-8, 8), RNG.uniform(-np.pi, np.pi), RNG.uniform(-15, 15),
                  RNG.uniform(-1.5, 1.5), RNG.uniform(-3, 3)]
        f = RNG.uniform(-3, 3)
        q = draw_phys() if VARY else dict(PHYS)
        ref = s.copy()
        for _ in range(FRAME_SKIP):
            A1 = f1(ref[0], ref[2], ref[1], ref[3], f, q['m1'], q['m2'], q['l1'], q['l2'], q['g'])
            A2 = f2(ref[0], ref[2], ref[1], ref[3], f, q['m1'], q['m2'], q['l1'], q['l2'], q['g'])
            ref = ref + H * np.array([ref[1], A1, ref[3], A2, ref[5], -K_SPRING * ref[4] - C_DAMP * ref[5]])
        env.reset_to_state(s, q)
        env.step(f)
        err_env = max(err_env, float(np.abs(env.state - ref).max()))
    say('a_equations', err_formula < 1e-9 and err_env < 1e-9,
        max_rel_err_formula_vs_lagrangian=err_formula, max_err_simulator_step_vs_lagrangian=err_env)


# ---------------------------------------------------------------- (b)
def check_b():
    n = 50
    s0 = np.column_stack([RNG.uniform(-np.pi, np.pi, n), RNG.uniform(-2, 2, n), RNG.uniform(-np.pi, np.pi, n),
                          RNG.uniform(-2, 2, n), np.zeros(n), np.zeros(n)])
    ph = draw_phys(n) if VARY else dict(PHYS)
    # energy scale: the arm's potential-energy amplitude, (m1 lc1 + m2 (l1 + lc2)) g
    scale = (ph['m1'] * ph['l1'] / 2 + ph['m2'] * (ph['l1'] + ph['l2'] / 2)) * ph['g']
    drift = {}
    for h in (1e-5, H):
        s = s0.copy()
        for _ in range(int(round(0.5 / h))):
            s = macro_step(s, 0.0, n=1, h=h, **ph)
        drift[h] = float((np.abs(energy(s, **ph) - energy(s0, **ph)) / scale).max())
    say('b_energy', drift[1e-5] < 1e-3, rel_drift_h1e5=drift[1e-5], rel_drift_h001_for_reference=drift[H])


# ---------------------------------------------------------------- data
def sim_data():
    from datagen.generate import simulate_states
    eps = simulate_states(vary_physics=VARY, env='acrobot')
    V = stack_episodes([(s, a) for s, a, _, _ in eps], env=ENV)
    P_ep = np.array([[ph[k] for k in PHYS_KEYS] for _, _, _, ph in eps])
    V['physics'] = P_ep[V['episode']]
    return V, eps


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def split_by_episode(ep, frac=0.8):
    ids = np.unique(ep)
    RNG.shuffle(ids)
    tr = np.isin(ep, ids[:int(len(ids) * frac)])
    return tr, ~tr


def mlp_r2(F, y, tr, te):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    ys = (y - y[tr].mean()) / y[tr].std()
    sc = StandardScaler().fit(F[tr])
    mdl = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=400, early_stopping=True, random_state=0,
                       tol=1e-6, n_iter_no_change=20)
    mdl.fit(sc.transform(F[tr]), ys[tr])
    return r2(ys[te], mdl.predict(sc.transform(F[te])))


# ---------------------------------------------------------------- (c)
def check_c(V):
    X = V['inputs']
    tr, te = split_by_episode(V['episode'])
    frames = angle_features(X[:, :6], ANGLE_IN)
    with_u = np.column_stack([frames, X[:, 7]])            # + u_{t-1}
    if VARY:
        frames, with_u = np.column_stack([frames, V['physics']]), np.column_stack([with_u, V['physics']])
    res = {}
    for k, name in enumerate(('w1', 'w2', 'pdot')):
        y = V['true_vel'][:, k]
        res[f'{name}_frames_only_R2'] = mlp_r2(frames, y, tr, te)
        res[f'{name}_frames_plus_u_tm1_R2'] = mlp_r2(with_u, y, tr, te)
        res[f'{name}_minus_bar_std_over_std'] = float((y - X[:, 2 * k + 1]).std() / y.std())
    ok = all(res[f'{n}_frames_plus_u_tm1_R2'] > 0.99 for n in ('w1', 'w2', 'pdot'))
    say('c_observability', ok, physics_given=VARY, **res)


# ---------------------------------------------------------------- (d), (e)
def transition_map(q, phys=None):
    """q = (s_{t-1} (6), u_{t-1}, u_t, u_{t-2}) -> (observed inputs o (9), targets (6)). Raw (unwrapped) angles:
    wrapping has derivative 1, so the Jacobian is the same."""
    phys = PHYS if phys is None else phys
    s_prev, u_tm1, u_t, u_tm2 = q[..., :6], q[..., 6], q[..., 7], q[..., 8]
    s_t = macro_step(s_prev, u_tm1, **phys)
    s_n = macro_step(s_t, u_t, **phys)
    o, tg = [], []
    for c in (0, 2, 4):
        bar_t = (s_t[..., c] - s_prev[..., c]) / DT
        bar_n = (s_n[..., c] - s_t[..., c]) / DT
        o += [s_t[..., c], bar_t]
        tg += [s_n[..., c] - s_t[..., c], bar_n - bar_t]
    return np.stack(o + [u_t, u_tm1, u_tm2], axis=-1), np.stack(tg, axis=-1)


def jacobians(Q, phys=None, eps=1e-6):
    n, k = Q.shape
    Jo, Jt = np.zeros((n, 9, k)), np.zeros((n, 6, k))
    for i in range(k):
        dq = np.zeros(k); dq[i] = eps
        op, tp = transition_map(Q + dq, phys)
        om, tm = transition_map(Q - dq, phys)
        Jo[:, :, i] = (op - om) / (2 * eps)
        Jt[:, :, i] = (tp - tm) / (2 * eps)
    return np.einsum('nti,nio->nto', Jt, np.linalg.inv(Jo))     # (n, 6 targets, 9 inputs)


def check_d_e(V, eps_raw):
    Q, P = [], []
    for s, a, _, ph in eps_raw:
        a = a.reshape(-1)
        for t in range(T_MIN, len(a)):
            Q.append(np.r_[s[t - 1], a[t - 1], a[t], a[t - 2]])
            P.append([ph[k] for k in PHYS_KEYS])
    Q, P = np.asarray(Q, dtype=np.float64), np.asarray(P)
    pick = RNG.choice(len(Q), size=min(4000, len(Q)), replace=False)
    Q, P = Q[pick], P[pick]
    phys = {k: P[:, i] for i, k in enumerate(PHYS_KEYS)} if VARY else None
    J = jacobians(Q, phys)
    sd_in, sd_tg = V['inputs'].std(0), V['targets'].std(0)
    share = (np.abs(J).mean(0) * sd_in[None, :] / sd_tg[:, None]).T
    maxabs = (np.abs(J).max(0) * sd_in[None, :] / sd_tg[:, None]).T
    nonzero = maxabs > 1e-6
    A = ENV.adjacency().astype(bool)
    mism = [f'{INPUT_NAMES[i]}->{TARGET_NAMES[j]} (hand={int(A[i, j])}, measured={maxabs[i, j]:.2g})'
            for i in range(len(INPUT_NAMES)) for j in range(len(TARGET_NAMES)) if nonzero[i, j] != A[i, j]]
    cross_zero = bool((~nonzero[ENV.cross_mask()]).all())
    say('d_jacobian_pattern', not mism, mismatches=mism or 'none', all_cross_edges_exactly_zero=cross_zero,
        n_present=int(nonzero.sum()), n_absent=int((~nonzero).sum()))

    tiers_hand = ENV.tiers()
    tiers_final = np.where(~nonzero, 'absent', np.where(share >= 0.1, 'strong', 'weak'))
    print('\n  normalised effect size (mean |dT/do| * sd_in / sd_target); rows = inputs')
    print('  ' + ' ' * 7 + ''.join(f'{t:>9s}' for t in TARGET_NAMES))
    for i, nm in enumerate(INPUT_NAMES):
        print(f'  {nm:7s}' + ''.join(f'{share[i, j]:9.3g}' for j in range(len(TARGET_NAMES))))
    changed = [f'{INPUT_NAMES[i]}->{TARGET_NAMES[j]}: {tiers_hand[i, j]} -> {tiers_final[i, j]}'
               for i in range(len(INPUT_NAMES)) for j in range(len(TARGET_NAMES))
               if tiers_hand[i, j] != tiers_final[i, j]]
    say('e_sensitivity', True, tier_changes=changed or 'none',
        n_strong=int((tiers_final == 'strong').sum()), n_weak=int((tiers_final == 'weak').sum()),
        n_absent=int((tiers_final == 'absent').sum()))
    final = {
        'env': 'acrobot', 'inputs': INPUT_NAMES, 'targets': TARGET_NAMES,
        'adjacency': nonzero.astype(int).tolist(),
        'tiers_derived': tiers_hand.tolist(),
        'tiers': tiers_final.tolist(),
        'effect_size': share.tolist(),
        'cross_mask': ENV.cross_mask().astype(int).tolist(),
        'rule': 'absent = exact zero in Jacobian; strong = normalised effect >= 0.1; else weak',
    }
    if VARY:
        final['physics'] = {'varied': True, 'ranges': physics_ranges(),
                            'note': 'each transition differentiated at its own episode physics; physics is not a node'}
    (RESULTS / f'expert_graph_final{SUFFIX}.json').write_text(json.dumps(final, indent=2))


# ---------------------------------------------------------------- (f)
def _history_gain(V, base, ext, tr, te):
    res, worst = {}, -1.0
    for j, nm in enumerate(TARGET_NAMES):
        y = V['targets'][:, j]
        s0, s1 = mlp_r2(base, y, tr, te), mlp_r2(ext, y, tr, te)
        res[f'{nm}_R2_markov_inputs'], res[f'{nm}_R2_plus_history'] = s0, s1
        worst = max(worst, s1 - s0)
    return res, float(worst)


def check_f(V):
    tr, te = split_by_episode(V['episode'])
    base = angle_features(V['inputs'][:, :8], ANGLE_IN)                       # without u_{t-2}
    ext = np.column_stack([angle_features(V['inputs'], ANGLE_IN), angle_features(V['d_prev'], ANGLE_IN)])
    if not VARY:
        res, worst = _history_gain(V, base, ext, tr, te)
        say('f_markov', worst < 0.005, max_R2_gain_from_history=worst, **res)
        return
    P = V['physics']
    res, worst = _history_gain(V, np.column_stack([base, P]), np.column_stack([ext, P]), tr, te)
    hid, hid_gain = _history_gain(V, base, ext, tr, te)
    say('f_markov', worst < 0.005, max_R2_gain_from_history_given_physics=worst, **res,
        hidden_physics_max_R2_gain_from_history=hid_gain, **{f'hidden_physics_{k}': v for k, v in hid.items()})


def main(argv=None):
    global VARY, SUFFIX
    p = argparse.ArgumentParser()
    p.add_argument('--vary-physics', action='store_true')
    p.add_argument('--env', default='acrobot')
    VARY = p.parse_args(argv).vary_physics
    SUFFIX = '_ac_phys' if VARY else '_ac'
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / 'expert_graph_acrobot.json').write_text(json.dumps(
        {'inputs': INPUT_NAMES, 'targets': TARGET_NAMES, 'adjacency': ENV.adjacency().tolist(),
         'tiers': ENV.tiers().tolist(), 'why_absent': ENV.WHY}, indent=2))
    check_a()
    check_b()
    print('[data] simulating states with the dataset policies (no rendering)'
          + (' with per-episode physics' if VARY else '') + ' ...', flush=True)
    V, eps_raw = sim_data()
    print(f'[data] {len(V["t"])} usable transitions (t >= {T_MIN}) from {len(eps_raw)} episodes', flush=True)
    check_c(V)
    check_d_e(V, eps_raw)
    check_f(V)
    (RESULTS / f'checks{SUFFIX}.json').write_text(json.dumps(REPORT, indent=2, default=str))
    n_fail = sum(not r['passed'] for r in REPORT.values())
    print(f'\n{len(REPORT) - n_fail}/{len(REPORT)} checks passed -> results/checks{SUFFIX}.json')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
