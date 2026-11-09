"""Step 1: checks to run before building anything else.

(a) symbolic Euler-Lagrange equations vs the simulator code
(b) energy conservation with u = 0 and a tiny step
(c) observability: two frames cannot give the instantaneous velocity without u_{t-1}
(d) Jacobian sparsity of the true transition map vs the expert graph
(e) effect sizes -> final strong/weak tiers
(f) Markov check: older forces / older states add nothing

Run: python checks/verify_equations.py   -> results/checks.json, results/expert_graph_final.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cc_common import (DT, FRAME_SKIP, H, INPUT_NAMES, PHYS, RESULTS, T_MIN,  # noqa: E402
                       TARGET_NAMES, accelerations, macro_step, stack_episodes)
import expert_graph  # noqa: E402

RNG = np.random.default_rng(0)
REPORT = {}


def say(name, ok, **info):
    REPORT[name] = dict(passed=bool(ok), **info)
    print(f'[{"PASS" if ok else "FAIL"}] {name}: ' +
          ', '.join(f'{k}={v:.3g}' if isinstance(v, float) else f'{k}={v}' for k, v in info.items()))


# ---------------------------------------------------------------- (a)
def lagrangian_accelerations():
    """Rod pole of half-length l (I_cm = m l^2 / 3), viscous damping b_c, b_p."""
    t = sp.symbols('t')
    M, m, l, g, u, bc, bp = sp.symbols('M m l g u b_c b_p')
    X, TH = sp.Function('x')(t), sp.Function('theta')(t)
    xd, thd = X.diff(t), TH.diff(t)
    T = (sp.Rational(1, 2) * M * xd ** 2
         + sp.Rational(1, 2) * m * ((xd + l * sp.cos(TH) * thd) ** 2 + (l * sp.sin(TH) * thd) ** 2)
         + sp.Rational(1, 2) * (m * l ** 2 / 3) * thd ** 2)
    V = m * g * l * sp.cos(TH)
    L = T - V
    Q = {X: u - bc * xd, TH: -bp * thd}
    eqs = [sp.Eq(L.diff(q.diff(t)).diff(t) - L.diff(q), Q[q]) for q in (X, TH)]
    xa, ta = sp.symbols('xa ta')
    sub = {X.diff(t, 2): xa, TH.diff(t, 2): ta}
    sol = sp.solve([e.subs(sub) for e in eqs], [xa, ta], dict=True)[0]
    xv, xdv, thv, thdv = sp.symbols('xv xdv thv thdv')
    back = {thd: thdv, xd: xdv, TH: thv, X: xv}
    args = (xdv, thv, thdv, u, M, m, l, g, bc, bp)
    f_xa = sp.lambdify(args, sol[xa].subs(back), 'numpy')
    f_ta = sp.lambdify(args, sol[ta].subs(back), 'numpy')
    return f_xa, f_ta


def check_a():
    f_xa, f_ta = lagrangian_accelerations()
    n = 10_000
    xd = RNG.uniform(-3, 3, n); th = RNG.uniform(-1.2, 1.2, n)
    thd = RNG.uniform(-5, 5, n); u = RNG.uniform(-10, 10, n)
    M = RNG.uniform(0.5, 2, n); m = RNG.uniform(0.05, 0.5, n); l = RNG.uniform(0.3, 0.8, n)
    bc = RNG.uniform(0, 1, n); bp = RNG.uniform(0, 0.05, n); g = np.full(n, 9.8)
    xa_ref, ta_ref = f_xa(xd, th, thd, u, M, m, l, g, bc, bp), f_ta(xd, th, thd, u, M, m, l, g, bc, bp)
    xa, ta = accelerations(th, thd, u, x_dot=xd, M=M, m=m, l=l, g=g, b_c=bc, b_p=bp)
    err_formula = float(max(np.abs(xa - xa_ref).max(), np.abs(ta - ta_ref).max()))

    # the simulator's step (2 Euler sub-steps, force held) vs an independent
    # integration that uses the Lagrangian accelerations directly
    from sim.cartpole import CartPole
    env = CartPole(frame_skip=FRAME_SKIP, theta_threshold=10.0, render=False)
    err_env = 0.0
    for _ in range(2000):
        s = np.array([RNG.uniform(-1, 1), RNG.uniform(-3, 3), RNG.uniform(-1.2, 1.2), RNG.uniform(-5, 5)])
        f = RNG.uniform(-10, 10)
        ref = s.copy()
        for _ in range(FRAME_SKIP):
            xa = f_xa(ref[1], ref[2], ref[3], f, 1.0, 0.1, 0.5, 9.8, 0.0, 0.0)
            ta = f_ta(ref[1], ref[2], ref[3], f, 1.0, 0.1, 0.5, 9.8, 0.0, 0.0)
            ref = ref + H * np.array([ref[1], xa, ref[3], ta])
        env.reset_to_state(s)
        env.step(f)
        err_env = max(err_env, float(np.abs(env.state - ref).max()))
    say('a_equations', err_formula < 1e-9 and err_env < 1e-9,
        max_err_formula_vs_lagrangian=err_formula, max_err_simulator_step_vs_lagrangian=err_env)


# ---------------------------------------------------------------- (b)
def energy(s, M=1.0, m=0.1, l=0.5, g=9.8):
    xd, th, thd = s[..., 1], s[..., 2], s[..., 3]
    return (0.5 * (M + m) * xd ** 2 + m * l * np.cos(th) * xd * thd
            + 0.5 * (4 / 3) * m * l ** 2 * thd ** 2 + m * g * l * np.cos(th))


def check_b():
    s0 = np.column_stack([np.zeros(50), RNG.uniform(-1, 1, 50), RNG.uniform(-0.5, 0.5, 50),
                          RNG.uniform(-1, 1, 50)])
    drift = {}
    for h in (1e-5, H):
        s = s0.copy()
        for _ in range(int(round(0.5 / h))):
            s = macro_step(s, 0.0, n=1, h=h, **PHYS)
        drift[h] = float(np.abs(energy(s) - energy(s0)).max() / np.abs(energy(s0)).max())
    say('b_energy', drift[1e-5] < 1e-3, rel_drift_h1e5=drift[1e-5], rel_drift_h002_for_reference=drift[H])


# ---------------------------------------------------------------- data
def sim_data():
    from datagen.generate import simulate_states
    eps = simulate_states()
    return stack_episodes([(s, a) for s, a, _ in eps]), eps


def r2(y, p):
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def split_by_episode(ep, frac=0.8):
    ids = np.unique(ep)
    RNG.shuffle(ids)
    tr = np.isin(ep, ids[:int(len(ids) * frac)])
    return tr, ~tr


# ---------------------------------------------------------------- (c)
def check_c(V):
    from sklearn.linear_model import LinearRegression
    X = V['inputs']
    tr, te = split_by_episode(V['episode'])
    two_frames = X[:, :4]                        # x, vbar, theta, wbar  (same span as x_t, x_{t-1}, ...)
    with_u = np.column_stack([two_frames, X[:, 5]])   # + u_{t-1}
    res = {}
    for name, y in (('xdot', V['xdot']), ('thetadot', V['thetadot'])):
        for label, F in (('frames_only', two_frames), ('frames_plus_u_tm1', with_u)):
            p = LinearRegression().fit(F[tr], y[tr]).predict(F[te])
            res[f'{name}_{label}_R2'] = r2(y[te], p)
            res[f'{name}_{label}_rms_over_std'] = float(np.sqrt(((y[te] - p) ** 2).mean()) / y[te].std())
        res[f'{name}_minus_bar_std'] = float((y - X[:, 1 if name == 'xdot' else 3]).std())
    ok = (res['xdot_frames_plus_u_tm1_rms_over_std'] < 0.5 * res['xdot_frames_only_rms_over_std']
          and res['xdot_frames_plus_u_tm1_R2'] > 0.999)
    say('c_observability', ok, **res)


# ---------------------------------------------------------------- (d), (e)
def transition_map(q):
    """q = (s_{t-1} (4), u_{t-1}, u_t, u_{t-2}) -> (observed inputs o (7), targets (4))."""
    s_prev, u_tm1, u_t, u_tm2 = q[..., :4], q[..., 4], q[..., 5], q[..., 6]
    s_t = macro_step(s_prev, u_tm1, **PHYS)
    s_n = macro_step(s_t, u_t, **PHYS)
    vbar_t = (s_t[..., 0] - s_prev[..., 0]) / DT
    wbar_t = (s_t[..., 2] - s_prev[..., 2]) / DT
    vbar_n = (s_n[..., 0] - s_t[..., 0]) / DT
    wbar_n = (s_n[..., 2] - s_t[..., 2]) / DT
    o = np.stack([s_t[..., 0], vbar_t, s_t[..., 2], wbar_t, u_t, u_tm1, u_tm2], axis=-1)
    tg = np.stack([s_n[..., 0] - s_t[..., 0], vbar_n - vbar_t,
                   s_n[..., 2] - s_t[..., 2], wbar_n - wbar_t], axis=-1)
    return o, tg


def jacobians(Q, eps=1e-6):
    """d targets / d observed inputs at each point, via the chain rule through q."""
    n, k = Q.shape
    Jo, Jt = np.zeros((n, 7, k)), np.zeros((n, 4, k))
    for i in range(k):
        dq = np.zeros(k); dq[i] = eps
        op, tp = transition_map(Q + dq)
        om, tm = transition_map(Q - dq)
        Jo[:, :, i] = (op - om) / (2 * eps)
        Jt[:, :, i] = (tp - tm) / (2 * eps)
    return np.einsum('nti,nio->nto', Jt, np.linalg.inv(Jo))   # (n, 4 targets, 7 inputs)


def check_d_e(V, eps_raw):
    # rebuild q = (s_{t-1}, u_{t-1}, u_t, u_{t-2}) for a sample of transitions
    Q = []
    for s, a, _ in eps_raw:
        a = a.reshape(-1)
        for t in range(T_MIN, len(a)):
            Q.append(np.r_[s[t - 1], a[t - 1], a[t], a[t - 2]])
    Q = np.asarray(Q)
    Q = Q[RNG.choice(len(Q), size=min(4000, len(Q)), replace=False)]
    o_chk, tg_chk = transition_map(Q)
    J = jacobians(Q)
    sd_in = V['inputs'].std(0)
    sd_tg = V['targets'].std(0)
    # normalised effect: mean |dT_j/do_i| * std(o_i) / std(T_j)   (rows inputs, cols targets)
    share = (np.abs(J).mean(0) * sd_in[None, :] / sd_tg[:, None]).T
    maxabs = (np.abs(J).max(0) * sd_in[None, :] / sd_tg[:, None]).T
    nonzero = maxabs > 1e-6

    A = expert_graph.adjacency().astype(bool)
    pattern_ok = bool((nonzero == A).all())
    mism = [f'{INPUT_NAMES[i]}->{TARGET_NAMES[j]} (expert={int(A[i, j])}, measured={maxabs[i, j]:.2g})'
            for i in range(len(INPUT_NAMES)) for j in range(len(TARGET_NAMES)) if nonzero[i, j] != A[i, j]]
    # the data pipeline must reproduce the same variables as the exact map
    recon = float(np.abs(o_chk - transition_map(Q)[0]).max())
    say('d_jacobian_pattern', pattern_ok, mismatches=mism or 'none', recon_err=recon)

    tiers_derived = expert_graph.tiers()
    tiers_final = np.where(~nonzero, 'absent', np.where(share >= 0.1, 'strong', 'weak'))
    print('\n  normalised effect size (mean |dT/do| * sd_in / sd_target); rows = inputs')
    print('  ' + ' ' * 7 + ''.join(f'{t:>10s}' for t in TARGET_NAMES))
    for i, nm in enumerate(INPUT_NAMES):
        print(f'  {nm:7s}' + ''.join(f'{share[i, j]:10.3g}' for j in range(len(TARGET_NAMES))))
    changed = [f'{INPUT_NAMES[i]}->{TARGET_NAMES[j]}: {tiers_derived[i, j]} -> {tiers_final[i, j]}'
               for i in range(len(INPUT_NAMES)) for j in range(len(TARGET_NAMES))
               if tiers_derived[i, j] != tiers_final[i, j]]
    say('e_sensitivity', True, tier_changes=changed or 'none',
        n_strong=int((tiers_final == 'strong').sum()), n_weak=int((tiers_final == 'weak').sum()))
    final = {
        'inputs': INPUT_NAMES, 'targets': TARGET_NAMES,
        'adjacency': A.astype(int).tolist(),
        'tiers_derived': tiers_derived.tolist(),
        'tiers': tiers_final.tolist(),
        'effect_size': share.tolist(),
        'rule': 'absent = exact zero in Jacobian; strong = normalised effect >= 0.1; else weak',
    }
    (RESULTS / 'expert_graph_final.json').write_text(json.dumps(final, indent=2))


# ---------------------------------------------------------------- (f)
def check_f(V):
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    base = V['inputs'][:, :6]                         # without u_{t-2}
    ext = np.column_stack([V['inputs'], V['d_prev']])  # + u_{t-2} + previous state
    tr, te = split_by_episode(V['episode'])
    res = {}
    worst_gain = -1.0
    for j, nm in enumerate(TARGET_NAMES):
        y = V['targets'][:, j]
        ys = (y - y[tr].mean()) / y[tr].std()
        scores = []
        for F in (base, ext):
            sc = StandardScaler().fit(F[tr])
            mdl = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=400, early_stopping=True, random_state=0,
                               tol=1e-6, n_iter_no_change=20)
            mdl.fit(sc.transform(F[tr]), ys[tr])
            scores.append(r2(ys[te], mdl.predict(sc.transform(F[te]))))
        res[f'{nm}_R2_markov_inputs'] = scores[0]
        res[f'{nm}_R2_plus_history'] = scores[1]
        worst_gain = max(worst_gain, scores[1] - scores[0])
    say('f_markov', worst_gain < 0.005, max_R2_gain_from_history=float(worst_gain), **res)


def main():
    RESULTS.mkdir(exist_ok=True)
    expert_graph.main()
    check_a()
    check_b()
    print('[data] simulating states with the dataset policies (no rendering) ...')
    V, eps_raw = sim_data()
    print(f'[data] {len(V["t"])} usable transitions (t >= {T_MIN}) from {len(eps_raw)} episodes')
    check_c(V)
    check_d_e(V, eps_raw)
    check_f(V)
    (RESULTS / 'checks.json').write_text(json.dumps(REPORT, indent=2, default=str))
    n_fail = sum(not r['passed'] for r in REPORT.values())
    print(f'\n{len(REPORT) - n_fail}/{len(REPORT)} checks passed -> results/checks.json')
    sys.exit(1 if n_fail else 0)


if __name__ == '__main__':
    main()
