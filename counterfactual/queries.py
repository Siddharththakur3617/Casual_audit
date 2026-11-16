"""Counterfactual queries for one dataset split.

A query window is what the JEPA sees at time t: states s_{t-5..t+K} and actions u_{t-5..t+K-1}
(K = horizon; K = 1 for one-step queries: 7 frames, 5 history latents, a 6-action context).

Interventions (each with sign +1 and -1, size delta = kappa * sd):
  node i (a graph input at t), all other inputs at t held fixed:
    action node u_{t-lag}: that action is changed. For lag >= 1 the start state s_{t-5} is re-solved
      so the frame states d_t (x, vbar, theta, wbar / th1, w1bar, ...) stay exactly as they were.
      u_t needs no solve.
    state node: d_t' = d_t + delta e_i. s_{t-5} is re-solved so the simulated window reaches d_t'.
  physics p: the episode physics is changed, and s_{t-5} is re-solved so that d_t and the actions stay fixed
    (the direct physics -> change path: physics is the hidden confounder).
The solve is a square system (k unknowns, k frame states), done with vectorised damped Newton. Every window
is then simulated with env.macro_step under its (counterfactual) physics, so the 5-step history the model
sees is physically consistent. Variables come from env.episode_variables, the study's own definitions.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import yaml

import envs
import paths
from common import episode_physics

HIST = 5                      # frames before t in a window (s_{t-5} .. s_{t-1})
SOLVE_TOL = 1e-11             # max |residual| / sd(frame state) for convergence
SOLVE_ITERS = 40


# ---------------------------------------------------------------- config and limits
def env_cfg(env=None):
    env = env or envs.current()
    return yaml.safe_load((paths.ENV_CONFIGS / env.CONFIG).read_text())


def phys_sd(env=None):
    """sd of each physics parameter's configured uniform range (the unit of a physics intervention)."""
    r = env_cfg(env)['physics_ranges']
    env = env or envs.current()
    return np.array([(r[k][1] - r[k][0]) / np.sqrt(12.0) for k in env.PHYS_KEYS])


def phys_range(env=None):
    r = env_cfg(env)['physics_ranges']
    env = env or envs.current()
    return np.array([r[k] for k in env.PHYS_KEYS], dtype=float)          # (n_phys, 2)


def action_range(env=None):
    return tuple(float(v) for v in env_cfg(env)['environment']['action_range'])


def valid_states(S, env=None):
    """S (n, W, k) -> (n,) True where every state of the window is one the dataset could contain."""
    env = env or envs.current()
    ok = np.isfinite(S).all(axis=(1, 2))
    if env.NAME == 'cartpole':
        d = env_cfg(env)['data']
        ok &= (np.abs(S[..., 0]) <= d['max_abs_x']).all(1) & (np.abs(S[..., 2]) <= d['max_abs_theta']).all(1)
    else:
        ok &= (np.abs(S[..., 1]) <= env.MAX_W1).all(1) & (np.abs(S[..., 3]) <= env.MAX_W2).all(1)
    return ok


# ---------------------------------------------------------------- windows
def load_windows(data_dir, split='test', n_max=3000, seed=0, horizon=1, with_obs=False):
    """Every usable window (t >= T_MIN, t + horizon <= T) of a split, at most n_max of them (seed fixed).
    Returns dict: S0 stored states (n, HIST+1+horizon, k), U (n, HIST+horizon), phys (n, n_phys),
    episode (n,), t (n,), and obs (n, W, h, w, 3) stored frames if with_obs."""
    env = envs.use_data(data_dir)
    W = HIST + 1 + horizon
    items = []
    with h5py.File(Path(data_dir) / f'{split}.hdf5', 'r') as f:
        grp = f['episodes']
        keys = sorted(grp.keys(), key=int)
        for e, key in enumerate(keys):
            T = len(grp[key]['actions'])
            items += [(e, key, t) for t in range(env.T_MIN, T - horizon + 1)]
        rng = np.random.default_rng(seed)
        if len(items) > n_max:
            pick = np.sort(rng.choice(len(items), n_max, replace=False))
            items = [items[i] for i in pick]
        S, U, P, O = [], [], [], []
        cache = {}
        for e, key, t in items:
            if key not in cache:
                ep = grp[key]
                cache = {key: (ep['states'][:].astype(np.float64), ep['actions'][:].reshape(-1).astype(np.float64),
                               episode_physics(ep.attrs, env), ep['observations'] if with_obs else None)}
            st, ac, ph, ob = cache[key]
            S.append(st[t - HIST:t - HIST + W])
            U.append(ac[t - HIST:t - HIST + W - 1])
            P.append([ph[k] for k in env.PHYS_KEYS])
            if with_obs:
                O.append(ob[t - HIST:t - HIST + W])
    out = {'S0': np.asarray(S), 'U': np.asarray(U), 'phys': np.asarray(P, float),
           'episode': np.array([i[0] for i in items]), 't': np.array([i[2] for i in items])}
    if with_obs:
        out['obs'] = np.asarray(O)
    return out


def load_obs(data_dir, split, episode, t, horizon=1):
    """Stored frames o_{t-5..t+horizon} of the given windows (episode = index into the sorted episode keys)."""
    W = HIST + 1 + horizon
    out = []
    with h5py.File(Path(data_dir) / f'{split}.hdf5', 'r') as f:
        grp = f['episodes']
        keys = sorted(grp.keys(), key=int)
        for e, tt in zip(episode, t):
            out.append(grp[keys[e]]['observations'][tt - HIST:tt - HIST + W])
    return np.asarray(out)


# ---------------------------------------------------------------- simulation and variables
def phys_dict(P, env=None):
    env = env or envs.current()
    return {k: P[:, i] for i, k in enumerate(env.PHYS_KEYS)}


def simulate(s0, U, P, env=None):
    """s0 (n, k), U (n, L), P (n, n_phys) -> states (n, L+1, k), each step env.macro_step at its row's physics."""
    env = env or envs.current()
    ph = phys_dict(P, env)
    out = [np.asarray(s0, np.float64)]
    for j in range(U.shape[1]):
        out.append(env.macro_step(out[-1], U[:, j], **ph))
    return np.stack(out, axis=1)


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def angle_mask(env=None):
    """(k,) True for frame-state columns that are wrapped angles."""
    env = env or envs.current()
    m = np.zeros(len(env.STATE_NAMES), bool)
    if env.NAME == 'acrobot':
        m[[0, 2]] = True
    return m


def frame_state(s_prev, s_t, env=None):
    """Frame states d_t (x, vbar, theta, wbar / th1, w1bar, th2, w2bar, p, vpbar) from s_{t-1}, s_t,
    vectorised. Must equal env.episode_variables (checked by check_frame_state)."""
    env = env or envs.current()
    if env.NAME == 'cartpole':
        x, th = s_t[..., 0], s_t[..., 2]
        return np.stack([x, (x - s_prev[..., 0]) / env.DT, th, (th - s_prev[..., 2]) / env.DT], axis=-1)
    cols = []
    for k, ang in zip(env.POS, env.ANGLE):
        q, qp = s_t[..., k], s_prev[..., k]
        cols += [_wrap(q) if ang else q, (_wrap(q - qp) if ang else q - qp) / env.DT]
    return np.stack(cols, axis=-1)


def window_variables(S, U, env=None):
    """S (n, W, k), U (n, W-1) -> inputs (n, K, n_in), targets (n, K, n_tg) at t..t+K-1 (K = W - HIST - 1),
    computed with env.episode_variables window by window (the study's exact definitions)."""
    env = env or envs.current()
    I, T = [], []
    for s, u in zip(S, U):
        v = env.episode_variables(s, u, t_min=HIST)
        I.append(v['inputs'])
        T.append(v['targets'])
    return np.asarray(I), np.asarray(T)


def check_frame_state(S, U, env=None):
    """Max |frame_state - episode_variables| over the windows (should be ~1e-12)."""
    I, _ = window_variables(S, U, env)
    d = frame_state(S[:, HIST - 1], S[:, HIST], env)
    k = d.shape[-1]
    diff = d - I[:, 0, :k]
    diff[:, angle_mask(env)] = _wrap(diff[:, angle_mask(env)])
    return float(np.abs(diff).max())


# ---------------------------------------------------------------- the solve
def solve_start(s0, U_hist, P, d_target, d_sd, env=None):
    """Find s'_{t-5} with frame_state(simulate(s', U_hist, P)[-2:]) = d_target. U_hist (n, HIST) = u_{t-5..t-1}.
    Damped Newton, finite-difference Jacobian. Returns (s', converged (n,), max scaled residual (n,))."""
    env = env or envs.current()
    am = angle_mask(env)
    n, k = s0.shape
    s = np.asarray(s0, np.float64).copy()
    r = resid_sub(s, U_hist, P, d_target, d_sd, env, am)
    nrm = np.abs(r).max(1)
    eps = 1e-7
    for _ in range(SOLVE_ITERS):
        active = nrm > SOLVE_TOL
        if not active.any():
            break
        idx = np.where(active)[0]
        sa, ra = s[idx], r[idx]
        J = np.empty((len(idx), k, k))
        for j in range(k):
            dp = np.zeros(k); dp[j] = eps
            J[:, :, j] = (resid_sub(sa + dp, U_hist[idx], P[idx], d_target[idx], d_sd, env, am)
                          - resid_sub(sa - dp, U_hist[idx], P[idx], d_target[idx], d_sd, env, am)) / (2 * eps)
        try:
            step = -np.linalg.solve(J, ra[..., None])[..., 0]
        except np.linalg.LinAlgError:
            step = -np.linalg.lstsq(J.reshape(-1, k), ra.reshape(-1), rcond=None)[0].reshape(len(idx), k)
        alpha = np.ones(len(idx))
        best_s, best_n = sa.copy(), np.abs(ra).max(1)
        for _ in range(8):                       # backtracking: accept the first step that lowers the residual
            cand = sa + alpha[:, None] * step
            rc = resid_sub(cand, U_hist[idx], P[idx], d_target[idx], d_sd, env, am)
            nc = np.abs(rc).max(1)
            better = np.isfinite(nc) & (nc < best_n)
            best_s[better], best_n[better] = cand[better], nc[better]
            alpha[~better] *= 0.5
            if better.all():
                break
        s[idx] = best_s
        r[idx] = resid_sub(best_s, U_hist[idx], P[idx], d_target[idx], d_sd, env, am)
        nrm[idx] = np.abs(r[idx]).max(1)
    return s, nrm <= SOLVE_TOL, nrm


def resid_sub(s, U_hist, P, d_target, d_sd, env, am):
    X = simulate(s, U_hist, P, env)
    r = frame_state(X[:, -2], X[:, -1], env) - d_target
    r[:, am] = _wrap(r[:, am])
    return r / d_sd


# ---------------------------------------------------------------- interventions
def variant_list(env, rollout=False):
    """[(kind, index, sign)]: ('factual', -1, 0), ('node', input column, +-1), ('phys', physics index, +-1).
    Rollout mode: factual plus u_t and u_{t-1} only."""
    V = [('factual', -1, 0)]
    nodes = env.ACTION_COLS[:2] if rollout else list(range(len(env.INPUT_NAMES)))
    for i in nodes:
        V += [('node', int(i), 1), ('node', int(i), -1)]
    if not rollout:
        for p in range(len(env.PHYS_KEYS)):
            V += [('phys', p, 1), ('phys', p, -1)]
    return V


def variant_name(env, v):
    kind, i, sg = v
    if kind == 'factual':
        return 'factual'
    name = env.INPUT_NAMES[i] if kind == 'node' else env.PHYS_KEYS[i]
    return f'{name}{"+" if sg > 0 else "-"}'


def build_queries(W, kappa, in_sd, rollout=False, env=None, verbose=True):
    """W: load_windows output. in_sd: train sd of the graph inputs (n_in,).
    Returns dict with, per variant v: S (n_var, n, W, k), U (n_var, n, W-1), phys (n_var, n, n_phys),
    valid (n_var, n), inputs (n_var, n, K, n_in), targets (n_var, n, K, n_tg), delta (n_var,), residual (n_var, n)."""
    env = env or envs.current()
    V = variant_list(env, rollout)
    k = len(env.STATE_NAMES)
    am = angle_mask(env)
    S_f = simulate(W['S0'][:, 0], W['U'], W['phys'], env)             # re-simulated factual
    d_t = frame_state(S_f[:, HIST - 1], S_f[:, HIST], env)
    d_sd = in_sd[:k]
    lo_u, hi_u = action_range(env)
    prange, psd = phys_range(env), phys_sd(env)
    out = {k_: [] for k_ in ('S', 'U', 'phys', 'valid', 'residual', 'delta')}
    for v in V:
        kind, i, sg = v
        U, P, d_tar, s_init = W['U'].copy(), W['phys'].copy(), d_t.copy(), W['S0'][:, 0].copy()
        ok = np.ones(len(U), bool)
        solve, delta = True, 0.0
        if kind == 'factual':
            solve = False
        elif kind == 'node' and i in env.ACTION_COLS:
            lag = env.ACTION_COLS.index(i)
            delta = kappa * in_sd[i]
            U[:, HIST - lag] += sg * delta
            ok &= (U[:, HIST - lag] >= lo_u) & (U[:, HIST - lag] <= hi_u)
            solve = lag > 0
        elif kind == 'node':
            delta = kappa * in_sd[i]
            d_tar[:, i] += sg * delta
            if am[i]:
                d_tar[:, i] = _wrap(d_tar[:, i])
        else:
            delta = kappa * psd[i]
            P[:, i] += sg * delta
            ok &= (P[:, i] >= prange[i, 0]) & (P[:, i] <= prange[i, 1])
        res = np.zeros(len(U))
        if solve:
            s_init, conv, res = solve_start(s_init, U[:, :HIST], P, d_tar, d_sd, env)
            ok &= conv
        S = simulate(s_init, U, P, env)
        ok &= valid_states(S, env)
        for key, val in (('S', S), ('U', U), ('phys', P), ('valid', ok), ('residual', res), ('delta', delta)):
            out[key].append(val)
        if verbose:
            print(f'    [query] {variant_name(env, v):10s} valid {ok.mean():.3f}'
                  + (f'  max residual {res[ok].max() if ok.any() else float("nan"):.1e}' if solve else ''), flush=True)
    out = {k_: np.asarray(v_) for k_, v_ in out.items()}
    I, T = [], []
    for S, U in zip(out['S'], out['U']):
        a, b = window_variables(S, U, env)
        I.append(a); T.append(b)
    out['inputs'], out['targets'] = np.asarray(I), np.asarray(T)
    out['variants'] = np.array([variant_name(env, v) for v in V])
    out['kind'] = np.array([v[0] for v in V])
    out['index'] = np.array([v[1] for v in V])
    out['sign'] = np.array([v[2] for v in V])
    out['episode'], out['t'] = W['episode'], W['t']
    out['kappa'] = np.float64(kappa)
    return out


def check_queries(Q, in_sd, env=None):
    """Did each intervention move only its own input, by exactly delta? Returns per-variant max errors (sd units)."""
    env = env or envs.current()
    f = Q['inputs'][0, :, 0]
    rep = {}
    am = np.zeros(f.shape[1], bool)
    am[:len(env.STATE_NAMES)] = angle_mask(env)
    for v in range(1, len(Q['variants'])):
        ok = Q['valid'][v]
        if not ok.any():
            rep[str(Q['variants'][v])] = None
            continue
        d = Q['inputs'][v, ok, 0] - f[ok]
        d[:, am] = _wrap(d[:, am])
        want = np.zeros_like(d)
        if Q['kind'][v] == 'node':
            want[:, Q['index'][v]] = Q['sign'][v] * Q['delta'][v]
        rep[str(Q['variants'][v])] = float((np.abs(d - want) / in_sd).max())
    return rep


# ---------------------------------------------------------------- rendering
class _WindowsMixin:
    def windows(self, S, P, ref_S=None, ref_P=None, ref_frames=None):
        """Frames for windows S (n, W, k) at physics P (n, n_phys). Frames whose state and physics equal the
        reference window's exactly are copied from ref_frames instead of rendered again."""
        n, W, k = S.shape
        frames = np.empty((n, W, self.size, self.size, 3), np.uint8)
        need = np.ones((n, W), bool)
        if ref_frames is not None:
            same = (S == ref_S).all(-1) & (P == ref_P).all(-1)[:, None]
            frames[same] = ref_frames[same]
            need = ~same
        if need.any():
            frames[need] = self(S[need], np.repeat(P[:, None], W, 1)[need])
        return frames

    def close(self):
        pass


class TorchRenderer(_WindowsMixin):
    """The sim renderers (sim/cartpole.py, sim/acrobot.py) batched on the GPU: same sub-sample grid (float64),
    same shapes, same paint order; per-frame trigonometry is done with numpy exactly as in the sim, so the frames
    are identical (checked against the sim renderer in the study's verification)."""

    def __init__(self, env=None, device='cuda', batch=32):
        import torch
        self.torch = torch
        self.env = env or envs.current()
        self.size = int(env_cfg(self.env)['environment']['image_size'])
        self.dev = torch.device(device)
        self.batch = batch
        if self.env.NAME == 'cartpole':
            from sim import cartpole as sm
        else:
            from sim import acrobot as sm
        self.sm = sm
        r = sm.Renderer(self.size)
        self.ss = r.ss
        self.gx = torch.as_tensor(r.gx, device=self.dev)[None]
        self.gy = torch.as_tensor(r.gy, device=self.dev)[None]

    def _col(self, rgb):
        return self.torch.as_tensor(np.asarray(rgb, np.float32), device=self.dev)

    def _paint(self, img, mask, rgb):
        return self.torch.where(mask[..., None], self._col(rgb), img)

    def _cartpole(self, states, phys):
        t, sm, gx, gy = self.torch, self.sm, self.gx, self.gy
        T = lambda a: t.as_tensor(np.asarray(a, np.float64), device=self.dev)[:, None, None]  # noqa: E731
        li = self.env.PHYS_KEYS.index('l')
        cx = np.array([s[0] * sm.SCALE + sm.SCENE_W / 2.0 for s in states])
        c = np.array([np.cos(s[2]) for s in states])
        sn = np.array([np.sin(s[2]) for s in states])
        pole_len = np.array([2.0 * p[li] * sm.SCALE for p in phys])
        cx_, c_, s_, pl_ = T(cx), T(c), T(sn), T(pole_len)
        img = self._col(sm.WHITE).expand(len(states), *gx.shape[1:], 3).clone()
        img = self._paint(img, (t.abs(gy - sm.CART_Y) <= 0.5).expand_as(img[..., 0]), sm.BLACK)
        img = self._paint(img, (t.abs(gx - cx_) <= sm.CART_W / 2) & (t.abs(gy - sm.CART_Y) <= sm.CART_H / 2), sm.BLACK)
        dx, dy = gx - cx_, gy - (sm.CART_Y + sm.AXLE_OFFSET)
        u = dx * c_ - dy * s_
        v = dx * s_ + dy * c_
        img = self._paint(img, (t.abs(u) <= sm.POLE_W / 2) & (v >= -sm.POLE_W / 2) & (v <= pl_ - sm.POLE_W / 2),
                          sm.POLE_RGB)
        img = self._paint(img, dx ** 2 + dy ** 2 <= (sm.POLE_W / 2) ** 2, sm.AXLE_RGB)
        return img

    def _acrobot(self, states, phys):
        t, sm, gx, gy = self.torch, self.sm, self.gx, self.gy
        T = lambda a: t.as_tensor(np.asarray(a, np.float64), device=self.dev)[:, None, None]  # noqa: E731
        i1, i2 = self.env.PHYS_KEYS.index('l1'), self.env.PHYS_KEYS.index('l2')
        ex, ey, tx, ty, p = [], [], [], [], []
        for s, ph in zip(states, phys):
            th1, th2, l1, l2 = s[0], s[2], ph[i1], ph[i2]
            e_x, e_y = l1 * np.sin(th1), -l1 * np.cos(th1)
            ex.append(e_x); ey.append(e_y)
            tx.append(e_x + l2 * np.sin(th1 + th2)); ty.append(e_y - l2 * np.cos(th1 + th2))
            p.append(s[4])
        ex, ey, tx, ty, p = (np.asarray(a, np.float64) for a in (ex, ey, tx, ty, p))

        def segment(ax, ay, bx, by, r):
            dx, dy = bx - ax, by - ay
            L2 = np.maximum(dx * dx + dy * dy, 1e-12)
            ax_, ay_, dx_, dy_, L2_ = T(ax), T(ay), T(dx), T(dy), T(L2)
            tt = t.clamp(((gx - ax_) * dx_ + (gy - ay_) * dy_) / L2_, 0, 1)
            return (gx - ax_ - tt * dx_) ** 2 + (gy - ay_ - tt * dy_) ** 2 <= r * r

        n = len(states)
        zero = np.zeros(n)
        img = self._col(sm.WHITE).expand(n, *gx.shape[1:], 3).clone()
        img = self._paint(img, (t.abs(gy - sm.TRACK_Y) <= 0.02).expand_as(img[..., 0]), sm.BLACK)
        img = self._paint(img, (t.abs(gx - T(p)) <= sm.BLOCK_W / 2) & (t.abs(gy - sm.TRACK_Y) <= sm.BLOCK_H / 2),
                          sm.BLOCK_RGB)
        img = self._paint(img, segment(zero, zero, ex, ey, sm.LINK_W / 2), sm.LINK1_RGB)
        img = self._paint(img, segment(ex, ey, tx, ty, sm.LINK_W / 2), sm.LINK2_RGB)
        img = self._paint(img, (gx ** 2 + gy ** 2 <= sm.JOINT_R ** 2).expand_as(img[..., 0]), sm.BLACK)
        img = self._paint(img, (gx - T(ex)) ** 2 + (gy - T(ey)) ** 2 <= sm.JOINT_R ** 2, sm.JOINT_RGB)
        return img

    def __call__(self, states, phys):
        """states (m, k), phys (m, n_phys) -> (m, h, w, 3) uint8."""
        out = []
        draw = self._cartpole if self.env.NAME == 'cartpole' else self._acrobot
        k, sz = self.ss, self.size
        for i in range(0, len(states), self.batch):
            img = draw(states[i:i + self.batch], phys[i:i + self.batch])
            img = img.reshape(len(img), sz, k, sz, k, 3).mean((2, 4))
            out.append(self.torch.round(img).to(self.torch.uint8).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, sz, sz, 3), np.uint8)


def make_renderer(env=None, workers=None):
    """GPU renderer when CUDA is visible, else the process-pool sim renderer."""
    import torch
    return TorchRenderer(env) if torch.cuda.is_available() else Renderer(env, workers=workers)


# ---------------------------------------------------------------- rendering (process pool, CPU fallback)
_SIM = None


def _init_render(env_name, image_size):
    global _SIM
    env = envs.use(env_name)
    if env.NAME == 'cartpole':
        from sim.cartpole import CartPole
        _SIM = CartPole(image_size=image_size)
    else:
        from sim.acrobot import Acrobot
        _SIM = Acrobot(image_size=image_size)


def _render_block(args):
    states, phys = args
    env = envs.current()
    return np.stack([_SIM.reset_to_state(s, {k: float(p[i]) for i, k in enumerate(env.PHYS_KEYS)})[0]
                     for s, p in zip(states, phys)])


class Renderer(_WindowsMixin):
    """Renders (state, physics) pairs in a process pool with the dataset's own renderer (sim reset_to_state)."""

    def __init__(self, env=None, workers=None, block=64):
        self.env = env or envs.current()
        self.size = int(env_cfg(self.env)['environment']['image_size'])
        self.workers = workers or max(1, min(24, (os.cpu_count() or 2) - 2))
        self.block = block
        self.pool = ProcessPoolExecutor(self.workers, initializer=_init_render, initargs=(self.env.NAME, self.size))

    def __call__(self, states, phys):
        """states (m, k), phys (m, n_phys) -> (m, h, w, 3) uint8."""
        if len(states) == 0:
            return np.zeros((0, self.size, self.size, 3), np.uint8)
        blocks = [(states[i:i + self.block], phys[i:i + self.block]) for i in range(0, len(states), self.block)]
        return np.concatenate(list(self.pool.map(_render_block, blocks)))

    def close(self):
        self.pool.shutdown()
