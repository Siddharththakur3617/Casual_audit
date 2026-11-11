"""Shared constants and helpers. The constants and dynamics here are cart-pole's (envs/cartpole.py
re-exports them); the generic helpers at the bottom work for any environment in envs/.

Variables (see expert_graph.py):
  inputs  x_t, vbar_t, theta_t, wbar_t, u_t, u_{t-1}, u_{t-2}
  targets dx, dvbar, dtheta, dwbar   (value at t+1 minus value at t)
vbar and wbar are finite-difference velocities between consecutive frames,
because that is the only velocity two frames can show.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / 'results'

H = 0.02          # physics sub-step (s), env default
FRAME_SKIP = 2
DT = H * FRAME_SKIP   # one transition = 0.04 s
T_MIN = 5         # first usable transition: 5 history latents need frames 0..t, u_{t-5}

PHYS = dict(M=1.0, m=0.1, l=0.5, g=9.8)

INPUT_NAMES = ['x', 'vbar', 'theta', 'wbar', 'u_t', 'u_tm1', 'u_tm2']
TARGET_NAMES = ['dx', 'dvbar', 'dtheta', 'dwbar']
STATE_NAMES = ['x', 'vbar', 'theta', 'wbar']


def accelerations(theta, theta_dot, u, x_dot=0.0, M=1.0, m=0.1, l=0.5, g=9.8,
                  b_c=0.0, b_p=0.0):
    """Cart-pole accelerations (Barto et al. 1983 / Florian 2007 form, viscous friction).
    l is the half-length of the pole; theta = 0 is upright, positive tilts right."""
    sin_t, cos_t = np.sin(theta), np.cos(theta)
    total = M + m
    ml = m * l
    temp = (u - b_c * x_dot + ml * theta_dot ** 2 * sin_t) / total
    theta_acc = (g * sin_t - cos_t * temp - b_p * theta_dot / ml) / (
        l * (4.0 / 3.0 - m * cos_t ** 2 / total))
    x_acc = temp - ml * theta_acc * cos_t / total
    return x_acc, theta_acc


def euler_substep(s, u, h=H, **phys):
    """One explicit-Euler sub-step. s: (..., 4) = (x, x_dot, theta, theta_dot)."""
    s = np.asarray(s, dtype=np.float64)
    x, xd, th, thd = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    xa, tha = accelerations(th, thd, u, x_dot=xd, **phys)
    return np.stack([x + h * xd, xd + h * xa, th + h * thd, thd + h * tha], axis=-1)


def macro_step(s, u, n=FRAME_SKIP, h=H, **phys):
    """One stored transition: the force is held for n sub-steps."""
    for _ in range(n):
        s = euler_substep(s, u, h=h, **phys)
    return s


def episode_variables(states, actions, t_min=T_MIN):
    """Inputs/targets for every usable transition of one episode.

    states : (T+1, 4) true states s_0..s_T
    actions: (T,) or (T, 1) forces u_0..u_{T-1} (u_t moves s_t -> s_{t+1})
    Returns dict with inputs (n, 7), targets (n, 4), t (n,), and extras used by
    the checks (instantaneous velocities, previous-step variables).
    """
    states = np.asarray(states, dtype=np.float64)
    u = np.asarray(actions, dtype=np.float64).reshape(-1)
    T = len(u)
    x, th = states[:, 0], states[:, 2]
    vbar = np.full(T + 1, np.nan)
    wbar = np.full(T + 1, np.nan)
    vbar[1:] = (x[1:] - x[:-1]) / DT
    wbar[1:] = (th[1:] - th[:-1]) / DT
    ts = np.arange(t_min, T)              # needs s_{t+1}, i.e. t <= T-1
    if len(ts) == 0:
        return None
    d = np.stack([x, vbar, th, wbar], axis=1)      # (T+1, 4)
    inputs = np.column_stack([d[ts], u[ts], u[ts - 1], u[ts - 2]])
    targets = d[ts + 1] - d[ts]
    return {
        't': ts,
        'inputs': inputs,
        'targets': targets,
        'd_prev': d[ts - 1],
        'xdot': states[ts, 1],
        'thetadot': states[ts, 3],
        'x_prev': x[ts - 1],
        'theta_prev': th[ts - 1],
    }


def stack_episodes(episodes, t_min=T_MIN, env=None):
    """episodes: iterable of (states (T+1,k), actions (T,)). Concatenate variables.
    env: environment spec (default: the active one, envs.current()); cart-pole uses episode_variables above."""
    if env is None:
        import envs
        env = envs.current()
    out, ep_index = {}, []
    for k, (s, a) in enumerate(episodes):
        v = env.episode_variables(s, a, t_min=t_min)
        if v is None:
            continue
        for key, val in v.items():
            out.setdefault(key, []).append(val)
        ep_index.append(np.full(len(v['t']), k))
    out = {k: np.concatenate(v) for k, v in out.items()}
    out['episode'] = np.concatenate(ep_index)
    return out


PHYS_KEYS = ('M', 'm', 'l', 'g')


def episode_physics(attrs, env=None):
    """An episode's physics dict from its HDF5 attrs phys_*, falling back to the env's PHYS (fixed-physics data)."""
    if env is None:
        import envs
        env = envs.current()
    return {k: float(attrs.get(f'phys_{k}', env.PHYS[k])) for k in env.PHYS_KEYS}


def load_split_episodes(data_dir, split='train', with_physics=False):
    """Read per-episode HDF5 (DiscreteHDF5TrajectoryDataset format): (states, actions) per episode,
    or (states, actions, physics dict) with with_physics=True. Activates the dataset's environment."""
    import h5py
    import envs
    envs.use_data(data_dir)
    eps = []
    with h5py.File(Path(data_dir) / f'{split}.hdf5', 'r') as f:
        grp = f['episodes']
        for key in sorted(grp.keys(), key=int):
            ep = (grp[key]['states'][:], grp[key]['actions'][:].reshape(-1))
            eps.append(ep + (episode_physics(grp[key].attrs),) if with_physics else ep)
    return eps
