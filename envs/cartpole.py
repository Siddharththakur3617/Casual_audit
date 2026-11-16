"""Cart-pole spec: variables, edge features, registry attributes (same layout as envs/acrobot.py).

Variables (one transition = 0.04 s; hand-derived graph in checks/expert_graph.py):
  inputs  x_t, vbar_t, theta_t, wbar_t, u_t, u_{t-1}, u_{t-2}
  targets dx, dvbar, dtheta, dwbar   (value at t+1 minus value at t)
vbar and wbar are finite-difference velocities between consecutive frames,
because that is the only velocity two frames can show.
Dynamics and renderer: sim/cartpole.py.
"""
from __future__ import annotations

import numpy as np

from sim.cartpole import FRAME_SKIP, H, PHYS, accelerations, euler_substep, macro_step  # noqa: F401

NAME = 'cartpole'
DT = H * FRAME_SKIP                # one transition = 0.04 s
T_MIN = 5                          # first usable transition: 5 history latents need frames 0..t, u_{t-5}
PHYS_KEYS = ('M', 'm', 'l', 'g')
INPUT_NAMES = ['x', 'vbar', 'theta', 'wbar', 'u_t', 'u_tm1', 'u_tm2']
TARGET_NAMES = ['dx', 'dvbar', 'dtheta', 'dwbar']
STATE_NAMES = ['x', 'vbar', 'theta', 'wbar']
ACTION_COLS = [4, 5, 6]            # u_t, u_{t-1}, u_{t-2}
DECOY_SOURCE = [4, 2]              # shuffled copies of u_t and theta
DECOY_NAMES = ['decoy_u_t', 'decoy_theta']
ACTION_EPS = 0.1                   # newtons
EXTRA_KEYS = ('xdot', 'thetadot')
CONFIG = 'cartpole.yaml'           # configs/envs/
RECALIBRATE_BN = False             # stats are fine (recalibration changes losses < 0.001); keeps old results


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


def edge_features(inputs):
    """Edge-scoring design matrix: the inputs themselves, one column per node."""
    return inputs, None
