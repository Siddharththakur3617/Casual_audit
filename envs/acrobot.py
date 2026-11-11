"""Acrobot spec: variables, graph, edge features, dataset policies.

Variables (one transition = 0.04 s):
  inputs  th1, w1bar, th2, w2bar, p, vpbar, u_t, u_{t-1}, u_{t-2}
  targets dth1, dw1bar, dth2, dw2bar, dp, dvpbar   (value at t+1 minus value at t)
w1bar, w2bar, vpbar are finite-difference velocities between consecutive frames. The angle inputs are
wrapped to (-pi, pi] (all the image can show) and the angle changes use the wrapped difference.
The distractor (p, vpbar) is independent of the arm and of the action, so every edge between the two
is truly absent.
"""
from __future__ import annotations

import numpy as np

from sim.acrobot import FRAME_SKIP, H, MAX_W1, MAX_W2, PHYS, macro_step  # noqa: F401

NAME = 'acrobot'
DT = H * FRAME_SKIP
T_MIN = 5
PHYS_KEYS = ('m1', 'm2', 'l1', 'l2', 'g')
STATE_NAMES = ['th1', 'w1bar', 'th2', 'w2bar', 'p', 'vpbar']
INPUT_NAMES = STATE_NAMES + ['u_t', 'u_tm1', 'u_tm2']
TARGET_NAMES = ['dth1', 'dw1bar', 'dth2', 'dw2bar', 'dp', 'dvpbar']
ARM_INPUTS = ['th1', 'w1bar', 'th2', 'w2bar', 'u_t', 'u_tm1', 'u_tm2']
ARM_TARGETS = ['dth1', 'dw1bar', 'dth2', 'dw2bar']
DISTRACTOR_INPUTS = ['p', 'vpbar']
DISTRACTOR_TARGETS = ['dp', 'dvpbar']
ACTION_COLS = [6, 7, 8]
DECOY_SOURCE = [6, 2]              # input NODES copied as shuffled decoys: u_t and th2
DECOY_NAMES = ['decoy_u_t', 'decoy_th2']
ACTION_EPS = 0.03                  # N m (0.1 N on cart-pole's +-10 range, scaled to +-3)
EXTRA_KEYS = ('true_vel',)
CONFIG = 'causal_acrobot.yaml'
RECALIBRATE_BN = True              # BN running stats go stale in training (train_cc.recalibrate_bn)
POS = [0, 2, 4]                    # state columns of th1, th2, p
ANGLE = [True, True, False]


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def episode_variables(states, actions, t_min=T_MIN):
    """states (T+1, 6) = (th1, w1, th2, w2, p, pdot), actions (T,). Same layout as cc_common's."""
    states = np.asarray(states, dtype=np.float64)
    u = np.asarray(actions, dtype=np.float64).reshape(-1)
    T = len(u)
    cols = []
    for k, ang in zip(POS, ANGLE):
        q = states[:, k]
        bar = np.full(T + 1, np.nan)
        step = q[1:] - q[:-1]
        bar[1:] = (wrap(step) if ang else step) / DT
        cols += [wrap(q) if ang else q, bar]
    d = np.stack(cols, axis=1)                                   # (T+1, 6)
    ts = np.arange(t_min, T)
    if len(ts) == 0:
        return None
    inputs = np.column_stack([d[ts], u[ts], u[ts - 1], u[ts - 2]])
    targets = d[ts + 1] - d[ts]
    for c, ang in zip((0, 2, 4), ANGLE):                         # angle change = wrapped difference
        if ang:
            targets[:, c] = DT * d[ts + 1, c + 1]
    return {'t': ts, 'inputs': inputs, 'targets': targets, 'd_prev': d[ts - 1],
            'true_vel': states[ts][:, [1, 3, 5]]}


def edge_features(inputs):
    """Angles enter as (sin, cos) pairs, one node each. Returns (X, groups): groups[i] = X columns of node i."""
    cols, groups = [], []
    for i, name in enumerate(INPUT_NAMES):
        x = inputs[:, i]
        start = sum(len(g) for g in groups)
        if name in ('th1', 'th2'):
            cols += [np.sin(x), np.cos(x)]
            groups.append([start, start + 1])
        else:
            cols.append(x)
            groups.append([start])
    return np.column_stack(cols), groups


# ---------------------------------------------------------------- hand-derived expert graph
# Both joint accelerations depend on th1 and th2 (gravity; th2 also through the inertia matrix), on
# w1 and w2 (Coriolis / centrifugal), and on the torque. As for cart-pole, the true velocity at t needs
# u_{t-1} besides the frame velocities, and dth = DT * (wbar + dwbar). The distractor is a damped spring.
_ARM_PARENTS = {'th1': 'strong', 'th2': 'strong', 'w1bar': 'strong', 'w2bar': 'strong',
                'u_t': 'strong', 'u_tm1': 'strong'}
EXPERT = {
    'dth1':   {'w1bar': 'strong', 'th1': 'weak', 'th2': 'weak', 'w2bar': 'weak', 'u_t': 'weak', 'u_tm1': 'weak'},
    'dw1bar': dict(_ARM_PARENTS),
    'dth2':   {'w2bar': 'strong', 'th1': 'weak', 'th2': 'weak', 'w1bar': 'weak', 'u_t': 'weak', 'u_tm1': 'weak'},
    'dw2bar': dict(_ARM_PARENTS),
    'dp':     {'vpbar': 'strong', 'p': 'weak'},
    'dvpbar': {'p': 'strong', 'vpbar': 'weak'},
}
WHY = {
    'u_tm2': 'Markov in (frame states, u_t, u_{t-1}): older torques have no effect',
    'p, vpbar -> arm': 'the distractor is not coupled to the arm',
    'arm, actions -> dp, dvpbar': 'the distractor is not driven by the arm or the torque',
}


def adjacency():
    A = np.zeros((len(INPUT_NAMES), len(TARGET_NAMES)), dtype=int)
    for j, tgt in enumerate(TARGET_NAMES):
        for parent in EXPERT[tgt]:
            A[INPUT_NAMES.index(parent), j] = 1
    return A


def tiers():
    T = np.full((len(INPUT_NAMES), len(TARGET_NAMES)), 'absent', dtype=object)
    for j, tgt in enumerate(TARGET_NAMES):
        for parent, tier in EXPERT[tgt].items():
            T[INPUT_NAMES.index(parent), j] = tier
    return T


def cross_mask():
    """(9, 6) bool: edges between the arm (and actions) and the distractor, in either direction."""
    M = np.zeros((len(INPUT_NAMES), len(TARGET_NAMES)), dtype=bool)
    for i, a in enumerate(INPUT_NAMES):
        for j, b in enumerate(TARGET_NAMES):
            M[i, j] = (a in DISTRACTOR_INPUTS) != (b in DISTRACTOR_TARGETS)
    return M


# ---------------------------------------------------------------- dataset policies
TYPES = ('random', 'prbs', 'passive', 'swing')


def initial_state(mode, d, rng):
    r_th, r_w = {'random': (d['random_init_angle'], d['random_init_vel']),
                 'prbs': (d['pe_init_angle'], d['pe_init_vel']),
                 'passive': (d['passive_init_angle'], d['passive_init_vel']),
                 'swing': (d['swing_init_angle'], d['swing_init_vel'])}[mode]
    th1, th2 = rng.uniform(-r_th, r_th, 2)
    w1, w2 = rng.uniform(-r_w, r_w, 2)
    p = rng.uniform(-d['distractor_init_pos'], d['distractor_init_pos'])
    v = rng.uniform(-d['distractor_init_vel'], d['distractor_init_vel'])
    return [th1, w1, th2, w2, p, v]


def collect_episode(env, mode, length, d, rng, amplitude=1.0, phys=None):
    """One episode. random: uniform torque; prbs: +-amplitude with random sign flips; passive: u = 0;
    swing: energy pumping, u = umax * tanh(5 w2) + noise (power in = u * w2), which swings the arm up."""
    lo, hi = env.low, env.high
    phys = dict(PHYS if phys is None else phys)
    obs, state = env.reset_to_state(initial_state(mode, d, rng), phys)
    sign = float(rng.choice([-1.0, 1.0]))
    O, S, A = [obs], [state], []
    for step in range(length):
        if mode == 'random':
            u = rng.uniform(lo, hi)
        elif mode == 'prbs':
            if step > 0 and rng.random() < d['pe_flip_prob']:
                sign = -sign
            u = sign * amplitude
        elif mode == 'passive':
            u = 0.0
        else:
            u = float(np.clip(hi * np.tanh(5.0 * state[3]) + rng.normal(0, d['swing_action_noise_std']), lo, hi))
        obs, state, done = env.step(u)
        if done:
            break
        O.append(obs); S.append(state); A.append([u])
    if not A:
        return None
    return {'observations': np.stack(O), 'states': np.stack(S).astype(np.float32),
            'actions': np.asarray(A, np.float32), 'trajectory_type': mode, 'physics': phys}


def make_env(cfg, render=True):
    from sim.acrobot import Acrobot
    e = cfg['environment']
    return Acrobot(image_size=int(e['image_size']), frame_skip=int(e['frame_skip']),
                   action_range=tuple(e['action_range']), render=render)


def episode_plan(d):
    return {'random': (d['n_random_episodes'], d['random_ep_len']),
            'prbs': (d['n_pe_episodes'], d['pe_ep_len']),
            'passive': (d['n_passive_episodes'], d['passive_ep_len']),
            'swing': (d['n_swing_episodes'], d['swing_ep_len'])}
