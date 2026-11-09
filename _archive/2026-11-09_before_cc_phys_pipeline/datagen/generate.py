"""Step 2: generate the frame_skip = 2 cart-pole dataset.

Episode types (counts and lengths in configs/causal_cartpole.yaml):
  random       uniform force in [-10, 10] each step (u_t, u_{t-1} uncorrelated)
  prbs         +-amplitude, sign flips with probability pe_flip_prob each step
  passive      u = 0, the pole falls on its own
  angle_sweep  broad initial angles, LQR + noise (LQR gain for the 0.04 s step)
An episode stops before the first step that leaves the view (|x| > 2.2) or
passes |theta| > 1.2, so every stored transition is complete and visible.

Output (per split): episodes/<i>/{observations (T+1,h,w,3) uint8, states (T+1,4),
actions (T,1)}, attrs trajectory_type; plus metadata.json and summary.json.

Physics: fixed at cc_common.PHYS, or with --vary-physics drawn per episode from physics_ranges
in the config (cart mass M, pole mass m, pole half-length l, gravity g; constant within an
episode, LQR gain recomputed for it, pole rendered at its true length). Every episode stores its
physics as attrs phys_M, phys_m, phys_l, phys_g.

Run:
  python datagen/generate.py                   # full dataset  -> datasets/cc
  python datagen/generate.py --small           # smoke-test set -> datasets/cc_small
  python datagen/generate.py --vary-physics    # varied physics -> datasets/cc_phys
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import scipy.linalg
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cc_common import DT, HERE, PHYS  # noqa: E402
from sim.cartpole import CartPole  # noqa: E402

CFG_PATH = HERE / 'configs' / 'causal_cartpole.yaml'
TYPES = ('random', 'prbs', 'passive', 'angle_sweep')


def load_cfg():
    return yaml.safe_load(CFG_PATH.read_text())


def lqr_gain(dt=DT, M=PHYS['M'], m=PHYS['m'], l=PHYS['l'], g=PHYS['g']):
    """u = K @ s for the upright equilibrium, discretised (zero-order hold) at dt."""
    d = 4 * M + m
    A = np.array([[0, 1, 0, 0],
                  [0, 0, -3 * m * g / d, 0],
                  [0, 0, 0, 1],
                  [0, 0, 3 * (M + m) * g / (l * d), 0]], float)
    B = np.array([[0], [4 / d], [0], [-3 / (l * d)]], float)
    E = scipy.linalg.expm(np.block([[A, B], [np.zeros((1, 5))]]) * dt)
    Ad, Bd = E[:4, :4], E[:4, 4:]
    Q, R = np.diag([1.0, 1.0, 10.0, 1.0]), np.array([[0.01]])
    P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
    return -(np.linalg.inv(R + Bd.T @ P @ Bd) @ Bd.T @ P @ Ad)[0]


def collect_episode(env, mode, length, d, rng, K, amplitude=3.0, phys=None):
    lo, hi = env.low, env.high
    phys = dict(PHYS if phys is None else phys)
    if mode == 'angle_sweep':
        th = float(rng.choice([-1.0, 1.0])) * rng.uniform(d['angle_min'], d['angle_max'])
        v = d.get('angle_velocity_range', 0.2)
        s0 = [rng.uniform(-0.1, 0.1), rng.uniform(-v, v), th, rng.uniform(-v, v)]
        obs, state = env.reset_to_state(s0, phys)
    else:
        r = d[{'random': 'random_init_range', 'prbs': 'pe_init_range',
               'passive': 'passive_init_range'}[mode]]
        obs, state = env.reset_to_state(rng.uniform(-r, r, size=4), phys)
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
            u = np.clip(K @ state + rng.normal(0, d['angle_action_noise_std']), lo, hi)
        obs, state, done = env.step(u)
        if done or abs(state[0]) > d['max_abs_x'] or abs(state[2]) > d['max_abs_theta']:
            break
        O.append(obs); S.append(state); A.append([u])
    if not A:
        return None
    return {'observations': np.stack(O), 'states': np.stack(S).astype(np.float32),
            'actions': np.asarray(A, np.float32), 'trajectory_type': mode, 'physics': phys}


PHYS_KEYS = ('M', 'm', 'l', 'g')
# summary.json names: 'M' and 'm' alone clash in case-insensitive readers such as PowerShell
SUMMARY_NAME = {'M': 'cart_mass_M', 'm': 'pole_mass_m', 'l': 'pole_half_length_l', 'g': 'gravity_g'}


def draw_physics(ranges, prng):
    """One episode's physics: each parameter uniform in its configured range."""
    return {k: float(prng.uniform(*ranges[k])) for k in PHYS_KEYS}


def collect(cfg, render=True, small=False, verbose=True, vary_physics=False):
    e, d = cfg['environment'], cfg['data']
    rng = np.random.default_rng(int(d.get('seed', 42)))
    env = CartPole(image_size=int(e['image_size']), frame_skip=int(e['frame_skip']),
                   action_range=tuple(e['action_range']), theta_threshold=d['max_abs_theta'],
                   render=render)
    K = lqr_gain()
    if vary_physics:
        ranges = cfg['physics_ranges']
        prng = np.random.default_rng(int(ranges.get('seed', 4242)))
    plan = {'random': (d['n_random_episodes'], d['random_ep_len']),
            'prbs': (d['n_pe_episodes'], d['pe_ep_len']),
            'passive': (d['n_passive_episodes'], d['passive_ep_len']),
            'angle_sweep': (d['n_angle_episodes'], d['angle_ep_len'])}
    episodes = []
    for mode in TYPES:
        n, length = plan[mode]
        n = max(4, n // 40) if small else n
        t0 = time.time()
        for k in range(n):
            amp = d['pe_action_amplitudes'][k % len(d['pe_action_amplitudes'])]
            if vary_physics:
                phys = draw_physics(ranges, prng)
                ep = collect_episode(env, mode, length, d, rng, lqr_gain(**phys), amplitude=amp, phys=phys)
            else:
                ep = collect_episode(env, mode, length, d, rng, K, amplitude=amp)
            if ep is not None:
                episodes.append(ep)
        if verbose:
            got = [ep for ep in episodes if ep['trajectory_type'] == mode]
            print(f'[data] {mode:12s} {len(got)} episodes, {sum(len(x["actions"]) for x in got)} '
                  f'transitions ({time.time() - t0:.0f}s)', flush=True)
    return episodes


def split_episodes(episodes, seed=0, val=0.1, test=0.1):
    """Random split by whole episodes, within each episode type."""
    rng = np.random.default_rng(seed)
    out = {'train': [], 'val': [], 'test': []}
    for mode in TYPES:
        idx = [i for i, ep in enumerate(episodes) if ep['trajectory_type'] == mode]
        rng.shuffle(idx)
        nv, nt = int(round(len(idx) * val)), int(round(len(idx) * test))
        out['val'] += idx[:nv]
        out['test'] += idx[nv:nv + nt]
        out['train'] += idx[nv + nt:]
    return {k: sorted(v) for k, v in out.items()}


def write_split(path, episodes):
    with h5py.File(path, 'w') as f:
        g = f.create_group('episodes')
        n = 0
        for i, ep in enumerate(episodes):
            e = g.create_group(str(i))
            e.create_dataset('observations', data=ep['observations'], compression='gzip',
                             compression_opts=4)
            e.create_dataset('states', data=ep['states'])
            e.create_dataset('actions', data=ep['actions'])
            e.attrs['trajectory_type'] = ep['trajectory_type']
            for k, v in ep.get('physics', PHYS).items():     # phys_M, phys_m, phys_l, phys_g
                e.attrs[f'phys_{k}'] = float(v)
            n += len(ep['actions'])
        f.attrs['n_episodes'] = len(episodes)
        f.attrs['n_transitions'] = n


def simulate_states(small=False):
    """States only (no rendering), same policies and seed: used by the checks."""
    eps = collect(load_cfg(), render=False, small=small, verbose=False)
    return [(ep['states'], ep['actions'].reshape(-1), ep['trajectory_type']) for ep in eps]


def action_stats(eps):
    """Per episode type: corr(u_t, u_{t-1}) and max |corr(u_t, state_t)|."""
    out = {}
    for typ in TYPES:
        sel = [(s, a) for s, a, t in eps if t == typ and len(a) >= 2]
        if not sel:
            continue
        u0 = np.concatenate([a[1:] for _, a in sel]); u1 = np.concatenate([a[:-1] for _, a in sel])
        S = np.concatenate([s[1:-1] for s, _ in sel])
        ok = u0.std() > 0
        out[typ] = {
            'episodes': len(sel),
            'transitions': int(sum(len(a) for _, a in sel)),
            'corr_u_t_u_tm1': float(np.corrcoef(u0, u1)[0, 1]) if ok else None,
            'max_abs_corr_u_state': (float(max(abs(np.corrcoef(u0, S[:, k])[0, 1]) for k in range(4)))
                                     if ok else None),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--small', action='store_true')
    p.add_argument('--vary-physics', action='store_true',
                   help='per-episode M, m, l, g from physics_ranges in the config (-> datasets/cc_phys)')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    name = ('cc_phys' if a.vary_physics else 'cc') + ('_small' if a.small else '')
    out = Path(a.out) if a.out else HERE / 'datasets' / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_cfg()
    episodes = collect(cfg, render=True, small=a.small, vary_physics=a.vary_physics)
    split = split_episodes(episodes)
    for name, idx in split.items():
        write_split(out / f'{name}.hdf5', [episodes[i] for i in idx])
        print(f'[data] {name}: {len(idx)} episodes, '
              f'{sum(len(episodes[i]["actions"]) for i in idx)} transitions')
    (out / 'metadata.json').write_text(json.dumps(
        {'action_scale': float(max(abs(v) for v in cfg['environment']['action_range'])),
         'frame_skip': int(cfg['environment']['frame_skip']), 'dt_transition': DT}, indent=2))
    eps = [(ep['states'], ep['actions'].reshape(-1), ep['trajectory_type']) for ep in episodes]
    lens = np.array([len(ep['actions']) for ep in episodes])
    summary = {'episodes': len(episodes), 'transitions': int(lens.sum()),
               'usable_transitions_t>=5': int(np.clip(lens - 5, 0, None).sum()),
               'episodes_shorter_than_9_frames': int((lens + 1 < 9).sum()),
               'by_type': action_stats(eps)}
    P = np.array([[ep['physics'][k] for k in PHYS_KEYS] for ep in episodes])
    summary['physics'] = {'varied': bool(a.vary_physics),
                          'ranges': ({SUMMARY_NAME.get(k, k): v for k, v in cfg['physics_ranges'].items()}
                                     if a.vary_physics else None),
                          **{SUMMARY_NAME[k]: {'min': float(P[:, i].min()), 'mean': float(P[:, i].mean()),
                                               'max': float(P[:, i].max())} for i, k in enumerate(PHYS_KEYS)}}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
