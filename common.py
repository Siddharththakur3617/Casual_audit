"""Generic data helpers, the same for every environment in envs/ and every model.

stack_episodes      per-episode (states, actions) -> stacked edge inputs / targets of the active environment
episode_physics     an episode's physics from its HDF5 attrs
load_split_episodes read one split of a dataset (and activate the dataset's environment)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def stack_episodes(episodes, t_min=None, env=None):
    """episodes: iterable of (states (T+1,k), actions (T,)). Concatenate variables.
    env: environment spec (default: the active one, envs.current()); t_min: default env.T_MIN."""
    if env is None:
        import envs
        env = envs.current()
    t_min = env.T_MIN if t_min is None else t_min
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
