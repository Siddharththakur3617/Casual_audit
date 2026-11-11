"""Environment registry for the causal-structure study.

Each environment is a module in this package with the same attributes (see envs/cartpole.py):
  NAME, H, FRAME_SKIP, DT, T_MIN, PHYS, PHYS_KEYS, INPUT_NAMES, TARGET_NAMES, STATE_NAMES,
  ACTION_COLS (input columns that are actions), DECOY_SOURCE (input nodes copied as shuffled decoys),
  ACTION_EPS, EXTRA_KEYS (per-transition extras that probing/extract.py keeps),
  episode_variables(states, actions, t_min), edge_features(inputs) -> (X, groups or None).

The active environment is chosen from a dataset's metadata.json ("env"; datasets without it are
cart-pole). Shared code calls current() at run time, so one process works on one environment.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

DEFAULT = 'cartpole'
NAMES = ('cartpole', 'acrobot')
_current = None


def get_env(name):
    if name not in NAMES:
        raise ValueError(f'unknown env {name!r}; known: {NAMES}')
    return importlib.import_module(f'envs.{name}')


def env_name_of(data_dir):
    p = Path(data_dir) / 'metadata.json'
    if p.exists():
        return json.loads(p.read_text()).get('env', DEFAULT)
    return DEFAULT


def use(name):
    """Make `name` the active environment and return its module."""
    global _current
    _current = get_env(name)
    return _current


def use_data(data_dir):
    """Activate the environment a dataset was generated with."""
    return use(env_name_of(data_dir)) if data_dir is not None else current()


def current():
    return _current if _current is not None else use(DEFAULT)
