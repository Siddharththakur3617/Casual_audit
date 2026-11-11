"""Cart-pole spec. The functions are the original ones in cc_common.py, so results are unchanged."""
from __future__ import annotations

from cc_common import (DT, FRAME_SKIP, H, INPUT_NAMES, PHYS, PHYS_KEYS, STATE_NAMES,  # noqa: F401
                       T_MIN, TARGET_NAMES, episode_variables, macro_step)

NAME = 'cartpole'
ACTION_COLS = [4, 5, 6]            # u_t, u_{t-1}, u_{t-2}
DECOY_SOURCE = [4, 2]              # shuffled copies of u_t and theta
DECOY_NAMES = ['decoy_u_t', 'decoy_theta']
ACTION_EPS = 0.1                   # newtons
EXTRA_KEYS = ('xdot', 'thetadot')
CONFIG = 'causal_cartpole.yaml'
RECALIBRATE_BN = False             # stats are fine (recalibration changes losses < 0.001); keeps old results


def edge_features(inputs):
    """Edge-scoring design matrix: the inputs themselves, one column per node."""
    return inputs, None
