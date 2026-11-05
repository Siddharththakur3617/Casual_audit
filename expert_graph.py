"""Expert causal graph for cart-pole at frame_skip = 2 (one transition = 0.04 s).

Derived by hand from the simulator equations (explicit Euler, force held over
both sub-steps, no friction). Neither acceleration depends on x or x_dot.

With h = 0.02 and vbar_t = (x_t - x_{t-1}) / 0.04:
  vbar_t     = x_dot_{t-1} + (h/2) a0(t-1)
  x_dot_t    = vbar_t + h (a0(t-1)/2 + a1(t-1))      <- needs u_{t-1}
  d vbar     = h (a0(t-1)/2 + a1(t-1)) + (h/2) a0(t)  <- u_{t-1}, u_t, theta, wbar
  dx         = 0.04 * (vbar_t + d vbar)
  dtheta     = 0.04 * (wbar_t + d wbar)
where a0, a1 are the accelerations at the two sub-steps of a transition.

Tiers here are the hand derivation. checks/verify_equations.py measures the
actual effect sizes and writes the final tiers to results/expert_graph_final.json.

Run: python expert_graph.py   -> results/expert_graph.json
"""
from __future__ import annotations

import json

import numpy as np

from cc_common import INPUT_NAMES, RESULTS, TARGET_NAMES

# target -> {parent: tier}
EXPERT = {
    'dx':     {'vbar': 'strong', 'theta': 'weak', 'wbar': 'weak', 'u_t': 'weak', 'u_tm1': 'weak'},
    'dtheta': {'wbar': 'strong', 'theta': 'weak', 'u_t': 'weak', 'u_tm1': 'weak'},
    'dvbar':  {'theta': 'strong', 'wbar': 'strong', 'u_t': 'strong', 'u_tm1': 'strong'},
    'dwbar':  {'theta': 'strong', 'wbar': 'strong', 'u_t': 'strong', 'u_tm1': 'strong'},
}

WHY = {
    'x': 'translation invariance: no acceleration depends on x',
    'u_tm2': 'Markov in (x, vbar, theta, wbar, u_t, u_{t-1}): older forces have no effect',
    'vbar->dvbar/dtheta/dwbar': 'no friction, so x_dot never enters an acceleration',
}


def adjacency():
    """(7, 4) 0/1 matrix, rows = inputs, cols = targets."""
    A = np.zeros((len(INPUT_NAMES), len(TARGET_NAMES)), dtype=int)
    for j, tgt in enumerate(TARGET_NAMES):
        for parent in EXPERT[tgt]:
            A[INPUT_NAMES.index(parent), j] = 1
    return A


def tiers():
    """(7, 4) array of 'strong' / 'weak' / 'absent'."""
    T = np.full((len(INPUT_NAMES), len(TARGET_NAMES)), 'absent', dtype=object)
    for j, tgt in enumerate(TARGET_NAMES):
        for parent, tier in EXPERT[tgt].items():
            T[INPUT_NAMES.index(parent), j] = tier
    return T


def main():
    RESULTS.mkdir(exist_ok=True)
    A = adjacency()
    out = {
        'inputs': INPUT_NAMES,
        'targets': TARGET_NAMES,
        'adjacency': A.tolist(),
        'tiers': tiers().tolist(),
        'n_true': int(A.sum()),
        'n_absent': int(A.size - A.sum()),
        'why_absent': WHY,
    }
    (RESULTS / 'expert_graph.json').write_text(json.dumps(out, indent=2))
    print(f'true edges: {out["n_true"]}, absent: {out["n_absent"]}')
    for j, tgt in enumerate(TARGET_NAMES):
        print(f'  {tgt:7s} <- ' + ', '.join(f'{p} ({t})' for p, t in EXPERT[tgt].items()))


if __name__ == '__main__':
    main()
