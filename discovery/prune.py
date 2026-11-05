"""Step 6: pruning rule, identical for every graph and blind to the expert graph.

An edge is kept if, in at least STABILITY_MIN of the random half-splits, its
LOCO weight beats the strongest decoy for its target by edge_scores.MARGIN (stability selection with a
decoy threshold; Meinshausen & Buhlmann 2010, Barber & Candes 2015).
"""
from __future__ import annotations

import numpy as np

STABILITY_MIN = 0.7


def prune(scores, stability_min=STABILITY_MIN):
    return (np.asarray(scores['stability']) >= stability_min).astype(int)
