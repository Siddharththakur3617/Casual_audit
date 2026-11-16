"""SIGReg (LeJEPA, Balestriero & LeCun 2025): push embeddings towards an isotropic
standard Gaussian, which prevents collapse without a target encoder.

The embeddings are projected onto random unit directions; for each 1-D projection
the Epps-Pulley statistic compares its empirical characteristic function with the
standard normal one, exp(-t^2/2), weighted by exp(-t^2/2) over t in [0, t_max].
"""
from __future__ import annotations

import torch


def sigreg_loss(z, num_slices=128, num_points=17, t_max=3.0):
    """z: (n, d). Returns the mean Epps-Pulley statistic over random 1-D slices."""
    d = z.shape[1]
    dirs = torch.randn(d, num_slices, device=z.device, dtype=z.dtype)
    dirs = dirs / dirs.norm(dim=0, keepdim=True)
    proj = z @ dirs                                              # (n, slices)
    t = torch.linspace(0.0, t_max, num_points, device=z.device, dtype=z.dtype)
    xt = proj[..., None] * t                                     # (n, slices, points)
    target = torch.exp(-0.5 * t ** 2)
    err = (torch.cos(xt).mean(0) - target) ** 2 + torch.sin(xt).mean(0) ** 2
    stat = torch.trapezoid(err * target, t, dim=-1)              # (slices,)
    return stat.mean()
