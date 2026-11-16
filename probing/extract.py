"""Step 4a: run any model's checkpoint over a data split and save what the later steps need.

Every extractor returns the same arrays, one row per usable transition t >= T_MIN (the rows of
env.episode_variables, so the nulls and data baselines of a dataset apply to every model):
  z          the model's latent at t                     (encoder side)
  z_next     the latent at t+1 computed WITH frame t+1    (encoder control)
  zhat_next  the model's prediction of the latent at t+1, WITHOUT frame t+1
  inputs, targets, <env.EXTRA_KEYS>, episode, t, physics, trajectory_type, dzhat_du, action_scale, epoch
Model-specific extras: hist_sens (JEPA), z_stoch / z_next_stoch / zhat_stoch (DreamerV3).

  jepa       probing/extract_jepa.py
  dreamerv3  probing/extract_dreamer.py

Run: python -m probing.extract --ckpt runs/jepa_cc_phys/checkpoints/checkpoint_epoch0200.pt --data datasets/cc_phys --out x.npz
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def model_type(ckpt_path):
    """'jepa' or 'dreamerv3' (JEPA checkpoints made before the key existed have none)."""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    return ck.get('model_type', 'jepa')


def extract(ckpt_path, data_dir, split='train', device='cuda', out=None):
    kind = model_type(ckpt_path)
    if kind == 'jepa':
        from probing import extract_jepa as m
    elif kind == 'dreamerv3':
        from probing import extract_dreamer as m
    else:
        raise ValueError(f'unknown model_type {kind!r} in {ckpt_path}')
    return m.extract(ckpt_path, data_dir, split, device=device, out=out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--split', default='train')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    r = extract(a.ckpt, a.data, a.split, out=a.out)
    print(f'{len(r["t"])} transitions from {len(np.unique(r["episode"]))} episodes -> {a.out}')


if __name__ == '__main__':
    main()
