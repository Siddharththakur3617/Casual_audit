"""Whole-episode batches for recurrent world models (DreamerV3).

Item i = episode i of a split, padded to the split's longest episode:
  obs           frames o_0 .. o_T                    (L, 3, h, w) uint8, zeros after the episode
  prev_actions  a_{t-1} / action_scale for frame t    (L, 1), 0 at t = 0 (u_t moves o_t -> o_{t+1})
  mask          True on real frames                   (L,)
  length        T + 1 frames
Every item starts at the episode's first frame, so training and extraction (probing/extract_dreamer.py)
both start the model from its zero state at t = 0. All frames are kept in RAM.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datagen.windows import load_meta


def prev_actions(actions, action_scale):
    """actions (T,) raw -> (T+1, 1) scaled a_{t-1} per frame t, zero at t = 0."""
    a = np.zeros((len(actions) + 1, 1), np.float32)
    a[1:, 0] = np.asarray(actions, np.float32).reshape(-1) / action_scale
    return a


class EpisodeDataset(Dataset):
    def __init__(self, path, action_scale):
        self.obs, self.pa = [], []
        with h5py.File(path, 'r') as f:
            for key in sorted(f['episodes'].keys(), key=int):
                ep = f['episodes'][key]
                self.obs.append(torch.from_numpy(ep['observations'][:]).permute(0, 3, 1, 2).contiguous())
                self.pa.append(torch.from_numpy(prev_actions(ep['actions'][:], action_scale)))
        self.L = max(len(o) for o in self.obs)

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, i):
        o, a, L = self.obs[i], self.pa[i], self.L
        n = len(o)
        obs = torch.zeros((L,) + tuple(o.shape[1:]), dtype=torch.uint8)
        obs[:n] = o
        pa = torch.zeros(L, 1)
        pa[:n] = a
        mask = torch.zeros(L, dtype=torch.bool)
        mask[:n] = True
        return {'obs': obs, 'prev_actions': pa, 'mask': mask, 'length': n}


def make_loaders(data_dir, batch_size, num_workers=0):
    scale = float(load_meta(data_dir)['action_scale'])
    out = {}
    for split in ('train', 'val'):
        ds = EpisodeDataset(Path(data_dir) / f'{split}.hdf5', scale)
        n_frames = sum(len(o) for o in ds.obs)
        print(f'[dataset:{split}] {len(ds)} episodes, {n_frames} frames, padded to {ds.L}')
        out[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'),
                                drop_last=(split == 'train'), num_workers=num_workers,
                                pin_memory=torch.cuda.is_available())
    return out, scale
