"""Training windows from the per-episode HDF5 files.

A window starting at transition s of an episode contains
  obs_seq      frames s .. s+horizon             (horizon+1, 3, h, w) uint8
  prev_obs     frame s-1 (frame s at the episode start)
  actions      u_s .. u_{s+horizon-1} / action_scale       (horizon, 1)
  prev_actions the ctx-1 forces before u_s (zeros before the episode start)
  states       s_s .. s_{s+horizon}                         (horizon+1, 4)
All frames are kept in RAM (about 1.8 GB for the full dataset).
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


def load_meta(data_dir):
    p = Path(data_dir) / 'metadata.json'
    return json.loads(p.read_text()) if p.exists() else {'action_scale': 10.0}


class WindowDataset(Dataset):
    def __init__(self, path, horizon, action_context_dim=2, action_scale=10.0):
        self.h, self.ctx = int(horizon), int(action_context_dim)
        frames, acts, states, starts = [], [], [], []
        f0 = a0 = 0
        with h5py.File(path, 'r') as f:
            for key in sorted(f['episodes'].keys(), key=int):
                ep = f['episodes'][key]
                o, a, s = ep['observations'][:], ep['actions'][:].reshape(-1), ep['states'][:]
                T = len(a)
                # window start s needs frames s..s+h, i.e. s+h <= T
                for st in range(0, T - self.h + 1):
                    starts.append((f0 + st, a0 + st, st))
                frames.append(o); acts.append(a); states.append(s)
                f0 += T + 1; a0 += T
        self.frames = torch.from_numpy(np.concatenate(frames)).permute(0, 3, 1, 2).contiguous()
        self.actions = torch.from_numpy(np.concatenate(acts).astype(np.float32) / action_scale)
        self.states = torch.from_numpy(np.concatenate(states).astype(np.float32))
        self.starts = np.asarray(starts, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        fi, ai, st = self.starts[i]
        h, c = self.h, self.ctx
        prev = self.frames[fi - 1] if st > 0 else self.frames[fi]
        n_back = min(c - 1, st)
        pa = torch.zeros(c - 1, 1)
        if n_back > 0:
            pa[c - 1 - n_back:, 0] = self.actions[ai - n_back:ai]
        return {
            'obs_seq': self.frames[fi:fi + h + 1],
            'prev_obs': prev,
            'actions': self.actions[ai:ai + h, None],
            'prev_actions': pa,
            'states': self.states[fi:fi + h + 1],
        }


def make_loaders(data_dir, horizon, batch_size, action_context_dim=2, num_workers=0):
    scale = float(load_meta(data_dir)['action_scale'])
    out = {}
    for split in ('train', 'val'):
        p = Path(data_dir) / f'{split}.hdf5'
        ds = WindowDataset(p, horizon, action_context_dim, scale)
        print(f'[dataset:{split}] {len(ds)} windows, {len(ds.frames)} frames '
              f'({ds.frames.numel() / 1e6:.0f} MB)')
        out[split] = DataLoader(ds, batch_size=batch_size, shuffle=(split == 'train'),
                                drop_last=(split == 'train'), num_workers=num_workers,
                                pin_memory=torch.cuda.is_available())
    return out, scale
