"""Low-data copy of a dataset: keep a fraction of the TRAINING episodes (stratified by episode type),
val and test unchanged. Used to test whether the graph prior helps when data is scarce.

  python -m datagen.subset --data datasets/cc_phys --fraction 0.25      # -> datasets/cc_phys_frac25

val.hdf5 / test.hdf5 are hard-linked when possible (copied otherwise). The subset has no expert graph
or null of its own: score its checkpoints against the full dataset
(run_all.py --data datasets/cc_phys --run-dir runs/<subset run> ...), which uses the same episodes' physics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path

import h5py
import numpy as np


def link_or_copy(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--fraction', type=float, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    a = p.parse_args()
    assert 0 < a.fraction < 1, '--fraction must be in (0, 1)'
    src = Path(a.data)
    out = Path(a.out) if a.out else src.parent / f'{src.name}_frac{round(100 * a.fraction):02d}'
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    with h5py.File(src / 'train.hdf5', 'r') as f:
        keys = sorted(f['episodes'].keys(), key=int)
        types = {}
        for k in keys:
            t = f['episodes'][k].attrs.get('trajectory_type', 'unknown')
            types.setdefault(t.decode() if isinstance(t, bytes) else str(t), []).append(k)
        keep = []
        for t, ks in sorted(types.items()):
            n = max(1, math.ceil(a.fraction * len(ks)))
            keep += [ks[i] for i in rng.choice(len(ks), size=n, replace=False)]
        keep = sorted(keep, key=int)
        n_tr = 0
        with h5py.File(out / 'train.hdf5', 'w') as g:
            dst = g.create_group('episodes')
            for i, k in enumerate(keep):
                f.copy(f['episodes'][k], dst, name=str(i))
                n_tr += len(f['episodes'][k]['actions'])
            g.attrs['n_episodes'] = len(keep)
            g.attrs['n_transitions'] = n_tr
    for split in ('val', 'test'):
        if (src / f'{split}.hdf5').exists():
            link_or_copy(src / f'{split}.hdf5', out / f'{split}.hdf5')
    meta = json.loads((src / 'metadata.json').read_text())
    meta['subset_of'] = {'data': str(src), 'fraction': a.fraction, 'seed': a.seed,
                         'train_episodes': len(keep), 'train_episodes_full': len(keys), 'train_transitions': n_tr}
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print(f'[subset] {len(keep)}/{len(keys)} training episodes ({n_tr} transitions) -> {out}')


if __name__ == '__main__':
    main()
