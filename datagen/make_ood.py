"""Held-out-physics evaluation set: same policies and renderer as the training data, but physics drawn
from ranges OUTSIDE the training ranges (configs/envs/ood/<env>_ood.yaml). Only evaluation data:
val.hdf5 holds the new episodes; train.hdf5 is hard-linked (or copied) from the source dataset so that
BatchNorm recalibration (Acrobot) still uses the TRAINING windows, never the held-out ones.

  python -m datagen.make_ood --data datasets/cc_phys        # -> datasets/cc_phys_ood
  python -m datagen.make_ood --data datasets/ac_phys        # -> datasets/ac_phys_ood

The default ranges change masses and gravity only; lengths stay inside the training range so the
images look like training images while the dynamics are new. Use it with
evaluate/compare_training.py --data datasets/<data>_ood (open-loop rollout errors on held-out physics).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

import envs
from datagen import generate as G
from datagen.subset import link_or_copy
from paths import ENV_CONFIGS


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='training dataset whose environment and train split are reused')
    p.add_argument('--ood-config', default=None, help='default: configs/envs/ood/<env>_ood.yaml')
    p.add_argument('--out', default=None)
    a = p.parse_args()
    src = Path(a.data)
    env_name = envs.env_name_of(src)
    G.ENV = env_name
    spec = G._spec()
    ood = yaml.safe_load(Path(a.ood_config or ENV_CONFIGS / 'ood' / f'{env_name}_ood.yaml').read_text())
    cfg = G.load_cfg()
    train_ranges = dict(cfg['physics_ranges'])
    cfg['physics_ranges'] = {**{k: v for k, v in train_ranges.items() if k != 'seed'},
                             **ood['physics_ranges'], 'seed': int(ood.get('physics_seed', 9090))}
    d = cfg['data']
    d['seed'] = int(ood.get('data_seed', 9191))          # new initial states and actions too
    frac = float(ood.get('episode_fraction', 0.2))
    for k, v in list(d.items()):
        if k.startswith('n_') and k.endswith('episodes') and isinstance(v, int) and v > 0:
            d[k] = max(4, int(round(v * frac)))
    out = Path(a.out) if a.out else src.parent / f'{src.name}_ood'
    out.mkdir(parents=True, exist_ok=True)
    episodes = G.collect(cfg, render=True, vary_physics=True)
    G.write_split(out / 'val.hdf5', episodes)
    link_or_copy(src / 'train.hdf5', out / 'train.hdf5')
    meta = json.loads((src / 'metadata.json').read_text())
    P = np.array([[ep['physics'][k] for k in spec.PHYS_KEYS] for ep in episodes])
    meta['ood'] = {'source': str(src), 'train_ranges': train_ranges, 'ood_ranges': cfg['physics_ranges'],
                   'episodes': len(episodes), 'transitions': int(sum(len(ep['actions']) for ep in episodes)),
                   'physics_drawn': {k: [float(P[:, i].min()), float(P[:, i].max())]
                                     for i, k in enumerate(spec.PHYS_KEYS)},
                   'note': 'val.hdf5 = held-out physics; train.hdf5 = the source training split (for BN recalibration only)'}
    (out / 'metadata.json').write_text(json.dumps(meta, indent=2))
    print(f'[ood] {len(episodes)} episodes with physics {meta["ood"]["physics_drawn"]} -> {out}')


if __name__ == '__main__':
    main()
