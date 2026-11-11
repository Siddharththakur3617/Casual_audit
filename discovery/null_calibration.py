"""Null scores for the null_cal pruning rules (discovery/prune.py).

For one dataset, score every edge (28 for cart-pole, 54 for acrobot) K times with the targets row-shuffled (every edge is then truly
absent) and keep each repeat's per-split Dvs / Dvs_se. prune.null_statistic turns these into one
statistic per edge and repeat; the rule compares the real statistic with them.

The rows (inputs, episodes) are the dataset's train split, which is exactly what the data graphs and
the model graphs are scored on (probing/extract.py extracts the same rows). Only the targets are
shuffled, so a null depends on the dataset, not on any model. Repeats use different shuffles and
different split seeds from the real graphs (real graphs: seed 0; the compare_pruning 'null' graph:
shuffle seed 12345), so that graph is an untouched check of the calibration.

Each repeat is cached, so the run resumes after a crash; the file name carries n_splits, because a
null is only valid for real scores computed with the same number of splits.

  python -m discovery.null_calibration --data datasets/cc_phys --reps 20        # ~4-5 min per repeat on the GPU
Output: results/null_calibration_<data>/rep###_s<splits>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cc_common import RESULTS, load_split_episodes, stack_episodes
from discovery import edge_scores
from discovery.prune import DELTA, null_statistic

SEED_PERM = 9000        # shuffle of repeat k: default_rng(SEED_PERM + k)
SEED_SCORE = 7000       # split / decoy / network seed of repeat k
REPS = 20               # smallest K that can reach p <= 0.05 with margin: 1 / (K + 1) = 0.048


def cache_dir(data_dir, split='train'):
    """A null is only valid for the row count it was made on, so other splits get their own folder."""
    tag = '' if split == 'train' else f'_{split}'
    return RESULTS / f'null_calibration_{Path(data_dir).name}{tag}'


def _path(out_dir, k, n_splits):
    return Path(out_dir) / f'rep{k:03d}_s{n_splits}.json'


def calibrate(data_dir, reps=REPS, n_splits=50, out_dir=None, device='cuda', verbose=True, cond=None,
              split='train'):
    """Run (or resume) the K null repeats. Returns the number of repeats on disk.
    cond: optional (n, k) conditioning columns (e.g. episode physics) given to every fit, as in
    edge_scores.score_graph; use a separate out_dir, a null is only valid for the same conditioning."""
    out_dir = Path(out_dir) if out_dir else cache_dir(data_dir, split)
    out_dir.mkdir(parents=True, exist_ok=True)
    V = stack_episodes(load_split_episodes(data_dir, split))
    X, Y, ep = V['inputs'], V['targets'], V['episode']
    n_in = X.shape[1]
    for k in range(reps):
        path = _path(out_dir, k, n_splits)
        if path.exists() and path.stat().st_size > 0:
            continue
        t0 = time.time()
        perm = np.random.default_rng(SEED_PERM + k).permutation(len(Y))
        s = edge_scores.score_graph(X, Y[perm], ep, n_splits=n_splits, seed=SEED_SCORE + k, device=device,
                                    cond=cond)
        P = s['per_split']
        path.write_text(json.dumps({'Dvs': P['Dvs'][:, :n_in].tolist(), 'Dvs_se': P['Dvs_se'][:, :n_in].tolist(),
                                    'heldout_R2': np.asarray(s['heldout_R2']).tolist(), 'n_splits': n_splits}))
        if verbose:
            print(f'  null repeat {k + 1}/{reps}: held-out R2 {np.round(s["heldout_R2"], 3)} '
                  f'({time.time() - t0:.0f}s)', flush=True)
    return reps


def load(data_dir, n_splits=50, reps=None, out_dir=None, delta=DELTA, split='train'):
    """(K, n_in, n_tg) null statistics, and the repeats' mean held-out R2 per target.
    delta: the minimum effect inside the statistic (scalar, or (n_in, 1) per input for null_cal_vif_fwer)."""
    out_dir = Path(out_dir) if out_dir else cache_dir(data_dir, split)
    files = sorted(out_dir.glob(f'rep*_s{n_splits}.json'))
    if reps:
        files = files[:reps]
    if not files:
        raise FileNotFoundError(f'no null repeats with {n_splits} splits in {out_dir}: run '
                                f'`python -m discovery.null_calibration --data {data_dir} --splits {n_splits}'
                                f'{"" if split == "train" else f" --split {split}"}`')
    stats, r2 = [], []
    for f in files:
        d = json.loads(f.read_text())
        stats.append(null_statistic(np.asarray(d['Dvs']), np.asarray(d['Dvs_se']), delta=delta))
        r2.append(d['heldout_R2'])
    return np.asarray(stats), np.mean(r2, 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--reps', type=int, default=REPS)
    p.add_argument('--splits', type=int, default=50)
    p.add_argument('--out', default=None)
    p.add_argument('--split', default='train', help='dataset split whose rows are shuffled (default train)')
    a = p.parse_args()
    print(f'[null calibration] {a.data} ({a.split}): {a.reps} repeats x {a.splits} splits -> '
          f'{a.out or cache_dir(a.data, a.split)}', flush=True)
    calibrate(a.data, a.reps, a.splits, a.out, split=a.split)
    N, r2 = load(a.data, a.splits, out_dir=a.out, split=a.split)
    print(f'done: {len(N)} repeats; mean held-out R2 of the shuffled fits {np.round(r2, 3)}; '
          f'largest statistic per repeat (all edges): median {np.median(N.max(axis=(1, 2))):.2f}')


if __name__ == '__main__':
    main()
