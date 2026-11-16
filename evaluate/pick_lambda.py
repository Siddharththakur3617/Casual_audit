"""Pick a loss weight by constraint (report: "minimise the structure penalty subject to prediction error
staying at or below the dense model's"): the LARGEST candidate weight whose val rollout loss at --epoch
stays within --tol of the reference run's at the same epoch. Short runs, same epochs and schedule.

  python -m evaluate.pick_lambda --reference runs/_sweep/standard_e10 \
      --candidate 0.1=runs/_sweep/peroutput_ctx0.1 --candidate 1.0=runs/_sweep/peroutput_ctx1.0 \
      --out runs/_sweep/pick_lambda_ctx.json

Writes {"chosen": value, ...}; if no candidate meets the constraint, the smallest weight is chosen and
"constraint_met" is false.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def val_rollout_at(run, epoch):
    h = json.loads((Path(run) / 'history.json').read_text())
    rows = [r for r in h if r['epoch'] == epoch] or [h[-1]]
    return float(rows[0]['val']['rollout']), int(rows[0]['epoch'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', required=True)
    p.add_argument('--candidate', action='append', required=True, help='value=run_dir')
    p.add_argument('--epoch', type=int, default=10)
    p.add_argument('--tol', type=float, default=0.05)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    ref, ref_ep = val_rollout_at(a.reference, a.epoch)
    cands = []
    for c in a.candidate:
        v, _, run = c.partition('=')
        loss, ep = val_rollout_at(run, a.epoch)
        cands.append({'value': float(v), 'run': run, 'val_rollout': loss, 'epoch': ep,
                      'ratio': loss / ref, 'ok': loss <= (1 + a.tol) * ref})
    ok = [c for c in cands if c['ok']]
    chosen = max(ok, key=lambda c: c['value']) if ok else min(cands, key=lambda c: c['value'])
    res = {'chosen': chosen['value'], 'constraint_met': bool(ok), 'reference': a.reference,
           'reference_val_rollout': ref, 'epoch': ref_ep, 'tol': a.tol, 'candidates': cands}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2))
    for c in cands:
        print(f'  {c["value"]:g}: val rollout {c["val_rollout"]:.4f} ({c["ratio"]:.3f} x reference) '
              f'{"ok" if c["ok"] else "too costly"}')
    print(f'[pick] {chosen["value"]:g}' + ('' if ok else ' (no candidate met the constraint: smallest taken)'))


if __name__ == '__main__':
    main()
