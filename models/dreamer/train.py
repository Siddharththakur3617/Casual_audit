"""Step 3 (DreamerV3): train the DreamerV3 world model offline on a dataset, pixels only.

Same data, splits and checkpoint layout as the JEPA (models/jepa/train.py): checkpoint_epoch####.pt
every `checkpoint_every` epochs, epoch 0 = untrained model (floor), checkpoint_last.pt for --resume,
history.json, config.yaml (with model_type: dreamerv3, which run_all.py and probing/extract.py read).
One epoch = every training episode once, as whole episodes from t = 0 (datagen/episodes.py).
Every `report_every` epochs: reports/epoch####.png with real frames, posterior reconstructions and an
open-loop prior rollout after `open_loop_context` frames (validation episodes).

Run: python -m models.dreamer.train --data datasets/cc_phys --config configs/models/dreamer_cartpole.yaml --save-dir runs/dreamer_cc_phys
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from datagen.episodes import make_loaders
from models.dreamer.model import DEVIATIONS, DreamerWM, LaProp
from paths import MODEL_CONFIGS

NAMES = ('loss', 'rec', 'dyn', 'rep', 'kl')


def save_atomic(obj, path):
    """Write to a temp file, then rename: a crash mid-save cannot corrupt the checkpoint."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def trim(batch, device):
    """Cut the padding shared by the whole batch and move it to the device."""
    L = int(batch['length'].max())
    return {k: batch[k][:, :L].to(device, non_blocking=True) for k in ('obs', 'prev_actions', 'mask')}


def run_epoch(model, loader, device, tc, opt=None):
    tot, n = np.zeros(len(NAMES)), 0
    for batch in loader:
        b = trim(batch, device)
        with torch.set_grad_enabled(opt is not None):
            loss, parts = model.loss(b, tc['loss_scales'], float(tc['free_nats']))
        if opt is not None:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        w = int(b['mask'].sum())
        tot += w * np.array([loss.item()] + [parts[k].item() for k in NAMES[1:]]); n += w
    return dict(zip(NAMES, (tot / max(n, 1)).tolist()))


@torch.no_grad()
def report_png(model, loader, device, context, path, n_eps=4):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    b = trim(next(iter(loader)), device)
    keep = torch.argsort(b['mask'].sum(1), descending=True)[:n_eps]
    obs, pa = b['obs'][keep], b['prev_actions'][keep]
    T = min(int(b['mask'][keep].sum(1).min()), context + 10)
    obs, pa = obs[:, :T], pa[:, :T]
    h, _, post, _ = model.observe(obs, pa, sample=False)
    rec = torch.stack([model.decoder(h[:, t], model.rssm.mode(post[:, t]).flatten(1)) for t in range(T)], 1)
    ol = model.open_loop(obs, pa, context)
    rows = []
    for i in range(len(keep)):
        rows += [obs[i].float() / 255, rec[i], ol[i]]
    fig, axes = plt.subplots(len(rows), T, figsize=(T * 0.9, len(rows) * 0.95))
    for r, seq in enumerate(rows):
        for t in range(T):
            ax = axes[r, t]
            ax.imshow(seq[t].permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(('real', 'posterior', 'open loop')[r % 3], fontsize=6)
            if r == 0:
                ax.set_title(f't={t}' + (' |' if t == context else ''), fontsize=6)
    fig.suptitle(f'open loop: prior only from t={context}', fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--config', default=str(MODEL_CONFIGS / 'dreamer_cartpole.yaml'))
    p.add_argument('--save-dir', required=True)
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', default=None, help='checkpoint to continue from (e.g. checkpoint_last.pt)')
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('[warning] CUDA is not available, training on CPU (very slow)')
    cfg_all = yaml.safe_load(Path(args.config).read_text())
    mc, tc = cfg_all['model'], dict(cfg_all['training'])
    if args.epochs is not None:
        tc['epochs'] = args.epochs
    if args.batch_size is not None:
        tc['batch_size'] = args.batch_size

    out = Path(args.save_dir)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(yaml.safe_dump({'model_type': 'dreamerv3', 'model': mc, 'training': tc,
                                                     'deviations_from_official': list(DEVIATIONS)}))
    loaders, action_scale = make_loaders(args.data, int(tc['batch_size']), int(tc.get('num_workers', 0)))

    model = DreamerWM(mc).to(device)
    print(f'[model] DreamerV3 world model, {sum(p.numel() for p in model.parameters()):,} params, '
          f'feat dim {model.feat_dim}, device={device}')
    opt = LaProp(model.parameters(), lr=float(tc['lr']), beta1=float(tc['beta1']), beta2=float(tc['beta2']),
                 eps=float(tc['eps']), agc=float(tc['agc']), warmup=int(tc['warmup']))
    start, history = 1, []

    def ckpt(epoch, val_loss):
        return {'model_type': 'dreamerv3', 'epoch': epoch, 'model': model.state_dict(),
                'optimizer': opt.state_dict(), 'model_config': mc, 'training_config': tc,
                'action_scale': action_scale, 'val_loss': val_loss,
                'rng': {'torch': torch.get_rng_state(), 'numpy': np.random.get_state()}}

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        start = int(ck['epoch']) + 1
        hp = out / 'history.json'
        history = [r for r in (json.loads(hp.read_text()) if hp.exists() else []) if r['epoch'] < start]
        print(f'[resume] from epoch {ck["epoch"]}')
    else:
        save_atomic(ckpt(0, float('nan')), out / 'checkpoints' / 'checkpoint_epoch0000.pt')
        report_png(model, loaders['val'], device, int(tc['open_loop_context']), out / 'reports' / 'epoch0000.png')

    epochs = int(tc['epochs'])
    for epoch in range(start, epochs + 1):
        t0 = time.time()
        model.train()
        train = run_epoch(model, loaders['train'], device, tc, opt)
        model.eval()
        val = run_epoch(model, loaders['val'], device, tc)
        row = {'epoch': epoch, 'train': train, 'val': val, 'sec': round(time.time() - t0, 1)}
        history.append(row)
        print(f'[epoch {epoch:3d}/{epochs}] ' + ' '.join(f'{k}={v:.3f}' for k, v in train.items())
              + ' | val ' + ' '.join(f'{k}={v:.3f}' for k, v in val.items()) + f' | {row["sec"]:.0f}s', flush=True)
        if epoch % int(tc['report_every']) == 0 or epoch == epochs:
            report_png(model, loaders['val'], device, int(tc['open_loop_context']),
                       out / 'reports' / f'epoch{epoch:04d}.png')
        if epoch % int(tc['checkpoint_every']) == 0 or epoch == epochs:
            save_atomic(ckpt(epoch, val['loss']), out / 'checkpoints' / f'checkpoint_epoch{epoch:04d}.pt')
        save_atomic(ckpt(epoch, val['loss']), out / 'checkpoints' / 'checkpoint_last.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2))
    print(f'[done] -> {out}')


if __name__ == '__main__':
    main()
