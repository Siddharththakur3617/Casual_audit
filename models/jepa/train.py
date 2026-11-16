"""Step 3 (JEPA): train the pixels-only JEPA with a 5-latent predictor history.

Losses (weights in configs/models/jepa_<env>.yaml):
  rollout  start from `history` encoded latents, predict the next latent, slide the
           window (drop the oldest, append the prediction), `rollout_steps` times
  inverse  recover the action contexts from the last history latent + rollout targets
  sigreg   keep latents close to an isotropic Gaussian (prevents collapse)
  fwd      optional teacher-forced one-step loss at every history position (off by default)
  jac      graph-prior arm: finite-difference sensitivity of each predicted slot to a slot the
           expert graph says is not its parent (needs a slot layout; models/jepa/graph.py)
  ctx      graph-prior arms: the ctx slot (hidden physics) should not change within a window
Each epoch row of history.json also records train_s (optimisation seconds) and wall_s (cumulative);
models with a slot layout also log graph_sensitivity_val on probe epochs.

Window layout (horizon = history - 1 + rollout_steps transitions, frames f_0..f_H):
  history latents f_0 .. f_{history-1}; rollout targets f_history .. f_H.
Every frame is encoded with its real previous frame.
The untrained model is saved as checkpoint_epoch0000.pt (lower-bound baseline).

Run: python -m models.jepa.train --data datasets/cc_phys --config configs/models/jepa_cartpole.yaml --save-dir runs/jepa_cc_phys
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from datagen.windows import make_loaders
from models.jepa.model import JEPA
from models.jepa.sigreg import sigreg_loss
from paths import MODEL_CONFIGS


def to_float(x, device):
    return x.to(device, non_blocking=True).float().div_(255.)


def encode_window(model, batch, device):
    obs = batch['obs_seq']                                   # (b, n, 3, h, w) uint8
    b, n = obs.shape[:2]
    prev = torch.cat((batch['prev_obs'][:, None], obs[:, :-1]), dim=1)
    cur = to_float(obs.reshape(b * n, *obs.shape[2:]), device)
    prv = to_float(prev.reshape(b * n, *prev.shape[2:]), device)
    return model.encode(cur, prv).reshape(b, n, -1)


def action_contexts(model, batch, device):
    """(b, H, ctx): context of transition k = [u_{k-ctx+1}, ..., u_k] (oldest first)."""
    a = batch['actions'].to(device).float()                  # (b, H, 1)
    ctx = model.action_context_dim
    if ctx > 1:
        a = torch.cat([batch['prev_actions'].to(device).float()[:, -(ctx - 1):], a], dim=1)
    return a.unfold(1, ctx, 1).reshape(a.shape[0], -1, ctx)


def block_masks(spec, pairs, d, device):
    """(P, d) 0/1 masks of the input block i and output block j of each (i, j) pair."""
    ins, outs = torch.zeros(len(pairs), d, device=device), torch.zeros(len(pairs), d, device=device)
    for p, (i, j) in enumerate(pairs):
        ins[p, spec['slices'][i]] = 1.
        outs[p, spec['slices'][j]] = 1.
    return ins, outs


def edge_sensitivity(model, zh, ah, spec, pairs, sigma, pair_idx=None):
    """Finite-difference sensitivity of predicted slot j to slot i, per sample.

    Adds noise delta (sigma x batch std of each dim of block i) to block i at every history
    position and returns ||zhat_j(z + delta) - zhat_j(z)||^2 / ||delta||^2, an estimate of
    ||d zhat_j / d z_i||_F^2 / n_i. pair_idx (b,) picks one pair per sample; None = every pair,
    returning (P, b). zh is detached, so the penalty shapes the predictor, not the encoder.
    Both passes run with the predictor in eval mode: dropout off, and BatchNorm on running
    statistics, since train-mode batch statistics would spread the perturbation to every slot.
    """
    zh = zh.detach()
    b, t, d = zh.shape
    ins, outs = block_masks(spec, pairs, d, zh.device)
    std = zh.reshape(-1, d).std(0).clamp_min(1e-6)
    was = model.predictor.training
    model.predictor.eval()
    try:
        p0 = model.predict(zh, ah)[:, -1]
        rows = range(len(pairs)) if pair_idx is None else [None]
        res = []
        for r in rows:
            im, om = (ins[pair_idx], outs[pair_idx]) if r is None else (ins[r].expand(b, -1), outs[r].expand(b, -1))
            delta = torch.randn_like(zh) * (sigma * std) * im[:, None, :]
            p1 = model.predict(zh + delta, ah)[:, -1]
            num = ((p1 - p0) ** 2 * om).sum(-1)
            res.append(num / (delta ** 2).sum((1, 2)).clamp_min(1e-12))
    finally:
        model.predictor.train(was)
    return res[0] if pair_idx is not None else torch.stack(res)


def losses(model, batch, device, cfg):
    hist, steps = cfg['history'], cfg['rollout_steps']
    spec = model.graph_spec
    z = encode_window(model, batch, device)                  # (b, hist+steps, d)
    a = action_contexts(model, batch, device)                # (b, hist+steps-1, ctx)
    zero = z.new_zeros(())

    z_win, rl = z[:, :hist], zero
    for k in range(steps):
        pred = model.predict(z_win, a[:, k:k + hist])[:, -1]
        rl = rl + F.mse_loss(pred, z[:, hist + k])
        z_win = torch.cat([z_win[:, 1:], pred[:, None]], dim=1)
    rl = rl / steps

    fwd = zero
    if cfg['lambda_fwd'] > 0:
        fwd = F.mse_loss(model.predict(z[:, :hist], a[:, :hist]), z[:, 1:hist + 1])

    inv = zero
    if model.inverse_model is not None and cfg['lambda_inverse'] > 0:
        L = model.inverse_model.seq_len
        inv = F.mse_loss(model.inverse_model(z[:, -L:]), a[:, -(L - 1):])

    sig = zero
    if cfg['lambda_sigreg'] > 0:
        sig = sigreg_loss(z.reshape(-1, z.shape[-1]), num_slices=cfg['sigreg_num_slices'],
                          num_points=cfg['sigreg_num_points'])

    jac = zero                       # Jacobian sparsity: predicted slot j must not move with non-parent i
    if cfg['lambda_jac'] > 0:
        assert spec is not None and spec['non_edges'], 'lambda_jac needs a slot layout with absent edges'
        pairs = spec['non_edges']
        idx = torch.randint(len(pairs), (z.shape[0],), device=z.device)
        jac = edge_sensitivity(model, z[:, :hist], a[:, :hist], spec, pairs, cfg['jac_sigma'], idx).mean()

    ctx = zero                       # hidden physics are constant within an episode: ctx should be too
    if cfg['lambda_ctx'] > 0:
        assert spec is not None and spec['ctx'] is not None, 'lambda_ctx needs a ctx slot'
        c = z[..., spec['slices'][spec['ctx']]]
        ctx = F.mse_loss(c[:, 1:], c[:, :-1])

    total = (cfg['lambda_rollout'] * rl + cfg['lambda_fwd'] * fwd
             + cfg['lambda_inverse'] * inv + cfg['lambda_sigreg'] * sig
             + cfg['lambda_jac'] * jac + cfg['lambda_ctx'] * ctx)
    return total, torch.stack([total, rl, fwd, inv, sig, jac, ctx]).detach(), z[:, hist - 1]


LOSS_NAMES = ['loss', 'rollout', 'fwd', 'inv', 'sig', 'jac', 'ctx']


@torch.no_grad()
def graph_diagnostic(model, loader, device, cfg, max_batches=20, sigma=0.1):
    """Mean sensitivity of predicted slots to non-parent and to parent slots on a loader.
    Only defined for models with a slot layout. ~0 non-parent for graph_masked by construction."""
    spec = model.graph_spec
    if spec is None:
        return None
    model.eval()
    hist = cfg['history']
    sums = {'non': torch.zeros(len(spec['non_edges']), device=device),
            'par': torch.zeros(len(spec['parent_edges']), device=device)}
    n = 0
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == 'cuda' else []):
        torch.manual_seed(0)                      # same noise every epoch; caller's RNG untouched
        for k, batch in enumerate(loader):
            if k >= max_batches:
                break
            z = encode_window(model, batch, device)
            a = action_contexts(model, batch, device)
            for key, pairs in (('non', spec['non_edges']), ('par', spec['parent_edges'])):
                if pairs:
                    sums[key] += edge_sensitivity(model, z[:, :hist], a[:, :hist], spec, pairs, sigma).sum(1)
            n += z.shape[0]
    nm = spec['names']
    non, par = (sums['non'] / max(n, 1)).cpu().numpy(), (sums['par'] / max(n, 1)).cpu().numpy()
    return {'nonparent_mean': float(non.mean()) if len(non) else 0.,
            'parent_mean': float(par.mean()) if len(par) else 0.,
            'nonparent': {f'{nm[i]}->{nm[j]}': float(v) for (i, j), v in zip(spec['non_edges'], non)},
            'parent': {f'{nm[i]}->{nm[j]}': float(v) for (i, j), v in zip(spec['parent_edges'], par)}}


def ridge_r2(z_tr, y_tr, z_te, y_te, ridge=1e-3):
    zm, zs = z_tr.mean(0), np.maximum(z_tr.std(0), 1e-6)
    x, xt = (z_tr - zm) / zs, (z_te - zm) / zs
    w = np.linalg.solve(x.T @ x + ridge * len(x) * np.eye(x.shape[1]), x.T @ (y_tr - y_tr.mean(0)))
    p = xt @ w + y_tr.mean(0)
    return 1 - ((p - y_te) ** 2).sum(0) / ((y_te - y_te.mean(0)) ** 2).sum(0)


BN_RECAL_BATCHES = 200


def loss_cfg(tc):
    """Loss settings from a training config block (as used by losses())."""
    cfg = {k: float(tc.get(k, 0.0)) for k in ('lambda_fwd', 'lambda_inverse', 'lambda_rollout',
                                              'lambda_sigreg', 'lambda_jac', 'lambda_ctx')}
    cfg.update(history=int(tc['history']), rollout_steps=int(tc['rollout_steps']),
               jac_sigma=float(tc.get('jac_sigma', 0.1)),
               sigreg_num_slices=int(tc.get('sigreg_num_slices', 128)),
               sigreg_num_points=int(tc.get('sigreg_num_points', 17)))
    return cfg


def needs_bn_recal(data_dir):
    """Environments whose BatchNorm running statistics go stale during training (acrobot: the
    encoder projector's running mean ends up ~5 SD off the batch mean, so eval-mode latents are
    shifted and every eval-mode number is wrong). Cart-pole keeps the old behaviour."""
    import envs
    return bool(getattr(envs.get_env(envs.env_name_of(data_dir)), 'RECALIBRATE_BN', False))


@torch.no_grad()
def recalibrate_bn(model, dataset, device, cfg, batch_size, n_batches=BN_RECAL_BATCHES, seed=0):
    """Recompute every BatchNorm's running mean/var as the plain average of the batch statistics
    over n_batches training windows ("precise BN"), with the weights frozen and the training forward
    pass (losses()), so the statistics are the ones the model was trained under. Deterministic
    (own sampler seed and RNG), and it leaves the caller's RNG state and train/eval mode unchanged."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    was_training, momenta = model.training, [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None                     # cumulative average instead of an EMA
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    model.train()
    cuda = [torch.cuda.current_device()] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=cuda):
        torch.manual_seed(seed)
        for i, batch in enumerate(loader):
            if i >= n_batches:
                break
            losses(model, batch, device, cfg)
    for m, mo in zip(bns, momenta):
        m.momentum = mo
    model.train(was_training)


@torch.no_grad()
def run_eval(model, loader, device, cfg, collect=False, max_batches=None):
    model.eval()
    tot, n, zs, ys = np.zeros(len(LOSS_NAMES)), 0, [], []
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        _, parts, z_last = losses(model, batch, device, cfg)
        b = batch['obs_seq'].shape[0]
        tot += b * parts.cpu().numpy(); n += b
        if collect:
            zs.append(z_last.cpu().numpy()); ys.append(batch['states'][:, cfg['history'] - 1].numpy())
    return (tot / max(n, 1), np.concatenate(zs) if collect else None,
            np.concatenate(ys) if collect else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--config', default=str(MODEL_CONFIGS / 'jepa_cartpole.yaml'))
    p.add_argument('--save-dir', required=True)
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', default=None, help='checkpoint to continue from')
    p.add_argument('--probe-every', type=int, help='epochs between R² probes / graph diagnostics')
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
    if args.probe_every is not None:
        tc['probe_every'] = args.probe_every
    tc['seed'] = args.seed
    hist, steps = int(tc['history']), int(tc['rollout_steps'])
    assert int(mc['history_size']) == hist, 'model.history_size must equal training.history'
    assert int(tc['horizon']) == hist - 1 + steps, 'horizon must be history - 1 + rollout_steps'

    out = Path(args.save_dir)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(yaml.safe_dump({'model_type': 'jepa', 'model': mc, 'training': tc}))
    loaders, action_scale = make_loaders(args.data, int(tc['horizon']), int(tc['batch_size']),
                                         int(mc['action_context_dim']), int(tc.get('num_workers', 0)))

    model = JEPA(mc).to(device)
    print(f'[model] {sum(p.numel() for p in model.parameters()):,} params, history={hist}, '
          f'action_context_dim={model.action_context_dim}, predictor={model.predictor_type}, device={device}')
    if model.graph_spec is not None:
        from models.jepa.graph import describe
        print(f'[graph] slots {dict(zip(model.graph_spec["names"], model.graph_spec["sizes"]))}, '
              f'edges={model.graph_spec["edges"]}\n' + describe(model.graph_spec))
    cfg = loss_cfg(tc)
    recal = needs_bn_recal(args.data)
    if recal:
        print(f'[bn] recalibrating BatchNorm statistics on {BN_RECAL_BATCHES} train batches before each eval')

    opt = torch.optim.AdamW(model.parameters(), lr=float(tc['lr']),
                            weight_decay=float(tc.get('weight_decay', 1e-3)))
    epochs, warm = int(tc['epochs']), int(tc.get('warmup_epochs', 5))
    base_lr, min_lr = float(tc['lr']), float(tc.get('min_lr', 1e-6))

    def lr_mult(e):
        if e < warm:
            return (e + 1) / max(warm, 1)
        prog = (e - warm) / max(epochs - warm - 1, 1)
        return (min_lr + (base_lr - min_lr) * .5 * (1 + math.cos(math.pi * prog))) / base_lr

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)
    start, best, history = 1, float('inf'), []

    def ckpt(epoch, val_loss):
        return {'model_type': 'jepa', 'epoch': epoch, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'scheduler': sched.state_dict(), 'model_config': mc, 'training_config': tc,
                'action_scale': action_scale, 'val_loss': val_loss, 'best_val': best,
                'bn_recal': recal}

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        start, best = int(ck['epoch']) + 1, float(ck.get('best_val', float('inf')))
        hp = out / 'history.json'
        history = json.loads(hp.read_text()) if hp.exists() else []
        if recal and not ck.get('bn_recal', False):
            best = float('inf')     # earlier val losses used stale BN statistics: not comparable
            print('[resume] earlier val losses used stale BN statistics; best-checkpoint tracking restarts')
        print(f'[resume] from epoch {ck["epoch"]}')
    else:
        torch.save(ckpt(0, float('nan')), out / 'checkpoints' / 'checkpoint_epoch0000.pt')

    names = LOSS_NAMES
    wall = float(history[-1].get('wall_s', 0.)) if history else 0.
    for epoch in range(start, epochs + 1):
        t0 = time.time()
        model.train()
        tot, n = torch.zeros(len(names), device=device), 0
        for batch in loaders['train']:
            loss, parts, _ = losses(model, batch, device, cfg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            b = batch['obs_seq'].shape[0]
            tot += b * parts; n += b
        train = (tot / n).cpu().numpy()
        if device.type == 'cuda':
            torch.cuda.synchronize()
        train_s = time.time() - t0            # optimisation time only (what "efficiency" compares)
        wall += train_s
        probe = epoch % int(tc.get('probe_every', 10)) == 0 or epoch == 1
        if recal:
            recalibrate_bn(model, loaders['train'].dataset, device, cfg, int(tc['batch_size']))
        val, zv, yv = run_eval(model, loaders['val'], device, cfg, collect=probe)
        row = {'epoch': epoch, 'lr': float(opt.param_groups[0]['lr']),
               'train': dict(zip(names, map(float, train))), 'val': dict(zip(names, map(float, val))),
               'train_s': train_s, 'wall_s': wall}
        if recal:
            row['bn_recal'] = True
        msg = (f'[epoch {epoch:3d}/{epochs}] ' + ' '.join(f'{k}={v:.4f}' for k, v in row['train'].items())
               + ' | val ' + ' '.join(f'{k}={v:.4f}' for k, v in row['val'].items()))
        if probe:
            _, zt, yt = run_eval(model, loaders['train'], device, cfg, collect=True, max_batches=60)
            r2 = ridge_r2(zt, yt, zv, yv)
            row['probe_r2_x_xdot_theta_thetadot'] = r2.tolist()
            msg += ' | R2=[' + ','.join(f'{v:.3f}' for v in r2) + ']'
            diag = graph_diagnostic(model, loaders['val'], device, cfg)
            if diag is not None:
                row['graph_sensitivity_val'] = diag
                msg += f' | sens non/par={diag["nonparent_mean"]:.2e}/{diag["parent_mean"]:.2e}'
        print(msg + f' | {time.time() - t0:.0f}s', flush=True)
        history.append(row)
        if val[0] < best:
            best = float(val[0])
            torch.save(ckpt(epoch, best), out / 'checkpoints' / 'checkpoint_best.pt')
        if epoch % int(tc.get('checkpoint_every', 10)) == 0 or epoch == epochs:
            torch.save(ckpt(epoch, float(val[0])), out / 'checkpoints' / f'checkpoint_epoch{epoch:04d}.pt')
        torch.save(ckpt(epoch, float(val[0])), out / 'checkpoints' / 'checkpoint_last.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        sched.step()
    print(f'[done] best val={best:.5f} -> {out}')


if __name__ == '__main__':
    main()
