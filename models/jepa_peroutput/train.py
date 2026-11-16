"""Train the per-output-mask JEPA (graph-prior arm P, report route 2). Cart-pole or Acrobot.

Same data, schedule, checkpoint layout, CLI and history.json keys as models/jepa/train.py, so
run_all.py and evaluate/compare_training.py read these runs unchanged. Losses (weights in
configs/models/graph_priors/*.yaml):
  rollout  start from `history` encoded latents, predict, slide the window, `rollout_steps` times
  fwd      optional teacher-forced one-step loss (off by default)
  inverse  recover the action contexts from the last latents
  sigreg   keep latents close to an isotropic Gaussian (prevents collapse)
  ctx      the ctx slot (hidden physics) should not change within a window
  anchor   optional (default 0): a linear head on slot i predicts its true variable (sin/cos for
           wrapped angles), standardised with training-set statistics. anchor_fraction < 1 uses
           labels from that fraction of training EPISODES only (fixed by anchor_seed).
On probe epochs, history.json gets graph_sensitivity_val: the finite-difference sensitivity of each
predicted CHANGE (zhat_j - z_j(t)) to slots that are / are not its parents in the uncorrupted
expert graph (self pairs included), so the dense and corrupted controls are scored on the same edges.

The resolved model config (slot layout, masks, hidden width) is saved in config.yaml and in every
checkpoint. --set overrides any config key, e.g. --set training.lambda_ctx=1.0 model.graph_corrupt.drop=2

Run: python -m models.jepa_peroutput.train --data datasets/cc_phys \
       --config configs/models/graph_priors/jepa_cartpole_peroutput.yaml --save-dir runs/jepa_cc_phys_peroutput_e100
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
from torch.utils.data import DataLoader, Dataset

import envs
from datagen.windows import make_loaders
from models.jepa.sigreg import sigreg_loss
from models.jepa.train import BN_RECAL_BATCHES, action_contexts, encode_window, needs_bn_recal, ridge_r2
from models.jepa_peroutput import graph
from models.jepa_peroutput.model import MODEL_CLASS, TRAIN_MODULE, JEPAPerOutput
from paths import MODEL_CONFIGS

LOSS_NAMES = ['loss', 'rollout', 'fwd', 'inv', 'sig', 'ctx', 'anchor']


def loss_cfg(tc):
    cfg = {k: float(tc.get(k, 0.0)) for k in ('lambda_fwd', 'lambda_inverse', 'lambda_rollout',
                                              'lambda_sigreg', 'lambda_ctx', 'lambda_anchor')}
    cfg.update(history=int(tc['history']), rollout_steps=int(tc['rollout_steps']),
               sigreg_num_slices=int(tc.get('sigreg_num_slices', 128)),
               sigreg_num_points=int(tc.get('sigreg_num_points', 17)),
               anchor_mean=tc.get('anchor_mean'), anchor_std=tc.get('anchor_std'))
    return cfg


def anchor_targets(states, env):
    """states (b, n, S) at consecutive frames -> (b, n-1, V): for each position column of the env
    (cart-pole x, theta; Acrobot th1, th2, p) the value at frames 1..n-1 (sin, cos for wrapped
    angles) and its finite-difference velocity, in the order of env.STATE_NAMES."""
    pos = list(getattr(env, 'POS', [0, 2]))
    ang = list(getattr(env, 'ANGLE', [False] * len(pos)))
    cols = []
    for k, a in zip(pos, ang):
        q = states[..., k]
        step = q[:, 1:] - q[:, :-1]
        if a:
            step = torch.remainder(step + math.pi, 2 * math.pi) - math.pi
            cols += [torch.sin(q[:, 1:]), torch.cos(q[:, 1:])]
        else:
            cols.append(q[:, 1:])
        cols.append(step / env.DT)
    return torch.stack(cols, dim=-1)


def episode_ids(starts):
    """Episode index of every window of a datagen.windows.WindowDataset (a new episode starts at st == 0)."""
    return np.cumsum(np.asarray(starts)[:, 2] == 0) - 1


class LabeledWindows(Dataset):
    """Training windows plus a 'labeled' flag (the window's episode is in the anchor label set)."""

    def __init__(self, ds, labeled):
        self.ds, self.labeled = ds, torch.as_tensor(labeled, dtype=torch.bool)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        item = self.ds[i]
        item['labeled'] = self.labeled[i]
        return item


def losses(model, batch, device, cfg):
    hist, steps, spec = cfg['history'], cfg['rollout_steps'], model.spec
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

    ctx = zero
    if cfg['lambda_ctx'] > 0:
        c = z[..., spec['slices'][spec['ctx']]]
        ctx = F.mse_loss(c[:, 1:], c[:, :-1])

    anc = zero
    if cfg['lambda_anchor'] > 0:
        assert model.anchor_heads is not None, 'lambda_anchor > 0 needs model.slot_anchor_heads: true'
        keep = batch['labeled'].to(device) if 'labeled' in batch else torch.ones(len(z), dtype=torch.bool, device=device)
        if keep.any():
            y = anchor_targets(batch['states'].to(device).float()[keep], envs.current())
            y = (y - y.new_tensor(cfg['anchor_mean'])) / y.new_tensor(cfg['anchor_std'])
            zk = z[keep, 1:]
            p = torch.cat([h(zk[..., spec['slices'][i]]) for h, i in zip(model.anchor_heads, spec['state_idx'])], -1)
            anc = F.mse_loss(p, y)

    total = (cfg['lambda_rollout'] * rl + cfg['lambda_fwd'] * fwd + cfg['lambda_inverse'] * inv
             + cfg['lambda_sigreg'] * sig + cfg['lambda_ctx'] * ctx + cfg['lambda_anchor'] * anc)
    return total, torch.stack([total, rl, fwd, inv, sig, ctx, anc]).detach(), z[:, hist - 1]


def change_sensitivity(model, zh, ah, spec, pairs, sigma=0.1):
    """(P, b): ||c_j(z + delta) - c_j(z)||^2 / ||delta||^2 for each (i, j) pair, where
    c_j = zhat_j - z_j(t) is the predicted change of slot j and delta is noise (sigma x batch std)
    on slot i at every history position. Removing the carry z_j(t) makes self pairs testable."""
    b, t, d = zh.shape
    std = zh.reshape(-1, d).std(0).clamp_min(1e-6)
    c0 = model.predict(zh, ah)[:, -1] - zh[:, -1]
    res = []
    for i, j in pairs:
        si, sj = spec['slices'][i], spec['slices'][j]
        delta = torch.zeros_like(zh)
        delta[..., si] = torch.randn_like(zh[..., si]) * sigma * std[si]
        z1 = zh + delta
        c1 = model.predict(z1, ah)[:, -1] - z1[:, -1]
        res.append(((c1 - c0)[:, sj] ** 2).sum(-1) / (delta ** 2).sum((1, 2)).clamp_min(1e-12))
    return torch.stack(res) if res else zh.new_zeros(0, b)


@torch.no_grad()
def graph_diagnostic(model, loader, device, cfg, max_batches=20, sigma=0.1):
    spec, hist, n = model.spec, cfg['history'], 0
    non, par = spec['expert_non_edges'], spec['expert_parent_edges']
    s_non, s_par = torch.zeros(len(non), device=device), torch.zeros(len(par), device=device)
    model.eval()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == 'cuda' else []):
        torch.manual_seed(0)                                  # same noise every epoch
        for k, batch in enumerate(loader):
            if k >= max_batches:
                break
            z = encode_window(model, batch, device)
            a = action_contexts(model, batch, device)
            s_non += change_sensitivity(model, z[:, :hist], a[:, :hist], spec, non, sigma).sum(1)
            s_par += change_sensitivity(model, z[:, :hist], a[:, :hist], spec, par, sigma).sum(1)
            n += z.shape[0]
    nm = spec['names']
    vn, vp = (s_non / max(n, 1)).cpu().numpy(), (s_par / max(n, 1)).cpu().numpy()
    cross = [v for (i, j), v in zip(non, vn) if i != j]
    selfs = [v for (i, j), v in zip(non, vn) if i == j]
    cross_par = [v for (i, j), v in zip(par, vp) if i != j]
    return {'nonparent_mean': float(np.mean(cross)) if cross else 0.,
            'self_nonparent_mean': float(np.mean(selfs)) if selfs else 0.,
            'parent_mean': float(np.mean(cross_par)) if cross_par else 0.,
            'nonparent': {f'{nm[i]}->d{nm[j]}': float(v) for (i, j), v in zip(non, vn)},
            'parent': {f'{nm[i]}->d{nm[j]}': float(v) for (i, j), v in zip(par, vp)}}


@torch.no_grad()
def recalibrate_bn(model, dataset, device, cfg, batch_size, n_batches=BN_RECAL_BATCHES, seed=0):
    """Precise-BN recalibration with this arm's losses (same method as models/jepa/train.py)."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    if not bns:
        return
    was_training, momenta = model.training, [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed))
    model.train()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == 'cuda' else []):
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


def apply_overrides(cfg_all, sets):
    """--set section.key[.subkey]=value (value parsed as YAML)."""
    for s in sets or []:
        key, _, val = s.partition('=')
        assert _ and '.' in key, f'--set expects section.key=value, got {s!r}'
        node, parts = cfg_all, key.split('.')
        for p in parts[:-1]:
            if not isinstance(node.get(p), dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = yaml.safe_load(val)
        print(f'[set] {key} = {node[parts[-1]]!r}')
    return cfg_all


def anchor_stats(ds, env):
    """Mean and std of the anchor targets over every consecutive frame pair of the training windows."""
    st = torch.as_tensor(np.asarray(ds.starts)[:, 0])
    pair = torch.stack([ds.states[st], ds.states[st + 1]], dim=1).float()
    y = anchor_targets(pair, env)[:, 0]
    return y.mean(0).tolist(), y.std(0).clamp_min(1e-6).tolist()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--config', default=str(MODEL_CONFIGS / 'graph_priors' / 'jepa_cartpole_peroutput.yaml'))
    p.add_argument('--save-dir', required=True)
    p.add_argument('--epochs', type=int)
    p.add_argument('--batch-size', type=int)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', default=None, help='checkpoint to continue from')
    p.add_argument('--probe-every', type=int, help='epochs between R² probes / graph diagnostics')
    p.add_argument('--set', nargs='*', default=[], help='config overrides, e.g. training.lambda_ctx=1.0')
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print('[warning] CUDA is not available, training on CPU (very slow)')
    env = envs.use_data(args.data)
    cfg_all = apply_overrides(yaml.safe_load(Path(args.config).read_text()), args.set)
    mc, tc = dict(cfg_all['model']), dict(cfg_all['training'])
    for key, val in (('epochs', args.epochs), ('batch_size', args.batch_size), ('probe_every', args.probe_every)):
        if val is not None:
            tc[key] = val
    tc['seed'] = args.seed
    hist, steps = int(tc['history']), int(tc['rollout_steps'])
    assert int(mc['history_size']) == hist, 'model.history_size must equal training.history'
    assert int(tc['horizon']) == hist - 1 + steps, 'horizon must be history - 1 + rollout_steps'

    out = Path(args.save_dir)
    (out / 'checkpoints').mkdir(parents=True, exist_ok=True)
    loaders, action_scale = make_loaders(args.data, int(tc['horizon']), int(tc['batch_size']),
                                         int(mc['action_context_dim']), int(tc.get('num_workers', 0)))
    graph.resolve(mc, args.data)                       # slot layout + masks for this dataset's environment
    if args.resume:                                    # keep the exact graph of the run being resumed
        mc = torch.load(args.resume, map_location='cpu', weights_only=False)['model_config']
    model = JEPAPerOutput(mc).to(device)
    mc = model.cfg                                     # resolved: masks and hidden width included

    if float(tc.get('lambda_anchor', 0)) > 0:
        ds = loaders['train'].dataset
        if tc.get('anchor_mean') is None:
            tc['anchor_mean'], tc['anchor_std'] = anchor_stats(ds, env)
        frac = float(tc.get('anchor_fraction', 1.0))
        ep = episode_ids(ds.starts)
        n_ep = int(ep.max()) + 1
        n_lab = max(1, int(math.ceil(frac * n_ep)))
        lab_eps = np.random.default_rng(int(tc.get('anchor_seed', 0))).choice(n_ep, size=n_lab, replace=False)
        labeled = np.isin(ep, lab_eps)
        tc['anchor_labeled_episodes'] = int(n_lab)
        loaders['train'] = DataLoader(LabeledWindows(ds, labeled), batch_size=int(tc['batch_size']), shuffle=True,
                                      drop_last=True, num_workers=int(tc.get('num_workers', 0)),
                                      pin_memory=torch.cuda.is_available())
        print(f'[anchor] labels from {n_lab}/{n_ep} training episodes ({labeled.mean():.1%} of windows)')

    (out / 'config.yaml').write_text(yaml.safe_dump({'model_type': 'jepa', 'model_class': MODEL_CLASS,
                                                     'model': mc, 'training': tc}))
    n_pred = sum(q.numel() for q in model.predictor.parameters())
    target = mc.get('per_output_param_target')
    print(f'[model] {sum(q.numel() for q in model.parameters()):,} params; per-output predictor {n_pred:,}'
          + (f' (target {target:,}, hidden {mc["per_output_hidden"]})' if target else '')
          + f'; env={env.NAME}, history={hist}, device={device}')
    print(f'[graph] source={mc["graph_source"]}, edges={model.spec["edges"]}, '
          f'slots {dict(zip(model.spec["names"], model.spec["sizes"]))}\n' + graph.describe(model.spec))
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
        return {'model_type': 'jepa', 'model_class': MODEL_CLASS, 'train_module': TRAIN_MODULE,
                'epoch': epoch, 'model': model.state_dict(), 'optimizer': opt.state_dict(),
                'scheduler': sched.state_dict(), 'model_config': mc, 'training_config': tc,
                'action_scale': action_scale, 'val_loss': val_loss, 'best_val': best, 'bn_recal': recal}

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        sched.load_state_dict(ck['scheduler'])
        start, best = int(ck['epoch']) + 1, float(ck.get('best_val', float('inf')))
        hp = out / 'history.json'
        history = json.loads(hp.read_text()) if hp.exists() else []
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
        train_s = time.time() - t0            # optimisation time only
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
            row['probe_r2_x_xdot_theta_thetadot'] = r2.tolist()     # key name kept for compare_training
            row['probe_r2_states'] = r2.tolist()                    # all state columns, any environment
            msg += ' | R2=[' + ','.join(f'{v:.3f}' for v in r2) + ']'
            diag = graph_diagnostic(model, loaders['val'], device, cfg)
            row['graph_sensitivity_val'] = diag
            msg += (f' | sens non/self/par={diag["nonparent_mean"]:.2e}/{diag["self_nonparent_mean"]:.2e}'
                    f'/{diag["parent_mean"]:.2e}')
        print(msg + f' | {time.time() - t0:.0f}s', flush=True)
        history.append(row)
        if val[1] < best:                     # best by val rollout loss (comparable across arms)
            best = float(val[1])
            torch.save(ckpt(epoch, best), out / 'checkpoints' / 'checkpoint_best.pt')
        if epoch % int(tc.get('checkpoint_every', 10)) == 0 or epoch == epochs:
            torch.save(ckpt(epoch, float(val[1])), out / 'checkpoints' / f'checkpoint_epoch{epoch:04d}.pt')
        torch.save(ckpt(epoch, float(val[1])), out / 'checkpoints' / 'checkpoint_last.pt')
        (out / 'history.json').write_text(json.dumps(history, indent=2))
        sched.step()
    print(f'[done] best val rollout={best:.5f} -> {out}')


if __name__ == '__main__':
    main()
