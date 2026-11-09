"""Step 4a: run a checkpoint over a data split and save what the later steps need.

For every transition t >= 5 of every episode:
  z_t       = encode(o_t, o_{t-1})                      single latent (used by the probe)
  zhat_next = predictor(z_{t-4..t}, contexts) last pos  the model's prediction of z_{t+1}
  z_next    = encode(o_{t+1}, o_t)                       real next latent (encoder control)
plus the true variables from cc_common.episode_variables (same t's).

Run: python probing/extract.py --ckpt runs/jepa_cc/checkpoints/checkpoint_epoch0200.pt --data datasets/cc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cc_common import PHYS, T_MIN, episode_variables  # noqa: E402

PHYS_KEYS = ('M', 'm', 'l', 'g')
from model.jepa import JEPA  # noqa: E402


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = JEPA(dict(ck['model_config'])).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, float(ck.get('action_scale', 10.0)), int(ck.get('epoch', -1))


def contexts(u_scaled, ctx):
    """u_scaled: (..., L) raw force sequence -> (..., L-ctx+1, ctx) sliding contexts."""
    return u_scaled.unfold(-1, ctx, 1)


@torch.no_grad()
def encode_episode(model, obs, device, image_size):
    """obs (T+1, h, w, 3) uint8 -> z (T+1, d); z[0] is NaN (no previous frame)."""
    x = torch.from_numpy(obs).to(device).permute(0, 3, 1, 2).float().div_(255.)
    if x.shape[-1] != image_size:
        x = torch.nn.functional.interpolate(x, size=(image_size, image_size), mode='bilinear',
                                            align_corners=False)
    z = model.encode(x[1:], x[:-1])
    return torch.cat([torch.full_like(z[:1], float('nan')), z], dim=0)


def history_inputs(model, z_frames, u_scaled, ts):
    """zh (n, hist, d) = z_{t-hist+1..t}; useq (n, hist+ctx-1) = u_{t-hist-ctx+2..t}."""
    hist, ctx = model.history_size, model.action_context_dim
    zh = torch.stack([z_frames[t - hist + 1:t + 1] for t in ts])
    useq = torch.stack([u_scaled[t - hist - ctx + 2:t + 1] for t in ts])
    return zh, useq


def predict_from(model, zh, useq):
    return model.predict(zh, contexts(useq, model.action_context_dim))[:, -1]


@torch.no_grad()
def action_derivatives(model, zh, useq, eps=1e-3):
    """d zhat / d u for u_t, u_{t-1}, u_{t-2} (raw force units), central differences.
    Returns (n, d, 3). useq is in scaled units (u / action_scale)."""
    out = []
    for lag in (0, 1, 2):
        col = useq.shape[1] - 1 - lag
        up, dn = useq.clone(), useq.clone()
        up[:, col] += eps; dn[:, col] -= eps
        out.append((predict_from(model, zh, up) - predict_from(model, zh, dn)) / (2 * eps))
    return torch.stack(out, dim=-1)


def history_sensitivity(model, zh, useq, n_probe=8):
    """Hutchinson estimate of ||d zhat / d z_{t-k}||_F for k = 0..hist-1 (k=0 is z_t)."""
    zh = zh.detach().clone().requires_grad_(True)
    zhat = predict_from(model, zh, useq)
    sq = torch.zeros(zh.shape[0], zh.shape[1], device=zh.device)
    for _ in range(n_probe):
        v = torch.randn_like(zhat)
        g, = torch.autograd.grad((zhat * v).sum(), zh, retain_graph=True)
        sq += (g ** 2).sum(-1) / n_probe
    return sq.sqrt().flip(1).detach()       # column k = latent k steps back


def extract(ckpt_path, data_dir, split='train', device='cuda', out=None):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model, action_scale, epoch = load_model(ckpt_path, device)
    hist, ctx = model.history_size, model.action_context_dim
    assert T_MIN >= hist and T_MIN >= hist + ctx - 2, 'T_MIN too small for this model'
    image_size = int(model.cfg.get('image_size', 128))
    keys = ('z', 'zhat_next', 'z_next', 'inputs', 'targets', 'xdot', 'thetadot',
            'episode', 't', 'dzhat_du', 'hist_sens', 'physics')
    acc = {k: [] for k in keys}
    types = []
    with h5py.File(Path(data_dir) / f'{split}.hdf5', 'r') as f:
        grp = f['episodes']
        for e, key in enumerate(sorted(grp.keys(), key=int)):
            ep = grp[key]
            states, actions = ep['states'][:], ep['actions'][:].reshape(-1)
            v = episode_variables(states, actions)
            if v is None:
                continue
            ts = v['t']
            zf = encode_episode(model, ep['observations'][:], device, image_size)
            u = torch.from_numpy(actions).to(device).float() / action_scale
            zh, useq = history_inputs(model, zf, u, ts)
            with torch.no_grad():
                zhat = predict_from(model, zh, useq)
                # per unit of raw force (N): divide the scaled-unit derivative by action_scale
                dzdu = action_derivatives(model, zh, useq) / action_scale
            acc['z'].append(zf[ts].cpu().numpy())
            acc['zhat_next'].append(zhat.cpu().numpy())
            acc['z_next'].append(zf[ts + 1].cpu().numpy())
            acc['dzhat_du'].append(dzdu.cpu().numpy().astype(np.float32))
            acc['hist_sens'].append(history_sensitivity(model, zh, useq).cpu().numpy())
            for k in ('inputs', 'targets', 'xdot', 'thetadot', 't'):
                acc[k].append(v[k])
            acc['episode'].append(np.full(len(ts), e))
            phys = [float(ep.attrs.get(f'phys_{k}', PHYS[k])) for k in PHYS_KEYS]   # (M, m, l, g)
            acc['physics'].append(np.tile(np.asarray(phys, np.float32), (len(ts), 1)))
            tt = ep.attrs.get('trajectory_type', 'unknown')
            types += [tt.decode() if isinstance(tt, bytes) else str(tt)] * len(ts)
    res = {k: np.concatenate(v) for k, v in acc.items()}
    res['trajectory_type'] = np.asarray(types)
    res['action_scale'] = np.float32(action_scale)
    res['epoch'] = np.int64(epoch)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **res)
    return res


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
