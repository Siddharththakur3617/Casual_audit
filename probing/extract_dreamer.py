"""Step 4a (DreamerV3): run a checkpoint over a data split and save what the later steps need.
Called through probing/extract.py, which picks the extractor from the checkpoint's model type.

Each episode is filtered from its first frame (zero state), feeding back the posterior MODE (deterministic).
Features use the categorical PROBABILITIES: feat = [h, probs(z)]. For every transition t >= T_MIN:
  z          = feat(h_t,     q(z_t | h_t, x_t))             posterior at t
  z_next     = feat(h_{t+1}, q(z_{t+1} | h_{t+1}, x_{t+1}))  posterior at t+1: sees frame t+1 (encoder control)
  zhat_next  = feat(h_{t+1}, p(z_{t+1} | h_{t+1}))           prior at t+1: the prediction, no frame t+1
with h_{t+1} = core(h_t, z_t, u_t). Prior and posterior share h_{t+1}, so z_next - zhat_next lives in the
z part only; z_stoch / z_next_stoch / zhat_stoch (the z part alone) give the z-only diagnostic probe.
dzhat_du: d zhat_next / d u for u_t, u_{t-1}, u_{t-2} per raw action unit, central differences; a perturbed
action is propagated through the core with the later posterior modes held fixed (the mode is piecewise
constant, so this is its derivative almost everywhere). True variables and rows come from
env.episode_variables, exactly as for the JEPA, so the dataset's nulls and data baselines apply.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

import envs
from datagen.episodes import prev_actions
from models.dreamer.model import DreamerWM

EPS = 1e-3          # finite-difference step in scaled action units (as in extract_jepa.action_derivatives)


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = DreamerWM(dict(ck['model_config'])).to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    return model, float(ck['action_scale']), int(ck.get('epoch', -1))


@torch.no_grad()
def filter_episode(model, obs, pa, device):
    """obs (T+1, h, w, 3) uint8, pa (T+1, 1) -> h (T+1, D), prior / post probs (T+1, S, C), z modes (T+1, S, C)."""
    o = torch.from_numpy(obs).to(device).permute(0, 3, 1, 2)[None]
    a = torch.from_numpy(pa).to(device)[None]
    h, prior, post, z = model.observe(o, a, sample=False)
    return h[0], prior[0], post[0], z[0]


@torch.no_grad()
def action_derivatives(model, h, z, pa, ts):
    """(n, feat_dim, 3): d feat(prior at t+1) / d a_{t-lag} in scaled units, lag 0, 1, 2."""
    rssm, out = model.rssm, []
    for lag in (0, 1, 2):
        ends = []
        for sgn in (1.0, -1.0):
            s = ts - lag                                  # step whose action is perturbed: a_s moves h_s -> h_{s+1}
            hh = h[s]
            a = pa[s + 1].clone()                         # pa[k] = a_{k-1}, so a_s = pa[s + 1]
            a += sgn * EPS
            hh = rssm.core(hh, z[s], a)
            for j in range(1, lag + 1):                   # replay later steps with the original actions / modes
                hh = rssm.core(hh, z[s + j], pa[s + j + 1])
            ends.append(model.feat(hh, rssm.prior(hh)))
        out.append((ends[0] - ends[1]) / (2 * EPS))
    return torch.stack(out, -1)


def extract(ckpt_path, data_dir, split='train', device='cuda', out=None):
    env = envs.use_data(data_dir)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model, action_scale, epoch = load_model(ckpt_path, device)
    D = model.cfg['deter']
    keys = ('z', 'zhat_next', 'z_next', 'dzhat_du', 'inputs', 'targets') + tuple(env.EXTRA_KEYS) + (
            'episode', 't', 'physics')
    acc = {k: [] for k in keys}
    types = []
    with h5py.File(Path(data_dir) / f'{split}.hdf5', 'r') as f:
        grp = f['episodes']
        for e, key in enumerate(sorted(grp.keys(), key=int)):
            ep = grp[key]
            states, actions = ep['states'][:], ep['actions'][:].reshape(-1)
            v = env.episode_variables(states, actions)
            if v is None:
                continue
            ts = v['t']
            pa_np = prev_actions(actions, action_scale)
            h, prior, post, z = filter_episode(model, ep['observations'][:], pa_np, device)
            pa = torch.from_numpy(pa_np).to(device)
            ti = torch.as_tensor(ts, device=device)
            with torch.no_grad():
                acc['z'].append(model.feat(h[ti], post[ti]).cpu().numpy())
                acc['z_next'].append(model.feat(h[ti + 1], post[ti + 1]).cpu().numpy())
                acc['zhat_next'].append(model.feat(h[ti + 1], prior[ti + 1]).cpu().numpy())
                # per raw action unit: divide the scaled-unit derivative by action_scale
                acc['dzhat_du'].append((action_derivatives(model, h, z, pa, ti) / action_scale).cpu().numpy())
            for k in ('inputs', 'targets', 't') + tuple(env.EXTRA_KEYS):
                acc[k].append(v[k])
            acc['episode'].append(np.full(len(ts), e))
            phys = [float(ep.attrs.get(f'phys_{k}', env.PHYS[k])) for k in env.PHYS_KEYS]
            acc['physics'].append(np.tile(np.asarray(phys, np.float32), (len(ts), 1)))
            tt = ep.attrs.get('trajectory_type', 'unknown')
            types += [tt.decode() if isinstance(tt, bytes) else str(tt)] * len(ts)
    res = {k: np.concatenate(v) for k, v in acc.items()}
    for k in ('z', 'z_next', 'zhat_next', 'dzhat_du'):
        res[k] = res[k].astype(np.float32)
    # z part alone (after the D deterministic dimensions), for the z-only diagnostic probe
    res['z_stoch'], res['z_next_stoch'], res['zhat_stoch'] = (res['z'][:, D:], res['z_next'][:, D:],
                                                               res['zhat_next'][:, D:])
    res['dzhat_du_stoch'] = res['dzhat_du'][:, D:]
    res['trajectory_type'] = np.asarray(types)
    res['action_scale'] = np.float32(action_scale)
    res['epoch'] = np.int64(epoch)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, **res)
    return res
