"""Open-loop rollout errors of a JEPA-type checkpoint on a dataset's validation split, any environment.

Same measurement as evaluate/compare_training.rollout_errors, which assumes cart-pole's 4 state
columns. Here the state width comes from the data, so Acrobot (6 columns) and the held-out-physics
sets work too. Two error measures, because raw latent MSE is not comparable across models:
  latent_rel_mse  ||zhat - z||^2 / ||z - mean z||^2 per rollout step
  state_nrmse     a ridge readout fitted on the model's own REAL encoder latents (never predictions)
                  maps predictions to the state; RMSE / state std, per step and state column
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def ridge_fit(z, y, ridge=1e-3):
    zm, zs = z.mean(0), np.maximum(z.std(0), 1e-6)
    x = (z - zm) / zs
    w = np.linalg.solve(x.T @ x + ridge * len(x) * np.eye(x.shape[1]), x.T @ (y - y.mean(0)))
    return lambda q: ((q - zm) / zs) @ w + y.mean(0)


@torch.no_grad()
def rollout_errors(run, data, ckpt_name, steps, batch_size=64, max_batches=40, device='cuda'):
    from torch.utils.data import DataLoader
    from datagen.windows import WindowDataset, load_meta
    from models.jepa.train import action_contexts, encode_window
    from probing.extract_jepa import load_model
    device = torch.device(device) if isinstance(device, str) else device
    if device.type == 'cuda' and not torch.cuda.is_available():
        device = torch.device('cpu')
    model, scale, epoch = load_model(Path(run) / 'checkpoints' / ckpt_name, device, data)
    hist = model.history_size
    ds = WindowDataset(Path(data) / 'val.hdf5', hist - 1 + steps, model.action_context_dim,
                       float(load_meta(data)['action_scale']))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=torch.Generator().manual_seed(0))
    P, Y, S, ZH, SH = [], [], [], [], []
    for i, batch in enumerate(dl):
        if i >= max_batches:
            break
        z = encode_window(model, batch, device)
        a = action_contexts(model, batch, device)
        zw, preds = z[:, :hist], []
        for k in range(steps):
            p = model.predict(zw, a[:, k:k + hist])[:, -1]
            preds.append(p)
            zw = torch.cat([zw[:, 1:], p[:, None]], dim=1)
        ns = batch['states'].shape[-1]
        P.append(torch.stack(preds, 1).cpu().numpy()); Y.append(z[:, hist:].cpu().numpy())
        S.append(batch['states'][:, hist:].numpy())
        ZH.append(z[:, :hist].reshape(-1, z.shape[-1]).cpu().numpy())
        SH.append(batch['states'][:, :hist].reshape(-1, ns).numpy())
    P, Y, S = np.concatenate(P), np.concatenate(Y), np.concatenate(S)
    zf = Y.reshape(-1, Y.shape[-1])
    lat = ((P - Y) ** 2).sum(-1).mean(0) / ((zf - zf.mean(0)) ** 2).sum(-1).mean()
    read = ridge_fit(np.concatenate(ZH), np.concatenate(SH))
    sp = read(P.reshape(-1, P.shape[-1])).reshape(S.shape)
    nrmse = np.sqrt(((sp - S) ** 2).mean(0)) / np.maximum(np.concatenate(SH).std(0), 1e-9)
    return {'epoch': epoch, 'n_windows': int(len(P)), 'latent_rel_mse': lat.tolist(), 'state_nrmse': nrmse.tolist()}
