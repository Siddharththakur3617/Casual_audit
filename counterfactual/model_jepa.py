"""JEPA on rendered query windows. Same calls as probing/extract_jepa.py (encode, predict_from, contexts).

Window frame f is time t-5+f. z_f = encode(o_f, o_{f-1}) for f >= 1. The predictor at t sees z_{t-4..t}
(frames 1..5) and u_{t-5..t} (window actions 0..5), as in extract_jepa.history_inputs.
"""
from __future__ import annotations

import numpy as np
import torch

from counterfactual.queries import HIST
from probing import extract_jepa as xj


class JEPARunner:
    def __init__(self, ckpt, data_dir, device='cuda', batch=256):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model, self.action_scale, self.epoch = xj.load_model(ckpt, self.device, data_dir)
        self.hist, self.ctx = self.model.history_size, self.model.action_context_dim
        assert self.hist == HIST and self.hist + self.ctx - 1 == HIST + 1, 'window layout assumes 5 latents, 2 actions'
        self.size = int(self.model.cfg.get('image_size', 128))
        self.batch = batch

    def _img(self, a):
        x = torch.from_numpy(a).to(self.device).permute(0, 3, 1, 2).float().div_(255.)
        if x.shape[-1] != self.size:
            x = torch.nn.functional.interpolate(x, size=(self.size, self.size), mode='bilinear', align_corners=False)
        return x

    @torch.no_grad()
    def encode_windows(self, frames):
        """frames (n, W, h, w, 3) uint8 -> Z (n, W, d) float32, Z[:, 0] = NaN."""
        n, W = frames.shape[:2]
        prev = frames[:, :-1].reshape(-1, *frames.shape[2:])
        cur = frames[:, 1:].reshape(-1, *frames.shape[2:])
        out = []
        for i in range(0, len(cur), self.batch):
            out.append(self.model.encode(self._img(cur[i:i + self.batch]), self._img(prev[i:i + self.batch])).float().cpu())
        z = torch.cat(out).reshape(n, W - 1, -1).numpy()
        return np.concatenate([np.full((n, 1, z.shape[-1]), np.nan, np.float32), z], axis=1)

    @torch.no_grad()
    def rollout(self, Z, U, K=1):
        """Z (n, W, d) encoder latents, U (n, W-1) raw actions -> zhat (n, K, d): zhat[:, k] predicts z_{t+k+1}.
        Step k > 0 appends the previous prediction to the 5-latent history (autoregressive, factual actions)."""
        out = []
        for i in range(0, len(Z), self.batch):
            zh = torch.from_numpy(Z[i:i + self.batch, 1:HIST + 1]).to(self.device)            # z_{t-4..t}
            u = torch.from_numpy(U[i:i + self.batch]).to(self.device).float() / self.action_scale
            preds = []
            for k in range(K):
                useq = u[:, k:k + HIST + 1]                                                   # u_{t-5+k..t+k}
                p = xj.predict_from(self.model, zh, useq)
                preds.append(p)
                zh = torch.cat([zh[:, 1:], p[:, None]], dim=1)
            out.append(torch.stack(preds, 1).float().cpu())
        return torch.cat(out).numpy()
