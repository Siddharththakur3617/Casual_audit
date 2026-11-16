"""JEPA with per-output parent masks (report route 2, CDL-style), for any environment.

Encoder and inverse model are the standard JEPA's (imported from models/jepa/model.py, not
modified). Only the predictor differs. For every slot j there is one MLP f_j:

  zhat_j(k+1) = z_j(k) + f_j( [z_{k-H+1..k} * cmask_j at every lag],  a_ctx(k) * amask_j )

- Slots that are not parents of the change of j are multiplied by zero before the first layer,
  so they cannot affect output j whatever the weights. Older lags carry only the same parents.
- Actions enter raw and masked per output (amask): on Acrobot the distractor has no action parent.
- z_j(k) is added back as integration and is not an input unless self is a parent.
- ctx_pool (optional): the ctx slot is fed to every f_j as its running mean over the window, so
  fast-changing state cannot ride on it (ctx should hold constant per-episode physics).
- No attention and no BatchNorm, so the masks act in a single hop.
- per_output_hidden: 'auto' picks the width whose parameter count matches the standard
  transformer predictor built from the same config (resolve_hidden).
encode/predict have the same shapes as models/jepa/model.JEPA, so probing/, run_all.py and
evaluate/compare_training.py work unchanged (see probing/extract_jepa.model_class).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.jepa.model import ARPredictor, InverseModel, PatchViTEncoder
from models.jepa_peroutput import graph

MODEL_CLASS = 'models.jepa_peroutput.model:JEPAPerOutput'      # stored in checkpoints (probing/extract_jepa)
TRAIN_MODULE = 'models.jepa_peroutput.train'


def per_output_params(hidden, depth, K, H, d, ctx):
    """Parameter count of PerOutputPredictor (K slots summing to d)."""
    in_dim = H * d + ctx
    return K * ((in_dim + 1) * hidden + (depth - 1) * (hidden * hidden + hidden)) + hidden * d + d


def transformer_params(cfg):
    """Parameter count of the standard predictor (models/jepa/model.ARPredictor) for this config."""
    p = ARPredictor(latent_dim=int(cfg.get('latent_dim', 192)), hidden_dim=int(cfg.get('predictor_hidden_dim', 192)),
                    history_size=int(cfg.get('history_size', 5)), depth=int(cfg.get('predictor_depth', 6)),
                    heads=int(cfg.get('predictor_heads', 16)), dim_head=int(cfg.get('predictor_dim_head', 64)),
                    mlp_dim=int(cfg.get('predictor_mlp_dim', 2048)))
    return sum(q.numel() for q in p.parameters())


def resolve_hidden(cfg):
    """Replace per_output_hidden: 'auto' by the multiple of 8 whose parameter count is closest to the
    transformer predictor's (in place). Stores the target in per_output_param_target."""
    if str(cfg.get('per_output_hidden', 'auto')) != 'auto':
        cfg['per_output_hidden'] = int(cfg['per_output_hidden'])
        return cfg
    K, d = len(cfg['slot_sizes']), int(cfg.get('latent_dim', 192))
    H, ctx, depth = int(cfg.get('history_size', 5)), int(cfg.get('action_context_dim', 2)), int(cfg.get('per_output_depth', 3))
    target = transformer_params(cfg)
    a = K * (depth - 1)
    b = K * (H * d + ctx + 1) + K * (depth - 1) + d
    h0 = (-b + math.sqrt(b * b + 4 * a * (target - d))) / (2 * a) if a else (target - d) / b
    cands = [max(8, 8 * k) for k in range(int(h0 // 8) - 2, int(h0 // 8) + 3)]
    best = min(cands, key=lambda h: abs(per_output_params(h, depth, K, H, d, ctx) - target))
    cfg.update(per_output_hidden=int(best), per_output_param_target=int(target))
    return cfg


class PerOutputPredictor(nn.Module):
    def __init__(self, spec, latent_dim=192, history_size=5, action_dim=2, hidden_dim=768, depth=3, ctx_pool=False):
        super().__init__()
        self.sizes = [int(s) for s in spec['sizes']]
        assert sum(self.sizes) == latent_dim
        self.K, self.d, self.H, self.c = len(self.sizes), int(latent_dim), int(history_size), int(action_dim)
        self.slices = spec['slices']
        self.ctx_slice = spec['slices'][spec['ctx']] if ctx_pool else None
        cm = torch.as_tensor(spec['cmask'], dtype=torch.bool)
        am = torch.as_tensor(spec['amask'], dtype=torch.float32)
        assert am.shape == (self.K, self.c), f'action mask {tuple(am.shape)} does not match {self.K} slots x {self.c}'
        dim_masks = torch.zeros(self.K, self.d)                 # (output slot, latent dim)
        for j in range(self.K):
            for i in range(self.K):
                if cm[j, i]:
                    dim_masks[j, self.slices[i]] = 1.
        # the window is flattened as (lag, dim), so the per-dim mask repeats once per lag
        self.register_buffer('win_masks', dim_masks.repeat(1, self.H), persistent=False)
        self.register_buffer('act_masks', am, persistent=False)
        self.nets = nn.ModuleList()
        for s in self.sizes:
            layers = [nn.Linear(self.H * self.d + self.c, hidden_dim), nn.GELU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
            layers.append(nn.Linear(hidden_dim, s))
            net = nn.Sequential(*layers)
            nn.init.normal_(net[-1].weight, std=0.02)           # small but non-zero: parents act from step 0
            nn.init.zeros_(net[-1].bias)
            self.nets.append(net)

    def forward(self, z, a_ctx):
        """z (b, t, d), a_ctx (b, t, action_dim) raw scaled actions -> (b, t, d); position k predicts z_{k+1}."""
        b, t, d = z.shape
        zin = z
        if self.ctx_slice is not None:                          # running mean of ctx over the window
            c = z[..., self.ctx_slice]
            cm = c.cumsum(1) / torch.arange(1, t + 1, device=z.device, dtype=z.dtype)[None, :, None]
            zin = torch.cat([z[..., :self.ctx_slice.start], cm, z[..., self.ctx_slice.stop:]], dim=-1)
        zp = F.pad(zin, (0, 0, self.H - 1, 0))                  # zeros before the first latent
        win = zp.unfold(1, self.H, 1).permute(0, 1, 3, 2).reshape(b, t, self.H * d)
        out = [z[..., sl] + net(torch.cat([win * self.win_masks[j], a_ctx * self.act_masks[j]], dim=-1))
               for j, (sl, net) in enumerate(zip(self.slices, self.nets))]
        return torch.cat(out, dim=-1)


class JEPAPerOutput(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = resolve_hidden(graph.resolve(dict(cfg)))
        cfg = self.cfg
        d = int(cfg.get('latent_dim', 192))
        self.history_size = int(cfg.get('history_size', 5))
        self.action_context_dim = int(cfg.get('action_context_dim', 2))
        self.encoder = PatchViTEncoder(
            image_size=int(cfg.get('image_size', 128)), patch_size=int(cfg.get('patch_size', 8)),
            in_channels=9, embed_dim=int(cfg.get('vit_embed_dim', 192)),
            depth=int(cfg.get('vit_depth', 4)), heads=int(cfg.get('vit_heads', 3)), latent_dim=d)
        self.spec = graph.spec_from_cfg(cfg)
        self.predictor = PerOutputPredictor(
            self.spec, latent_dim=d, history_size=self.history_size, action_dim=self.action_context_dim,
            hidden_dim=int(cfg['per_output_hidden']), depth=int(cfg.get('per_output_depth', 3)),
            ctx_pool=bool(cfg.get('ctx_pool', False)))
        self.inverse_model = (InverseModel(d, self.action_context_dim,
                                           int(cfg.get('inverse_hidden_dim', 256)),
                                           int(cfg.get('inverse_seq_len', 4)))
                              if cfg.get('use_inverse_model', True) else None)
        # optional anchoring: one linear readout per state slot to its true variable
        # (2 outputs, sin and cos, for wrapped angles)
        self.anchor_heads = (nn.ModuleList([nn.Linear(self.spec['sizes'][i], k)
                                            for i, k in zip(self.spec['state_idx'], self.spec['anchor_dims'])])
                             if cfg.get('slot_anchor_heads', False) else None)

    def encode(self, obs, prev_obs):
        """obs, prev_obs: (b, 3, h, w) floats in [0, 1]."""
        return self.encoder(torch.cat([prev_obs, obs, obs - prev_obs], dim=1))

    def predict(self, z_hist, a_ctx):
        """z_hist (b, t, d), a_ctx (b, t, ctx) -> (b, t, d); position k predicts z_{k+1}."""
        return self.predictor(z_hist, a_ctx.float())
