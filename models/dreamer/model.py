"""DreamerV3 world model (no actor-critic, no reward / continue heads), PyTorch port.

Written from the official implementation (github.com/danijar/dreamerv3: configs.yaml, rssm.py) and the
paper (Hafner et al., arXiv 2301.04104); nothing is imported from it. Settings: the official `size12m`
preset (configs/models/dreamer_<env>.yaml). Differences from the official code are listed in DEVIATIONS.

State s_t = (h_t, z_t): h deterministic (block-diagonal GRU-style core), z = `stoch` categoricals with
`classes` classes each (1% unimix).
  core      h_t = f(h_{t-1}, z_{t-1}, a_{t-1})          no frame t: the model's prediction
  prior     p(z_t | h_t)
  posterior q(z_t | h_t, x_t)                           sees frame t
  decoder   x_t ~ dec(h_t, z_t)
Losses per step (masked on padding): rec = sum over pixels of the squared error,
dyn = max(free_nats, KL[sg(q) || p]), rep = max(free_nats, KL[q || sg(p)]),
loss = rec * scale_rec + dyn * scale_dyn + rep * scale_rep.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEVIATIONS = (
    '128x128 frames: one extra encoder/decoder stage (mults (2,3,4,4,4)) so the last map is still 4x4',
    'float32 instead of bfloat16 compute',
    'RMSNorm eps 1e-4 and PyTorch default inits (official: trunc_normal_in) - not checked against nets.py',
    'image loss on targets in [0, 1] (decoder sigmoid), summed over H, W, C',
)


def rms(n, eps=1e-4):
    return nn.RMSNorm(n, eps=eps)


class LinNormAct(nn.Sequential):
    def __init__(self, i, o):
        super().__init__(nn.Linear(i, o, bias=False), rms(o), nn.SiLU())


class BlockLinear(nn.Module):
    """g independent linear maps: (..., g, i) -> (..., g, o)."""

    def __init__(self, g, i, o, bias=True):
        super().__init__()
        self.w = nn.Parameter(torch.randn(g, i, o) / math.sqrt(i))
        self.b = nn.Parameter(torch.zeros(g, o)) if bias else None

    def forward(self, x):
        y = torch.einsum('...gi,gio->...go', x, self.w)
        return y + self.b if self.b is not None else y


class ChannelNorm(nn.Module):
    """RMSNorm over the channel axis of an NCHW map."""

    def __init__(self, c):
        super().__init__()
        self.n = rms(c)

    def forward(self, x):
        return self.n(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Encoder(nn.Module):
    """Per stage: conv k5 stride 1, 2x2 max-pool, RMSNorm, SiLU (official `simple` encoder, strided: False)."""

    def __init__(self, image_size, depth, mults, kernel=5, minres=4):
        super().__init__()
        chans = [depth * m for m in mults]
        assert image_size // 2 ** len(chans) == minres, 'need image_size / 2**stages == minres'
        layers, c_in = [], 3
        for c in chans:
            layers += [nn.Conv2d(c_in, c, kernel, padding=kernel // 2), nn.MaxPool2d(2), ChannelNorm(c), nn.SiLU()]
            c_in = c
        self.net = nn.Sequential(*layers)
        self.out_dim = chans[-1] * minres * minres

    def forward(self, img):                          # (B, 3, H, W) uint8 or float in [0, 255]
        return self.net(img.float() / 255.0 - 0.5).flatten(1)


class Decoder(nn.Module):
    """feat -> spatial (minres x minres x C) from deter (block-linear, `bspace` blocks) + stoch (MLP),
    then per stage: nearest 2x upsample, conv k5, RMSNorm, SiLU; last stage conv to RGB, sigmoid."""

    def __init__(self, image_size, depth, mults, deter, stoch_flat, units, bspace=8, kernel=5, minres=4):
        super().__init__()
        chans = [depth * m for m in mults]
        self.minres, self.c0 = minres, chans[-1]
        sp = minres * minres * chans[-1]
        assert deter % bspace == 0 and sp % bspace == 0
        self.bspace = bspace
        self.from_deter = BlockLinear(bspace, deter // bspace, sp // bspace)
        self.from_stoch = nn.Sequential(LinNormAct(stoch_flat, 2 * units), nn.Linear(2 * units, sp))
        self.norm0 = ChannelNorm(chans[-1])
        ups, c_in = [], chans[-1]
        for c in reversed(chans[:-1]):
            ups += [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c_in, c, kernel, padding=kernel // 2),
                    ChannelNorm(c), nn.SiLU()]
            c_in = c
        ups += [nn.Upsample(scale_factor=2, mode='nearest'), nn.Conv2d(c_in, 3, kernel, padding=kernel // 2)]
        self.ups = nn.Sequential(*ups)

    def forward(self, deter, stoch_flat):
        b = deter.shape[0]
        x0 = self.from_deter(deter.reshape(b, self.bspace, -1)).reshape(b, -1)
        x = (x0 + self.from_stoch(stoch_flat)).reshape(b, self.c0, self.minres, self.minres)
        x = F.silu(self.norm0(x))
        return torch.sigmoid(self.ups(x))             # (B, 3, H, W) in [0, 1]


class RSSM(nn.Module):
    def __init__(self, deter, hidden, stoch, classes, action_dim, embed_dim, blocks=8, unimix=0.01,
                 imglayers=2, obslayers=1):
        super().__init__()
        assert deter % blocks == 0
        self.deter, self.stoch, self.classes, self.blocks, self.unimix = deter, stoch, classes, blocks, unimix
        self.in_h, self.in_z, self.in_a = (LinNormAct(deter, hidden), LinNormAct(stoch * classes, hidden),
                                           LinNormAct(action_dim, hidden))
        gd = deter // blocks
        self.hid = BlockLinear(blocks, 3 * hidden + gd, gd)
        self.hid_norm = rms(deter)
        self.gru = BlockLinear(blocks, gd, 3 * gd)
        img = [LinNormAct(deter, hidden)] + [LinNormAct(hidden, hidden) for _ in range(imglayers - 1)]
        self.prior_net = nn.Sequential(*img, nn.Linear(hidden, stoch * classes))
        obs = [LinNormAct(deter + embed_dim, hidden)] + [LinNormAct(hidden, hidden) for _ in range(obslayers - 1)]
        self.post_net = nn.Sequential(*obs, nn.Linear(hidden, stoch * classes))

    # ---- distributions over (stoch, classes)
    def probs(self, logits):
        p = logits.reshape(*logits.shape[:-1], self.stoch, self.classes).softmax(-1)
        return (1 - self.unimix) * p + self.unimix / self.classes

    @staticmethod
    def sample_st(p):
        """One-hot sample with straight-through gradients."""
        idx = torch.multinomial(p.reshape(-1, p.shape[-1]), 1).reshape(p.shape[:-1])
        oh = F.one_hot(idx, p.shape[-1]).to(p.dtype)
        return oh + p - p.detach()

    @staticmethod
    def mode(p):
        return F.one_hot(p.argmax(-1), p.shape[-1]).to(p.dtype)

    # ---- steps
    def initial(self, b, device):
        return (torch.zeros(b, self.deter, device=device),
                torch.zeros(b, self.stoch, self.classes, device=device))

    def core(self, h, z, a):
        """h_t = f(h_{t-1}, z_{t-1}, a_{t-1}); z one-hot (B, stoch, classes), a (B, A) in [-1, 1]."""
        a = a / torch.clamp(a.abs(), min=1).detach()
        x = torch.cat([self.in_h(h), self.in_z(z.flatten(1)), self.in_a(a)], -1)       # (B, 3*hidden)
        b, g = h.shape[0], self.blocks
        x = torch.cat([x[:, None].expand(b, g, x.shape[-1]), h.reshape(b, g, -1)], -1)
        x = self.hid(x).reshape(b, -1)
        x = F.silu(self.hid_norm(x)).reshape(b, g, -1)
        reset, cand, update = self.gru(x).chunk(3, -1)
        reset, update = torch.sigmoid(reset), torch.sigmoid(update - 1)
        cand = torch.tanh(reset * cand)
        return (update * cand + (1 - update) * h.reshape(b, g, -1)).reshape(b, -1)

    def prior(self, h):
        return self.probs(self.prior_net(h))

    def posterior(self, h, embed):
        return self.probs(self.post_net(torch.cat([h, embed], -1)))


def kl_cat(p, q):
    """KL[p || q] for (..., stoch, classes) probabilities, summed over the stoch latents."""
    return (p * (p.clamp_min(1e-8).log() - q.clamp_min(1e-8).log())).sum((-1, -2))


class DreamerWM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        c = self.cfg
        self.encoder = Encoder(c['image_size'], c['depth'], c['mults'], c['kernel'], c['minres'])
        self.rssm = RSSM(c['deter'], c['hidden'], c['stoch'], c['classes'], c['action_dim'], self.encoder.out_dim,
                         c['blocks'], c['unimix'], c['imglayers'], c['obslayers'])
        self.decoder = Decoder(c['image_size'], c['depth'], c['mults'], c['deter'], c['stoch'] * c['classes'],
                               c['units'], c['bspace'], c['kernel'], c['minres'])
        self.feat_dim = c['deter'] + c['stoch'] * c['classes']

    @staticmethod
    def feat(h, zp):
        """Model state as one vector: [h, z] (z as one-hot samples, modes or probabilities)."""
        return torch.cat([h, zp.flatten(1)], -1)

    def observe(self, obs, prev_actions, sample=True):
        """obs (B, T, 3, H, W) uint8; prev_actions (B, T, A): a_{t-1} for frame t (zeros at t = 0).
        Every sequence starts at its episode's first frame (zero state). Returns per-step
        h, prior probs, posterior probs and the z fed back into the core (sampled or mode)."""
        B, T = obs.shape[:2]
        embed = self.encoder(obs.flatten(0, 1)).reshape(B, T, -1)
        h, z = self.rssm.initial(B, obs.device)
        hs, priors, posts, zs = [], [], [], []
        for t in range(T):
            h = self.rssm.core(h, z, prev_actions[:, t])
            prior = self.rssm.prior(h)
            post = self.rssm.posterior(h, embed[:, t])
            z = self.rssm.sample_st(post) if sample else self.rssm.mode(post)
            hs.append(h); priors.append(prior); posts.append(post); zs.append(z)
        return torch.stack(hs, 1), torch.stack(priors, 1), torch.stack(posts, 1), torch.stack(zs, 1)

    def loss(self, batch, scales, free_nats=1.0):
        """batch: obs (B, T, 3, H, W) uint8, prev_actions (B, T, A), mask (B, T) bool (False = padding)."""
        obs, mask = batch['obs'], batch['mask'].float()
        h, prior, post, z = self.observe(obs, batch['prev_actions'], sample=True)
        B, T = obs.shape[:2]
        rec = self.decoder(h.flatten(0, 1), z.flatten(0, 1).flatten(1))
        target = obs.flatten(0, 1).float() / 255.0
        rec_l = ((rec - target) ** 2).sum((1, 2, 3)).reshape(B, T)
        dyn = kl_cat(post.detach(), prior)
        rep = kl_cat(post, prior.detach())
        dyn_l, rep_l = dyn.clamp_min(free_nats), rep.clamp_min(free_nats)
        n = mask.sum().clamp_min(1)
        parts = {k: (v * mask).sum() / n for k, v in
                 (('rec', rec_l), ('dyn', dyn_l), ('rep', rep_l), ('kl', dyn))}
        total = scales['rec'] * parts['rec'] + scales['dyn'] * parts['dyn'] + scales['rep'] * parts['rep']
        return total, parts

    @torch.no_grad()
    def open_loop(self, obs, prev_actions, context):
        """Posterior on the first `context` frames, then the prior alone (no frames) for the rest.
        Returns decoded frames (B, T, 3, H, W) in [0, 1]."""
        B, T = obs.shape[:2]
        h, _, post, _ = self.observe(obs[:, :context], prev_actions[:, :context], sample=False)
        frames = [self.decoder(h[:, t], self.rssm.mode(post[:, t]).flatten(1)) for t in range(context)]
        hh, z = h[:, -1], self.rssm.mode(post[:, -1])
        for t in range(context, T):
            hh = self.rssm.core(hh, z, prev_actions[:, t])
            z = self.rssm.mode(self.rssm.prior(hh))
            frames.append(self.decoder(hh, z.flatten(1)))
        return torch.stack(frames, 1)


class LaProp(torch.optim.Optimizer):
    """The official DreamerV3 optimizer: RMS-normalise the gradient, then momentum (LaProp), with
    adaptive gradient clipping (agc) per parameter tensor and a linear warmup."""

    def __init__(self, params, lr=4e-5, beta1=0.9, beta2=0.999, eps=1e-20, agc=0.3, warmup=1000):
        super().__init__(params, dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps, agc=agc, warmup=warmup))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st['step'], st['m'], st['v'] = 0, torch.zeros_like(p), torch.zeros_like(p)
                grad = p.grad
                if g['agc']:
                    pn = p.norm().clamp_min(1e-3)
                    gn = grad.norm()
                    grad = grad * torch.clamp(g['agc'] * pn / gn.clamp_min(1e-12), max=1.0)
                st['step'] += 1
                t = st['step']
                st['v'].mul_(g['beta2']).addcmul_(grad, grad, value=1 - g['beta2'])
                vhat = st['v'] / (1 - g['beta2'] ** t)
                st['m'].mul_(g['beta1']).add_(grad / (vhat.sqrt() + g['eps']), alpha=1 - g['beta1'])
                mhat = st['m'] / (1 - g['beta1'] ** t)
                lr = g['lr'] * min(1.0, t / max(g['warmup'], 1))
                p.add_(mhat, alpha=-lr)
