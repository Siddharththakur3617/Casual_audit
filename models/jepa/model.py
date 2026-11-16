"""Action-conditioned JEPA world model (pixels only).

Encoder: patch ViT on [previous frame, current frame, difference] (9 channels),
CLS token -> MLP projector -> latent z (latent_dim).
Predictor: causal transformer over the last `history_size` latents, each
conditioned (AdaLN-zero) on its action context [u_{k-ctx+1}, ..., u_k].
Inverse model: MLP from a short latent sequence to the actions between them
(an auxiliary loss that keeps action information in z).

Same architecture as the JEPA_control cart-pole model (patch 8, ViT 4x192,
predictor 6 blocks, hidden 192, 16 heads x 64, MLP 2048).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchViTEncoder(nn.Module):
    def __init__(self, image_size=128, patch_size=8, in_channels=9, embed_dim=192,
                 depth=4, heads=3, latent_dim=192):
        super().__init__()
        self.patch = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)
        n = (image_size // patch_size) ** 2
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(embed_dim, heads, 4 * embed_dim, dropout=0.0,
                                           activation='gelu', batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.projector = nn.Sequential(nn.Linear(embed_dim, 2048), nn.BatchNorm1d(2048), nn.GELU(),
                                       nn.Linear(2048, latent_dim))
        nn.init.trunc_normal_(self.cls, std=.02)
        nn.init.trunc_normal_(self.pos, std=.02)

    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls.expand(len(x), -1, -1), x], 1) + self.pos
        return self.projector(self.norm(self.blocks(x))[:, 0])


class ActionEmbedder(nn.Module):
    def __init__(self, action_dim, embed_dim, mlp_scale=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(action_dim, mlp_scale * embed_dim), nn.SiLU(),
                                 nn.Linear(mlp_scale * embed_dim, embed_dim))

    def forward(self, a):
        return self.net(a.float())


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ConditionalBlock(nn.Module):
    """Causal transformer block with AdaLN-zero action conditioning."""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.heads, self.dim_head, self.dropout = heads, dim_head, dropout
        inner = heads * dim_head
        self.to_qkv = nn.Linear(dim, 3 * inner, bias=False)
        self.to_out = nn.Linear(inner, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim), nn.Dropout(dropout))
        self.adaln = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.adaln[-1].weight)
        nn.init.zeros_(self.adaln[-1].bias)

    def forward(self, x, c):
        s1, k1, g1, s2, k2, g2 = self.adaln(c).chunk(6, dim=-1)
        q = modulate(self.norm1(x), s1, k1)
        b, t, _ = q.shape
        q, k, v = (y.reshape(b, t, self.heads, self.dim_head).transpose(1, 2)
                   for y in self.to_qkv(q).chunk(3, dim=-1))
        h = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                           dropout_p=self.dropout if self.training else 0.)
        x = x + g1 * self.to_out(h.transpose(1, 2).reshape(b, t, -1))
        return x + g2 * self.mlp(modulate(self.norm2(x), s2, k2))


class ARPredictor(nn.Module):
    def __init__(self, latent_dim=192, hidden_dim=192, history_size=5, depth=6, heads=16,
                 dim_head=64, mlp_dim=2048, dropout=.1):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim) if latent_dim != hidden_dim else nn.Identity()
        self.cond_proj = nn.Linear(latent_dim, hidden_dim) if latent_dim != hidden_dim else nn.Identity()
        self.pos = nn.Parameter(torch.randn(1, history_size, latent_dim) * .02)
        self.blocks = nn.ModuleList([ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.output = nn.Linear(hidden_dim, latent_dim) if hidden_dim != latent_dim else nn.Identity()
        self.pred_projector = nn.Sequential(nn.Linear(latent_dim, 2048), nn.BatchNorm1d(2048),
                                            nn.GELU(), nn.Linear(2048, latent_dim))

    def forward(self, z, a_emb):
        x = self.input_proj(z + self.pos[:, :z.shape[1]])
        c = self.cond_proj(a_emb)
        for blk in self.blocks:
            x = blk(x, c)
        x = self.output(self.norm(x))
        b, t, d = x.shape
        return self.pred_projector(x.reshape(b * t, d)).reshape(b, t, d)


class InverseModel(nn.Module):
    """seq_len latents -> the seq_len-1 action contexts between them."""

    def __init__(self, latent_dim=192, action_dim=2, hidden_dim=256, seq_len=4):
        super().__init__()
        self.seq_len, self.action_dim = int(seq_len), int(action_dim)
        self.net = nn.Sequential(nn.Linear(self.seq_len * latent_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, (self.seq_len - 1) * action_dim))

    def forward(self, z_seq):
        return self.net(z_seq.flatten(1)).reshape(len(z_seq), self.seq_len - 1, self.action_dim)


class JEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        d = int(cfg.get('latent_dim', 192))
        self.history_size = int(cfg.get('history_size', 5))
        self.action_context_dim = int(cfg.get('action_context_dim', 2))
        self.encoder = PatchViTEncoder(
            image_size=int(cfg.get('image_size', 128)), patch_size=int(cfg.get('patch_size', 8)),
            in_channels=9, embed_dim=int(cfg.get('vit_embed_dim', 192)),
            depth=int(cfg.get('vit_depth', 4)), heads=int(cfg.get('vit_heads', 3)), latent_dim=d)
        self.action_encoder = ActionEmbedder(self.action_context_dim, d)
        self.predictor = ARPredictor(
            latent_dim=d, hidden_dim=int(cfg.get('predictor_hidden_dim', 192)),
            history_size=self.history_size, depth=int(cfg.get('predictor_depth', 6)),
            heads=int(cfg.get('predictor_heads', 16)), dim_head=int(cfg.get('predictor_dim_head', 64)),
            mlp_dim=int(cfg.get('predictor_mlp_dim', 2048)),
            dropout=float(cfg.get('predictor_dropout', .1)))
        self.inverse_model = (InverseModel(d, self.action_context_dim,
                                           int(cfg.get('inverse_hidden_dim', 256)),
                                           int(cfg.get('inverse_seq_len', 4)))
                              if cfg.get('use_inverse_model', True) else None)

    def encode(self, obs, prev_obs):
        """obs, prev_obs: (b, 3, h, w) floats in [0, 1]."""
        return self.encoder(torch.cat([prev_obs, obs, obs - prev_obs], dim=1))

    def predict(self, z_hist, a_ctx):
        """z_hist (b, t, d), a_ctx (b, t, ctx) -> (b, t, d); position k predicts z_{k+1}."""
        b, t, c = a_ctx.shape
        a = self.action_encoder(a_ctx.reshape(b * t, c)).reshape(b, t, -1)
        return self.predictor(z_hist, a)
