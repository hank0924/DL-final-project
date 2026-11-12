"""Improved FTS-Diffusion following the group slides (Group_10_Slides.pdf).

Changes with respect to the paper (ftsd/fts_diffusion.py keeps the original):
  1. Long-memory pattern evolution: a causal Transformer reads the past `context` segments,
     each token being (pattern, length, magnitude, volume level) plus a shape embedding.
     A pattern head gives the next pattern; an MDN head samples length and magnitude.
  2. DiT denoiser with adaLN conditioning on diffusion step, pattern, Transformer history and the
     sampled length/magnitude. There is no scaling autoencoder and a single diffusion loss: the
     pattern is stretched to the sampled length and the DiT generates the daily residuals of the
     segment around it (padded to l_max, masked), which are then scaled by the magnitude.
  3. GARCH filtering: GJR-GARCH-X with relative volume as X carries volatility. SISC runs on the
     cumulative standardized residuals and the diffusion model learns z_t = eps_t / sigma_t;
     returns are rebuilt as r_t = mu + sigma_t * z_t with sigma_t simulated forward.
  4. Volume is a second generated channel; it drives the GARCH variance and enters the history.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dtw import minmax, resample
from .garch import GJRGARCHX, log_relative_volume
from .sisc import SISC, segment_features
from .utils import DDPM, cosine_betas, timestep_embedding


class EvolutionTransformer(nn.Module):
    def __init__(self, k, L, d=64, n_layers=2, n_heads=4, context=16, n_mix=5, dropout=0.1):
        super().__init__()
        self.n_mix = n_mix
        self.p_emb = nn.Embedding(k, d)
        self.feat = nn.Linear(3, d)
        self.shape = nn.Linear(L, d)
        self.pos = nn.Parameter(torch.randn(context, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, n_heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head_p = nn.Linear(d, k)
        self.head_mdn = nn.Linear(d, n_mix * 5)

    def forward(self, p, feats, shapes):
        S = p.shape[1]
        h = self.p_emb(p) + self.feat(feats) + self.shape(shapes) + self.pos[:S]
        mask = torch.triu(torch.full((S, S), float("-inf"), device=p.device), diagonal=1)
        return self.norm(self.tf(h, mask=mask))

    def heads(self, h):
        m = self.head_mdn(h).view(*h.shape[:-1], self.n_mix, 5)
        return self.head_p(h), m[..., 0], m[..., 1:3], m[..., 3:5].clamp(-5, 2)


def mdn_nll(pi_logit, mu, log_sd, y):
    z = (y[..., None, :] - mu) / log_sd.exp()
    lp = (-0.5 * z ** 2 - log_sd - 0.5 * math.log(2 * math.pi)).sum(-1)
    return -torch.logsumexp(F.log_softmax(pi_logit, -1) + lp, -1).mean()


def mdn_sample(pi_logit, mu, log_sd):
    comp = torch.distributions.Categorical(logits=pi_logit).sample()
    idx = comp[:, None, None].expand(-1, 1, mu.shape[-1])
    m = mu.gather(1, idx).squeeze(1)
    s = log_sd.gather(1, idx).squeeze(1).exp()
    return m + s * torch.randn_like(m)


class DiTBlock(nn.Module):
    """Transformer block with adaLN-Zero conditioning (Peebles & Xie, 2023)."""

    def __init__(self, d, n_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        nn.init.zeros_(self.ada[1].weight)
        nn.init.zeros_(self.ada[1].bias)

    def forward(self, x, c, pad=None):
        s1, sc1, g1, s2, sc2, g2 = self.ada(c)[:, None].chunk(6, dim=-1)
        h = self.norm1(x) * (1 + sc1) + s1
        x = x + g1 * self.attn(h, h, h, key_padding_mask=pad, need_weights=False)[0]
        h = self.norm2(x) * (1 + sc2) + s2
        return x + g2 * self.mlp(h)


class DiT(nn.Module):
    """eps_theta over daily tokens with channels (residual increment, volume).

    Each token also sees the stretched pattern increment and a validity flag; positions past
    the segment length are padding and are excluded from attention and from the loss.
    """

    def __init__(self, k, L, ch=2, d=64, depth=4, n_heads=4, hist_dim=64, cond_dim=2):
        super().__init__()
        self.inp = nn.Linear(ch + 2, d)
        self.pos = nn.Parameter(torch.randn(L, d) * 0.02)
        self.t_emb = nn.Sequential(nn.Linear(128, d), nn.SiLU(), nn.Linear(d, d))
        self.p_emb = nn.Embedding(k, d)
        self.h_proj = nn.Linear(hist_dim, d)
        self.c_proj = nn.Linear(cond_dim, d)
        self.blocks = nn.ModuleList(DiTBlock(d, n_heads) for _ in range(depth))
        self.final_norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out = nn.Linear(d, ch)
        for m in (self.final_ada[1], self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, i, p_idx, p_inc, valid, hist, cond):
        c = (self.t_emb(timestep_embedding(i, 128)) + self.p_emb(p_idx)
             + self.h_proj(hist) + self.c_proj(cond))
        h = self.inp(torch.cat([x, p_inc[..., None], valid[..., None].float()], dim=-1)) + self.pos
        for block in self.blocks:
            h = block(h, c, pad=~valid)
        shift, scale = self.final_ada(c)[:, None].chunk(2, dim=-1)
        return self.out(self.final_norm(h) * (1 + scale) + shift)


class FTSPlus:
    name = "fts_plus"

    def __init__(self, k, l_min=10, l_max=21, sisc_iters=20, n_steps=100, context=16, n_mix=5,
                 dit_epochs=2000, evo_steps=4000, batch_size=32, lr=5e-4, evo_lr=4e-4,
                 use_garch=True, use_volume=True, device="cpu", seed=0, verbose=True):
        self.k, self.l_min, self.l_max, self.L = k, l_min, l_max, l_max + 1
        self.sisc_iters, self.n_steps, self.context, self.n_mix = sisc_iters, n_steps, context, n_mix
        self.dit_epochs, self.evo_steps, self.batch_size = dit_epochs, evo_steps, batch_size
        self.lr, self.evo_lr = lr, evo_lr
        self.use_garch, self.use_volume = use_garch, use_volume
        self.device, self.seed, self.verbose = device, seed, verbose

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ---------------------------------------------------------------- training
    def fit(self, logp, volume=None, dates=None):
        torch.manual_seed(self.seed)
        logp = np.asarray(logp, dtype=float)
        self.logp_train, self.logp0 = logp, logp[0]
        r = np.diff(logp)
        has_volume = volume is not None and np.mean(np.asarray(volume) > 0) > 0.9
        self.use_volume = self.use_volume and has_volume
        self.lrv = log_relative_volume(volume)[1:] if has_volume else np.zeros_like(r)

        if self.use_garch:
            self.garch = GJRGARCHX(self.use_volume).fit(r, self.lrv)
        else:
            self.garch = GJRGARCHX(False).fit_constant(r)
        self.sigma, z = self.garch.filter(r, self.lrv)
        self.z_train = z
        self.log(f"[fts_plus] GARCH: {self.garch.summary()}")
        Z = np.concatenate([[0.0], np.cumsum(z)])

        self.log(f"[fts_plus] SISC on standardized residual path: K={self.k}, T={len(Z)}")
        self.sisc = SISC(self.k, self.l_min, self.l_max, self.sisc_iters, seed=self.seed,
                         device=self.device, verbose=self.verbose).fit(Z)
        f = segment_features(Z, self.sisc.segmentation_, self.l_max)
        M, Lz = len(f["labels"]), self.l_max
        # Pattern increments stretched to every admissible length: stretch by length (slides).
        self.p_inc_table = np.zeros((self.k, self.l_max - self.l_min + 1, Lz))
        for k, c in enumerate(self.sisc.centroids_):
            for l in range(self.l_min, self.l_max + 1):
                self.p_inc_table[k, l - self.l_min, :l] = np.diff(resample(c, l + 1))
        f["shape"] = np.stack([resample(u[:n], self.L) for u, n in zip(f["u"], f["n"])])
        f["inc"], f["vol"], f["valid"] = np.zeros((M, Lz)), np.zeros((M, Lz)), np.zeros((M, Lz), dtype=bool)
        for m, (s, l) in enumerate(zip(f["starts"], f["lengths"])):
            f["inc"][m, :l] = z[s:s + l] / max(f["alpha"][m], 1e-8)
            f["vol"][m, :l] = self.lrv[s:s + l]
            f["valid"][m, :l] = True
        f["p_inc"] = self.p_inc_table[f["labels"], f["lengths"] - self.l_min]
        self.feat = f
        self.patterns = torch.tensor(self.sisc.centroids_, dtype=torch.float32, device=self.device)
        self.log(f"[fts_plus] {M} segments, cluster sizes {np.bincount(f['labels'], minlength=self.k).tolist()}")

        lm = np.log(np.maximum(f["alpha"], 1e-8))
        self.l_mu, self.l_sd = f["lengths"].mean(), f["lengths"].std() + 1e-8
        self.m_mu, self.m_sd = lm.mean(), lm.std() + 1e-8
        vol_valid = f["vol"][f["valid"]]
        vm = (f["vol"] * f["valid"]).sum(1) / f["valid"].sum(1)
        self.vm_mu, self.vm_sd = vm.mean(), vm.std() + 1e-8
        self.v_mu, self.v_sd = vol_valid.mean(), vol_valid.std() + 1e-8
        self.v_lo, self.v_hi = self.lrv.min(), self.lrv.max()
        tok = np.stack([(f["lengths"] - self.l_mu) / self.l_sd, (lm - self.m_mu) / self.m_sd,
                        (vm - self.vm_mu) / self.vm_sd], axis=1)
        t = lambda a, dt=torch.float32: torch.tensor(a, dtype=dt, device=self.device)
        self.tok_p, self.tok_f, self.tok_s = t(f["labels"], torch.long), t(tok), t(f["shape"])
        self._fit_evolution()
        self._fit_dit()
        return self

    def _fit_evolution(self):
        dev = self.device
        M = len(self.tok_p)
        W = min(self.context + 1, M)
        self.evo = EvolutionTransformer(self.k, self.L, context=self.context, n_mix=self.n_mix).to(dev)
        opt = torch.optim.AdamW(self.evo.parameters(), lr=self.evo_lr, weight_decay=1e-2)
        offs = torch.arange(W, device=dev)
        for step in range(self.evo_steps):
            idx = torch.randint(0, M - W + 1, (self.batch_size,), device=dev)[:, None] + offs
            p, fe, sh = self.tok_p[idx], self.tok_f[idx], self.tok_s[idx]
            logits, pi, mu, lsd = self.evo.heads(self.evo(p[:, :-1], fe[:, :-1], sh[:, :-1]))
            loss = (F.cross_entropy(logits.reshape(-1, self.k), p[:, 1:].reshape(-1))
                    + mdn_nll(pi, mu, lsd, fe[:, 1:, :2]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (step + 1) % max(1, self.evo_steps // 5) == 0:
                self.log(f"[fts_plus] evolution step {step + 1}/{self.evo_steps}: loss {loss.item():.4f}")
        self.evo.eval()
        # History seen by the denoiser for segment j: Transformer state after segments < j.
        hist = torch.zeros(M, self.evo.norm.normalized_shape[0], device=dev)
        with torch.no_grad():
            for j in range(1, M):
                a = max(0, j - self.context)
                hist[j] = self.evo(self.tok_p[None, a:j], self.tok_f[None, a:j], self.tok_s[None, a:j])[0, -1]
        self.hist = hist

    def _fit_dit(self):
        # Single diffusion loss on the daily standardized residuals of each segment (divided by
        # the segment magnitude), relative to the stretched pattern increments as in Eq. (3).
        dev, f = self.device, self.feat
        t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        lab = self.tok_p
        p_inc, valid = t(f["p_inc"]), torch.tensor(f["valid"], device=dev)
        res = t(f["inc"]) - p_inc
        self.s_res = res[valid].std().item() + 1e-6
        vol = (t(f["vol"]) - self.v_mu) / self.v_sd
        vol = vol if self.use_volume else torch.zeros_like(vol)
        x0 = torch.stack([res / self.s_res, vol], dim=-1) * valid[..., None]
        cond = self.tok_f[:, :2]
        self.dit = DiT(self.k, self.l_max).to(dev)
        self.ddpm = DDPM(cosine_betas(self.n_steps), dev)
        opt = torch.optim.Adam(self.dit.parameters(), lr=self.lr)
        M = len(x0)
        for ep in range(self.dit_epochs):
            tot = 0.0
            for b in torch.randperm(M, device=dev).split(self.batch_size):
                i = torch.randint(0, self.n_steps, (len(b),), device=dev)
                noise = torch.randn_like(x0[b])
                pred = self.dit(self.ddpm.q_sample(x0[b], i, noise), i, lab[b], p_inc[b], valid[b],
                                self.hist[b], cond[b])
                w = valid[b][..., None].float()
                loss = (((pred - noise) ** 2) * w).sum() / (w.sum() * 2)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            if (ep + 1) % max(1, self.dit_epochs // 10) == 0:
                self.log(f"[fts_plus] DiT epoch {ep + 1}/{self.dit_epochs}: diffusion {tot / M:.4f}")
        self.dit.eval()

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def sample(self, n_paths, T, logp0=None, seed=0):
        """Generates residual and volume streams segment by segment, rescales the residuals to
        the in-sample residual scale, then rebuilds returns through the GARCH recursion."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        dev, f, g = self.device, self.feat, self.garch
        M = len(self.tok_p)
        W = min(self.context, M)
        start = self.logp0 if logp0 is None else logp0
        # Prompt: W consecutive observed segments; the last one opens the synthetic path.
        js = rng.integers(W - 1, M, size=n_paths)
        ctx = torch.tensor(js[:, None] - W + 1 + np.arange(W), dtype=torch.long, device=dev)
        ctx_p, ctx_f, ctx_s = self.tok_p[ctx], self.tok_f[ctx], self.tok_s[ctx]
        heads = [f["lengths"][j] for j in js]
        need = [T - l for l in heads]
        zs, vs = [[] for _ in range(n_paths)], [[] for _ in range(n_paths)]
        have = np.zeros(n_paths, dtype=int)
        while (have < need).any():
            h = self.evo(ctx_p, ctx_f, ctx_s)[:, -1]
            logits, pi, mdn_mu, lsd = self.evo.heads(h)
            p = torch.distributions.Categorical(logits=logits).sample()
            y = mdn_sample(pi, mdn_mu, lsd).clamp(-4, 4)
            steps = torch.round(y[:, 0] * self.l_sd + self.l_mu).clamp(self.l_min, self.l_max).long()
            cond = torch.stack([(steps.float() - self.l_mu) / self.l_sd, y[:, 1]], dim=1)
            steps_np = steps.cpu().numpy()
            p_inc_np = self.p_inc_table[p.cpu().numpy(), steps_np - self.l_min]
            p_inc = torch.tensor(p_inc_np, dtype=torch.float32, device=dev)
            valid = torch.arange(self.l_max, device=dev)[None] < steps[:, None]
            x0 = self.ddpm.sample(lambda x, i: self.dit(x, i, p, p_inc, valid, h, cond),
                                  (n_paths, self.l_max, 2), dev).clamp(-6, 6).cpu().numpy()
            inc = x0[..., 0] * self.s_res + p_inc_np
            vol = np.clip(x0[..., 1] * self.v_sd + self.v_mu, self.v_lo, self.v_hi)
            mag = np.exp(y[:, 1].cpu().numpy() * self.m_sd + self.m_mu)
            vol_level, shape = np.empty(n_paths), np.empty((n_paths, self.L))
            for b in range(n_paths):
                l = steps_np[b]
                z = np.clip(mag[b] * inc[b, :l], -8, 8)
                v = vol[b, :l]
                vol_level[b] = v.mean()
                shape[b] = minmax(resample(np.concatenate([[0.0], np.cumsum(z)]), self.L))
                if have[b] < need[b]:
                    zs[b].append(z)
                    vs[b].append(v)
                    have[b] += l
            new_f = torch.stack([cond[:, 0], cond[:, 1],
                                 torch.tensor((vol_level - self.vm_mu) / self.vm_sd, dtype=torch.float32, device=dev)],
                                dim=1)
            ctx_p = torch.cat([ctx_p[:, 1:], p[:, None]], dim=1)
            ctx_f = torch.cat([ctx_f[:, 1:], new_f[:, None]], dim=1)
            ctx_s = torch.cat([ctx_s[:, 1:], torch.tensor(shape, dtype=torch.float32, device=dev)[:, None]], dim=1)
        zs = [np.concatenate(z)[:n] for z, n in zip(zs, need)]
        vs = [np.concatenate(v)[:n] for v, n in zip(vs, need)]
        z_all = np.concatenate(zs)
        kurt = lambda a: ((a - a.mean()) ** 4).mean() / a.var() ** 2 - 3
        self.log(f"[fts_plus] synthetic standardized residuals before calibration: std {z_all.std():.3f}, "
                 f"kurtosis {kurt(z_all):.2f} (in-sample: {self.z_train.std():.3f}, {kurt(self.z_train):.2f})")
        scale = self.z_train.std() / z_all.std()

        mu = g.params[0]
        paths = []
        for b, j in enumerate(js):
            s, l = f["starts"][j], f["lengths"][j]
            head = self.logp_train[s:s + l + 1] - self.logp_train[s]
            last = s + l - 1
            s2 = (100 * self.sigma[last]) ** 2
            eps = 100 * (self.logp_train[last + 1] - self.logp_train[last]) - mu
            xv = math.exp(self.lrv[last])
            r = np.empty(need[b])
            z_b = np.clip(zs[b] * scale, self.z_train.min(), self.z_train.max())
            for t, (z, v) in enumerate(zip(z_b, vs[b])):
                s2 = g.step(s2, eps, xv)
                eps = math.sqrt(s2) * z
                r[t] = (mu + eps) / 100
                xv = math.exp(v)
            paths.append(start + np.concatenate([head, head[-1] + np.cumsum(r)]))
        return np.stack([x[:T + 1] for x in paths])
