"""Baselines of Appendix E.2: RCGAN, TimeGAN and unconditional CSDI.

All three are trained on non-overlapping 21-step subsequences of standardized daily log
returns and generate a long series by concatenating independently generated subsequences.
"""
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import DDPM, timestep_embedding


class SegmentModel:
    seq_len = 21

    def __init__(self, iters=5000, batch_size=32, lr=1e-3, device="cpu", seed=0, verbose=True):
        self.iters, self.batch_size, self.lr = iters, batch_size, lr
        self.device, self.seed, self.verbose = device, seed, verbose

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    def fit(self, logp, volume=None, dates=None):
        torch.manual_seed(self.seed)
        logp = np.asarray(logp, dtype=float)
        r = np.diff(logp)
        self.logp0, self.mu, self.sd = logp[0], r.mean(), r.std()
        n = len(r) // self.seq_len
        segs = ((r[:n * self.seq_len] - self.mu) / self.sd).reshape(n, self.seq_len)
        if dates is not None:
            month = np.asarray(dates.month)[1:][::self.seq_len][:n] - 1
        else:
            month = np.zeros(n, dtype=int)
        self.log(f"[{self.name}] {n} training subsequences of length {self.seq_len}")
        self._train(torch.tensor(segs, dtype=torch.float32, device=self.device), month)
        return self

    @torch.no_grad()
    def sample(self, n_paths, T, logp0=None, seed=0):
        torch.manual_seed(seed)
        n_seg = math.ceil(T / self.seq_len)
        z = self._generate(n_paths, n_seg, np.random.default_rng(seed))
        r = z.reshape(n_paths, -1)[:, :T] * self.sd + self.mu
        start = self.logp0 if logp0 is None else logp0
        return start + np.concatenate([np.zeros((n_paths, 1)), np.cumsum(r, axis=1)], axis=1)


def month_features(month):
    a = 2 * np.pi * np.asarray(month) / 12
    return np.stack([np.sin(a), np.cos(a)], axis=-1).astype(np.float32)


class _SeqNet(nn.Module):
    def __init__(self, n_in, hidden, n_out, layers=1, rnn=nn.LSTM, out_act=None):
        super().__init__()
        self.rnn = rnn(n_in, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, n_out)
        self.out_act = out_act

    def forward(self, x):
        y = self.out(self.rnn(x)[0])
        return self.out_act(y) if self.out_act else y


class RCGAN(SegmentModel):
    """Recurrent conditional GAN (Esteban et al., 2017) conditioned on the calendar month."""

    name = "rcgan"

    def __init__(self, hidden=64, noise_dim=5, lr=2e-4, **kw):
        super().__init__(lr=lr, **kw)
        self.hidden, self.noise_dim = hidden, noise_dim

    def _train(self, segs, month):
        dev, L = self.device, self.seq_len
        cond = torch.tensor(month_features(month), device=dev)
        self.G = _SeqNet(self.noise_dim + 2, self.hidden, 1).to(dev)
        self.D = _SeqNet(1 + 2, self.hidden, 1).to(dev)
        opt_g = torch.optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()
        for it in range(self.iters):
            idx = torch.randint(0, len(segs), (self.batch_size,), device=dev)
            x, c = segs[idx][..., None], cond[idx][:, None].expand(-1, L, -1)
            fake = self.G(torch.cat([torch.randn(len(idx), L, self.noise_dim, device=dev), c], -1))
            d_real, d_fake = self.D(torch.cat([x, c], -1)), self.D(torch.cat([fake.detach(), c], -1))
            loss_d = bce(d_real, torch.ones_like(d_real)) + bce(d_fake, torch.zeros_like(d_fake))
            opt_d.zero_grad()
            loss_d.backward()
            opt_d.step()
            d_gen = self.D(torch.cat([fake, c], -1))
            loss_g = bce(d_gen, torch.ones_like(d_gen))
            opt_g.zero_grad()
            loss_g.backward()
            opt_g.step()
            if (it + 1) % max(1, self.iters // 5) == 0:
                self.log(f"[rcgan] iter {it + 1}/{self.iters}: D {loss_d.item():.3f} G {loss_g.item():.3f}")

    def _generate(self, n_paths, n_seg, rng):
        L = self.seq_len
        month = (rng.integers(12, size=(n_paths, 1)) + np.arange(n_seg)[None]) % 12
        c = torch.tensor(month_features(month.reshape(-1)), device=self.device)[:, None].expand(-1, L, -1)
        z = torch.randn(len(c), L, self.noise_dim, device=self.device)
        return self.G(torch.cat([z, c], -1)).squeeze(-1).cpu().numpy()


class TimeGAN(SegmentModel):
    """TimeGAN (Yoon et al., 2019) on one-dimensional returns, following the reference implementation."""

    name = "timegan"

    def __init__(self, hidden=24, layers=3, gamma=1.0, batch_size=128, **kw):
        super().__init__(batch_size=batch_size, **kw)
        self.hidden, self.layers, self.gamma = hidden, layers, gamma

    def _train(self, segs, month):
        dev, h, nl = self.device, self.hidden, self.layers
        x_all = segs[..., None]
        self.lo, self.hi = x_all.min(), x_all.max()
        x_all = (x_all - self.lo) / (self.hi - self.lo + 1e-7)
        sig = torch.sigmoid
        self.E = _SeqNet(1, h, h, nl, nn.GRU, sig).to(dev)
        self.R = _SeqNet(h, h, 1, nl, nn.GRU, sig).to(dev)
        self.Gn = _SeqNet(1, h, h, nl, nn.GRU, sig).to(dev)
        self.S = _SeqNet(h, h, h, max(1, nl - 1), nn.GRU, sig).to(dev)
        self.Dn = _SeqNet(h, h, 1, nl, nn.GRU).to(dev)
        params = lambda *m: [p for n in m for p in n.parameters()]
        opt_e = torch.optim.Adam(params(self.E, self.R), lr=self.lr)
        opt_g = torch.optim.Adam(params(self.Gn, self.S), lr=self.lr)
        opt_d = torch.optim.Adam(params(self.Dn), lr=self.lr)
        bce = nn.BCEWithLogitsLoss()
        batch = lambda: x_all[torch.randint(0, len(x_all), (self.batch_size,), device=dev)]
        noise = lambda x: torch.rand_like(x)

        for it in range(self.iters):  # 1. embedding network
            x = batch()
            loss = 10 * torch.sqrt(F.mse_loss(self.R(self.E(x)), x))
            opt_e.zero_grad()
            loss.backward()
            opt_e.step()
        self.log(f"[timegan] embedding loss {loss.item():.4f}")
        for it in range(self.iters):  # 2. supervised loss only
            H = self.E(batch()).detach()
            loss = F.mse_loss(self.S(H)[:, :-1], H[:, 1:])
            opt_g.zero_grad()
            loss.backward()
            opt_g.step()
        self.log(f"[timegan] supervised loss {loss.item():.4f}")
        for it in range(self.iters):  # 3. joint training
            for _ in range(2):
                x = batch()
                H = self.E(x)
                E_hat = self.Gn(noise(x))
                H_hat = self.S(E_hat)
                X_hat = self.R(H_hat)
                y_fake, y_fake_e = self.Dn(H_hat), self.Dn(E_hat)
                g_u = bce(y_fake, torch.ones_like(y_fake))
                g_u_e = bce(y_fake_e, torch.ones_like(y_fake_e))
                g_s = F.mse_loss(self.S(H.detach())[:, :-1], H.detach()[:, 1:])
                v1 = torch.mean(torch.abs(torch.sqrt(X_hat.var(0) + 1e-6) - torch.sqrt(x.var(0) + 1e-6)))
                v2 = torch.mean(torch.abs(X_hat.mean(0) - x.mean(0)))
                loss_g = g_u + self.gamma * g_u_e + 100 * torch.sqrt(g_s) + 100 * (v1 + v2)
                opt_g.zero_grad()
                loss_g.backward()
                opt_g.step()
                H = self.E(x)
                loss_e = 10 * torch.sqrt(F.mse_loss(self.R(H), x)) + 0.1 * F.mse_loss(self.S(H)[:, :-1], H[:, 1:])
                opt_e.zero_grad()
                loss_e.backward()
                opt_e.step()
            x = batch()
            H = self.E(x).detach()
            E_hat = self.Gn(noise(x)).detach()
            H_hat = self.S(E_hat).detach()
            y_real, y_fake, y_fake_e = self.Dn(H), self.Dn(H_hat), self.Dn(E_hat)
            loss_d = (bce(y_real, torch.ones_like(y_real)) + bce(y_fake, torch.zeros_like(y_fake))
                      + self.gamma * bce(y_fake_e, torch.zeros_like(y_fake_e)))
            if loss_d.item() > 0.15:
                opt_d.zero_grad()
                loss_d.backward()
                opt_d.step()
            if (it + 1) % max(1, self.iters // 5) == 0:
                self.log(f"[timegan] joint iter {it + 1}/{self.iters}: G {loss_g.item():.3f} "
                         f"E {loss_e.item():.3f} D {loss_d.item():.3f}")

    def _generate(self, n_paths, n_seg, rng):
        z = torch.rand(n_paths * n_seg, self.seq_len, 1, device=self.device)
        x = self.R(self.S(self.Gn(z))) * (self.hi - self.lo + 1e-7) + self.lo
        return x.squeeze(-1).cpu().numpy()


class _CSDIBlock(nn.Module):
    def __init__(self, ch, side_dim, emb_dim):
        super().__init__()
        self.diff_proj = nn.Linear(emb_dim, ch)
        self.time_layer = nn.TransformerEncoderLayer(ch, 8, 64, activation="gelu", batch_first=True)
        self.cond_proj = nn.Conv1d(side_dim, 2 * ch, 1)
        self.mid_proj = nn.Conv1d(ch, 2 * ch, 1)
        self.out_proj = nn.Conv1d(ch, 2 * ch, 1)

    def forward(self, x, side, emb):
        y = x + self.diff_proj(emb)[..., None]
        y = self.time_layer(y.transpose(1, 2)).transpose(1, 2)
        gate, filt = (self.mid_proj(y) + self.cond_proj(side)).chunk(2, dim=1)
        res, skip = self.out_proj(torch.sigmoid(gate) * torch.tanh(filt)).chunk(2, dim=1)
        return (x + res) / math.sqrt(2.0), skip


class _CSDINet(nn.Module):
    """CSDI denoiser (Tashiro et al., 2021) for one feature; the feature transformer is skipped as in CSDI when K = 1."""

    def __init__(self, L, ch=64, n_layers=4, emb_dim=128, time_dim=128, feat_dim=16):
        super().__init__()
        self.emb_dim = emb_dim
        self.inp = nn.Conv1d(1, ch, 1)
        self.diff_emb = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim), nn.SiLU())
        pos = timestep_embedding(torch.arange(L), time_dim).T
        self.register_buffer("time_side", pos)
        self.feat_side = nn.Parameter(torch.randn(feat_dim, 1) * 0.02)
        self.blocks = nn.ModuleList(_CSDIBlock(ch, time_dim + feat_dim, emb_dim) for _ in range(n_layers))
        self.skip_proj = nn.Conv1d(ch, ch, 1)
        self.out = nn.Conv1d(ch, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, i):
        B, L = x.shape
        side = torch.cat([self.time_side, self.feat_side.expand(-1, L)], 0)[None].expand(B, -1, -1)
        emb = self.diff_emb(timestep_embedding(i, self.emb_dim))
        h = F.relu(self.inp(x[:, None]))
        skips = 0
        for block in self.blocks:
            h, s = block(h, side, emb)
            skips = skips + s
        h = F.relu(self.skip_proj(skips / math.sqrt(len(self.blocks))))
        return self.out(h).squeeze(1)


class CSDI(SegmentModel):
    """Unconditional CSDI: 50 diffusion steps with the quadratic schedule of the original configuration."""

    name = "csdi"

    def __init__(self, n_steps=50, **kw):
        super().__init__(**kw)
        self.n_steps = n_steps

    def _train(self, segs, month):
        dev = self.device
        betas = torch.linspace(1e-4 ** 0.5, 0.5 ** 0.5, self.n_steps) ** 2
        self.ddpm = DDPM(betas, dev)
        self.net = _CSDINet(self.seq_len).to(dev)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-6)
        for it in range(self.iters):
            x0 = segs[torch.randint(0, len(segs), (self.batch_size,), device=dev)]
            i = torch.randint(0, self.n_steps, (len(x0),), device=dev)
            noise = torch.randn_like(x0)
            loss = F.mse_loss(self.net(self.ddpm.q_sample(x0, i, noise), i), noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if (it + 1) % max(1, self.iters // 5) == 0:
                self.log(f"[csdi] iter {it + 1}/{self.iters}: loss {loss.item():.4f}")

    def _generate(self, n_paths, n_seg, rng):
        self.net.eval()
        x = self.ddpm.sample(self.net, (n_paths * n_seg, self.seq_len), self.device)
        return x.cpu().numpy()
