"""FTS-Diffusion (Sec. 4): pattern recognition, pattern generation and pattern evolution."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from .sisc import SISC, segment_features
from .utils import DDPM, cosine_betas, timestep_embedding


class ScalingAE(nn.Module):
    """Scaling autoencoder (Sec. 4.2): variable-length segment <-> fixed-length representation.

    The representation has as many points as the patterns. The decoder is told the target
    length and the relative position of every step, so one representation can be stretched
    to any duration.
    """

    def __init__(self, L, hidden=64, layers=2):
        super().__init__()
        self.L = L
        self.enc = nn.GRU(1, hidden, layers, batch_first=True)
        self.enc_out = nn.Linear(hidden, L)
        self.dec_in = nn.Linear(L + 2, hidden)
        self.dec = nn.GRU(hidden, hidden, layers, batch_first=True)
        self.dec_out = nn.Linear(hidden, 1)

    def encode(self, u, n):
        packed = pack_padded_sequence(u.unsqueeze(-1), n.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.enc(packed)
        return self.enc_out(h[-1])

    def decode(self, z, n):
        steps = torch.arange(self.L, device=z.device).float()
        pos = steps[None] / (n[:, None].float() - 1)
        size = (n[:, None].float() / self.L).expand(-1, self.L)
        inp = torch.cat([z[:, None].expand(-1, self.L, -1), pos[..., None], size[..., None]], dim=-1)
        h, _ = self.dec(F.silu(self.dec_in(inp)))
        return self.dec_out(h).squeeze(-1)


class TCNBlock(nn.Module):
    """Residual temporal convolution block with the step embedding added at its top (Appendix C.2)."""

    def __init__(self, ch, emb_dim, dilation):
        super().__init__()
        self.emb = nn.Linear(emb_dim, ch)
        self.norm1, self.norm2 = nn.GroupNorm(8, ch), nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)

    def forward(self, h, emb):
        x = h + self.emb(emb)[..., None]
        x = self.conv1(F.silu(self.norm1(x)))
        x = self.conv2(F.silu(self.norm2(x)))
        return h + x


class PatternDenoiser(nn.Module):
    """eps_theta(x^i, i, p): six residual TCN blocks, conditioned on the pattern as an input channel."""

    def __init__(self, ch=64, emb_dim=128, n_blocks=6):
        super().__init__()
        self.emb_dim = emb_dim
        self.inp = nn.Conv1d(2, ch, 1)
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        self.blocks = nn.ModuleList(TCNBlock(ch, emb_dim, 2 ** (b % 3)) for b in range(n_blocks))
        self.out = nn.Conv1d(ch, 1, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, i, p):
        emb = self.t_mlp(timestep_embedding(i, self.emb_dim))
        h = self.inp(torch.stack([x, p], dim=1))
        for block in self.blocks:
            h = block(h, emb)
        return self.out(F.silu(h)).squeeze(1)


class PatternEvolutionNet(nn.Module):
    """phi in Eq. (6): current state (p, alpha, beta) -> next pattern logits, magnitude and length."""

    def __init__(self, k, hidden=64):
        super().__init__()
        self.k = k
        self.body = nn.Sequential(nn.Linear(k + 2, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())
        self.head_p = nn.Linear(hidden, k)
        self.head_a = nn.Linear(hidden, 1)
        self.head_l = nn.Linear(hidden, 1)

    def forward(self, p, a, l):
        h = self.body(torch.cat([F.one_hot(p, self.k).float(), a[:, None], l[:, None]], dim=1))
        return self.head_p(h), self.head_a(h).squeeze(1), self.head_l(h).squeeze(1)


class FTSDiffusion:
    """Fits on a 1-D log-price series and samples synthetic log-price paths (Algorithm 2).

    Magnitudes enter the evolution network as standardized log(alpha) and lengths as
    standardized l. Sampling draws the next magnitude and length from Gaussians centred on
    the network outputs with the training residual spread, the likelihood that the MSE terms
    of Eq. (7) correspond to; evo_noise=False keeps the point predictions instead.
    """

    name = "fts"

    def __init__(self, k, l_min=10, l_max=21, sisc_iters=20, n_steps=100, gen_epochs=2000, evo_epochs=1000,
                 batch_size=32, gen_lr=5e-4, evo_lr=4e-4, evo_noise=True, device="cpu", seed=0, verbose=True):
        self.k, self.l_min, self.l_max, self.L = k, l_min, l_max, l_max + 1
        self.sisc_iters, self.n_steps = sisc_iters, n_steps
        self.gen_epochs, self.evo_epochs, self.batch_size = gen_epochs, evo_epochs, batch_size
        self.gen_lr, self.evo_lr, self.evo_noise = gen_lr, evo_lr, evo_noise
        self.device, self.seed, self.verbose = device, seed, verbose

    def log(self, msg):
        if self.verbose:
            print(msg, flush=True)

    # ---------------------------------------------------------------- training
    def fit(self, logp, volume=None, dates=None):
        torch.manual_seed(self.seed)
        logp = np.asarray(logp, dtype=float)
        self.logp0 = logp[0]
        self.log(f"[fts] SISC: K={self.k}, l in [{self.l_min}, {self.l_max}], T={len(logp)}")
        self.sisc = SISC(self.k, self.l_min, self.l_max, self.sisc_iters, seed=self.seed,
                         device=self.device, verbose=self.verbose).fit(logp)
        self.feat = segment_features(logp, self.sisc.segmentation_, self.l_max)
        self.patterns = torch.tensor(self.sisc.centroids_, dtype=torch.float32, device=self.device)
        self.log(f"[fts] {len(self.feat['labels'])} segments, cluster sizes "
                 f"{np.bincount(self.feat['labels'], minlength=self.k).tolist()}")
        self._fit_generation()
        self._fit_evolution()
        return self

    def _fit_generation(self):
        # Joint objective of Eq. (5): reconstruction of the scaling AE plus the simplified DDPM
        # loss. The diffusion network models the residual between the representation and its
        # pattern (Eq. 3). Its target is detached from the encoder so the encoder cannot shrink
        # the residual to make the noise trivially predictable.
        dev, f = self.device, self.feat
        u = torch.tensor(f["u"], dtype=torch.float32, device=dev)
        n = torch.tensor(f["n"], dtype=torch.long, device=dev)
        lab = torch.tensor(f["labels"], dtype=torch.long, device=dev)
        mask = torch.arange(self.L, device=dev)[None] < n[:, None]
        self.ae = ScalingAE(self.L).to(dev)
        self.denoiser = PatternDenoiser().to(dev)
        self.ddpm = DDPM(cosine_betas(self.n_steps), dev)
        self.res_scale = None
        opt = torch.optim.Adam(list(self.ae.parameters()) + list(self.denoiser.parameters()), lr=self.gen_lr)
        for ep in range(self.gen_epochs):
            tot_rec = tot_diff = 0.0
            for b in torch.randperm(len(u), device=dev).split(self.batch_size):
                z = self.ae.encode(u[b], n[b])
                rec = self.ae.decode(z, n[b])
                loss_rec = ((rec - u[b]) ** 2)[mask[b]].mean()
                p = self.patterns[lab[b]]
                r0 = z.detach() - p
                with torch.no_grad():
                    s = r0.std().clamp(min=1e-3)
                    self.res_scale = s if self.res_scale is None else 0.99 * self.res_scale + 0.01 * s
                r0 = r0 / self.res_scale
                i = torch.randint(0, self.n_steps, (len(b),), device=dev)
                noise = torch.randn_like(r0)
                loss_diff = F.mse_loss(self.denoiser(self.ddpm.q_sample(r0, i, noise), i, p), noise)
                opt.zero_grad()
                (loss_rec + loss_diff).backward()
                opt.step()
                tot_rec += loss_rec.item() * len(b)
                tot_diff += loss_diff.item() * len(b)
            if (ep + 1) % max(1, self.gen_epochs // 10) == 0:
                self.log(f"[fts] generation epoch {ep + 1}/{self.gen_epochs}: "
                         f"recon {tot_rec / len(u):.5f}  diffusion {tot_diff / len(u):.4f}")

    def _fit_evolution(self):
        # Eq. (7): cross-entropy on the next pattern plus squared errors on magnitude and length.
        dev, f = self.device, self.feat
        la = np.log(np.maximum(f["alpha"], 1e-8))
        self.a_mu, self.a_sd = la.mean(), la.std() + 1e-8
        self.l_mu, self.l_sd = f["lengths"].mean(), f["lengths"].std() + 1e-8
        p = torch.tensor(f["labels"], dtype=torch.long, device=dev)
        a = torch.tensor((la - self.a_mu) / self.a_sd, dtype=torch.float32, device=dev)
        l = torch.tensor((f["lengths"] - self.l_mu) / self.l_sd, dtype=torch.float32, device=dev)
        x, y = (p[:-1], a[:-1], l[:-1]), (p[1:], a[1:], l[1:])
        self.evo = PatternEvolutionNet(self.k).to(dev)
        opt = torch.optim.Adam(self.evo.parameters(), lr=self.evo_lr)
        for ep in range(self.evo_epochs):
            tot = 0.0
            for b in torch.randperm(len(x[0]), device=dev).split(self.batch_size):
                logits, a_hat, l_hat = self.evo(x[0][b], x[1][b], x[2][b])
                loss = (F.cross_entropy(logits, y[0][b]) + F.mse_loss(a_hat, y[1][b])
                        + F.mse_loss(l_hat, y[2][b]))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(b)
            if (ep + 1) % max(1, self.evo_epochs // 5) == 0:
                self.log(f"[fts] evolution epoch {ep + 1}/{self.evo_epochs}: loss {tot / len(x[0]):.4f}")
        with torch.no_grad():
            logits, a_hat, l_hat = self.evo(*x)
            self.a_resid = (y[1] - a_hat).std().item()
            self.l_resid = (y[2] - l_hat).std().item()
            acc = (logits.argmax(1) == y[0]).float().mean().item()
        self.log(f"[fts] evolution train acc {acc:.3f}, residual sd: log-alpha {self.a_resid:.3f}, "
                 f"length {self.l_resid:.3f} (standardized)")

    # ---------------------------------------------------------------- sampling
    @torch.no_grad()
    def generate_segments(self, labels, n_points):
        """Normalized segments of the given patterns and lengths: denoise, then decode."""
        p = self.patterns[labels]
        r0 = self.ddpm.sample(lambda x, i: self.denoiser(x, i, p), (len(labels), self.L), self.device)
        return self.ae.decode(r0 * self.res_scale + p, n_points)

    @torch.no_grad()
    def _next_state(self, p, a, l):
        logits, a_hat, l_hat = self.evo(p, a, l)
        p_next = torch.distributions.Categorical(logits=logits).sample()
        if self.evo_noise:
            a_hat = a_hat + self.a_resid * torch.randn_like(a_hat)
            l_hat = l_hat + self.l_resid * torch.randn_like(l_hat)
        a_hat = a_hat.clamp(-4, 4)
        steps = torch.round(l_hat * self.l_sd + self.l_mu).clamp(self.l_min, self.l_max).long()
        return p_next, a_hat, (steps.float() - self.l_mu) / self.l_sd, steps

    @torch.no_grad()
    def sample(self, n_paths, T, logp0=None, seed=0):
        """n_paths synthetic log-price paths of T steps (T + 1 points), Algorithm 2."""
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        dev, f = self.device, self.feat
        start = self.logp0 if logp0 is None else logp0
        # Initial segment drawn from the observed data.
        idx = rng.integers(len(f["labels"]), size=n_paths)
        paths = []
        for m in idx:
            u = f["u"][m, :f["n"][m]]
            paths.append(start + f["alpha"][m] * (u - u[0]))
        la = np.log(np.maximum(f["alpha"][idx], 1e-8))
        p = torch.tensor(f["labels"][idx], dtype=torch.long, device=dev)
        a = torch.tensor((la - self.a_mu) / self.a_sd, dtype=torch.float32, device=dev)
        l = torch.tensor((f["lengths"][idx] - self.l_mu) / self.l_sd, dtype=torch.float32, device=dev)
        while min(len(x) for x in paths) < T + 1:
            p, a, l, steps = self._next_state(p, a, l)
            shapes = self.generate_segments(p, steps + 1).cpu().numpy()
            alpha = np.exp(a.cpu().numpy() * self.a_sd + self.a_mu)
            for b in range(n_paths):
                if len(paths[b]) >= T + 1:
                    continue
                w = shapes[b, :steps[b].item() + 1]
                paths[b] = np.concatenate([paths[b], paths[b][-1] + alpha[b] * (w[1:] - w[0])])
        return np.stack([x[:T + 1] for x in paths])
