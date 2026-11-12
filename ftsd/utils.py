import math
import random

import numpy as np
import torch


def get_device(name="auto"):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def timestep_embedding(t, dim):
    """Sinusoidal embedding of integer diffusion steps, shape (B, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device).float() / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


def cosine_betas(n_steps, s=0.008):
    """Cosine noise schedule (Nichol & Dhariwal, 2021); reaches pure noise even with few steps."""
    x = torch.linspace(0, n_steps, n_steps + 1, dtype=torch.float64)
    f = torch.cos((x / n_steps + s) / (1 + s) * math.pi / 2) ** 2
    abar = f / f[0]
    return (1 - abar[1:] / abar[:-1]).clamp(max=0.999).float()


class DDPM:
    """Noise schedule plus forward noising and ancestral sampling (Ho et al., 2020; Appendix C.1)."""

    def __init__(self, betas, device):
        self.n_steps = len(betas)
        self.betas = betas.to(device)
        self.alphas = 1.0 - self.betas
        self.abar = torch.cumprod(self.alphas, 0)
        self.abar_prev = torch.cat([torch.ones(1, device=device), self.abar[:-1]])

    def q_sample(self, x0, i, noise):
        ab = self.abar[i].view(-1, *([1] * (x0.dim() - 1)))
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @torch.no_grad()
    def sample(self, eps_fn, shape, device):
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.n_steps)):
            it = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = eps_fn(x, it)
            beta, alpha, abar = self.betas[i], self.alphas[i], self.abar[i]
            x = (x - beta / (1 - abar).sqrt() * eps) / alpha.sqrt()
            if i > 0:
                var = beta * (1 - self.abar_prev[i]) / (1 - abar)
                x = x + var.sqrt() * torch.randn_like(x)
        return x
