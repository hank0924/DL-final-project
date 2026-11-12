"""Dynamic time warping (Eq. 2): batched distances for segmentation and DBA barycenters."""
import numpy as np
import torch


def minmax(x, axis=-1):
    """Rescales to [0, 1] along `axis` (the normalized space of Sec. 4.1); flat input maps to 0."""
    x = np.asarray(x, dtype=float)
    lo = x.min(axis=axis, keepdims=True)
    rng = x.max(axis=axis, keepdims=True) - lo
    return (x - lo) / np.where(rng > 0, rng, 1.0)


def resample(x, n):
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)


@torch.no_grad()
def dtw_to_centroids(segs, lens, cents, device="cpu", band=None, chunk=40_000):
    """DTW between every segment and every centroid with squared pointwise cost.

    segs: (S, n_max) segments padded past their length; lens: (S,) lengths; cents: (K, m).
    Padding never influences the result because DTW cell (i, j) only depends on earlier cells.
    The cost is divided by (len + m), an upper bound of the warping path length, so that
    segments of different lengths compete fairly in Eq. (1).
    band: optional warping window on relative positions, |i/(n-1) - j/(m-1)| <= band. It keeps
    the global stretch between lengths free while limiting local warping.
    """
    segs_t = torch.as_tensor(np.asarray(segs), dtype=torch.float32, device=device)
    lens_t = torch.as_tensor(np.asarray(lens), dtype=torch.long, device=device)
    c = torch.as_tensor(np.asarray(cents), dtype=torch.float32, device=device)
    K, m = c.shape
    cols = torch.arange(m, device=device).float() / (m - 1)
    out = torch.empty(len(segs_t), K, device=device)
    for a in range(0, len(segs_t), chunk):
        x, n = segs_t[a:a + chunk], lens_t[a:a + chunk]
        res = torch.full((len(x), K), float("inf"), device=device)
        prev = None
        for i in range(int(n.max())):
            cost = (x[:, i, None, None] - c[None]) ** 2
            if band is not None:
                rel = i / (n.float() - 1).clamp(min=1)
                outside = (rel[:, None] - cols[None]).abs() > band + 1e-9
                cost = cost.masked_fill(outside[:, None, :], float("inf"))
            cur = torch.empty_like(cost)
            cur[..., 0] = cost[..., 0] if prev is None else cost[..., 0] + prev[..., 0]
            for j in range(1, m):
                best = cur[..., j - 1]
                if prev is not None:
                    best = torch.minimum(torch.minimum(prev[..., j], prev[..., j - 1]), best)
                cur[..., j] = cost[..., j] + best
            res = torch.where((n == i + 1)[:, None], cur[..., -1], res)
            prev = cur
        out[a:a + chunk] = res / (n[:, None] + m).float()
    return out.cpu().numpy()


def dtw_path(x, y, band=None):
    """Optimal warping path between two 1-D sequences and its accumulated cost."""
    n, m = len(x), len(y)
    cost = (np.asarray(x)[:, None] - np.asarray(y)[None]) ** 2
    if band is not None:
        rel = np.abs(np.arange(n)[:, None] / max(n - 1, 1) - np.arange(m)[None] / max(m - 1, 1))
        cost = np.where(rel > band + 1e-9, np.inf, cost)
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i, j] = cost[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    i, j, path = n, m, []
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        step = np.argmin((D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]))
        if step == 0:
            i, j = i - 1, j - 1
        elif step == 1:
            i -= 1
        else:
            j -= 1
    return path[::-1], D[n, m]


def dba(members, init, n_iter=3, band=None):
    """DTW barycenter averaging (Petitjean et al., 2011) of variable-length members."""
    c = np.array(init, dtype=float)
    for _ in range(n_iter):
        acc, cnt = np.zeros_like(c), np.zeros_like(c)
        for x in members:
            path, _ = dtw_path(x, c, band)
            for i, j in path:
                acc[j] += x[i]
                cnt[j] += 1
        c = np.where(cnt > 0, acc / np.maximum(cnt, 1), c)
    return c
