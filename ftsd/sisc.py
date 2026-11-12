"""Scale-Invariant Subsequence Clustering, SISC (Sec. 4.1, Algorithm 1)."""
from dataclasses import dataclass

import numpy as np

from .dtw import dba, dtw_to_centroids, minmax, resample


@dataclass
class Segmentation:
    starts: np.ndarray   # first point of each segment
    lengths: np.ndarray  # number of steps l; a segment spans points start..start+l
    labels: np.ndarray   # nearest pattern
    dists: np.ndarray    # DTW distance to that pattern

    @property
    def inertia(self):
        return float(self.dists.sum())


def window_bank(x, l_min, l_max):
    """All normalized subsequences x[t:t+l+1] for l in [l_min, l_max], padded to l_max+1 points."""
    segs, lens, ts, ls = [], [], [], []
    for l in range(l_min, l_max + 1):
        n = len(x) - l
        if n <= 0:
            continue
        w = minmax(x[np.arange(n)[:, None] + np.arange(l + 1)[None]])
        pad = np.zeros((n, l_max + 1))
        pad[:, :l + 1] = w
        segs.append(pad)
        lens.append(np.full(n, l + 1))
        ts.append(np.arange(n))
        ls.append(np.full(n, l))
    return np.concatenate(segs), np.concatenate(lens), np.concatenate(ts), np.concatenate(ls)


def segment_features(x, seg, l_max):
    """Normalized shape u, magnitude alpha and length of every segment.

    A segment is recovered exactly as x[s:s+l+1] = x[s] + alpha * (u - u[0]).
    """
    M = len(seg.starts)
    u = np.zeros((M, l_max + 1))
    alpha = np.empty(M)
    for m, (s, l) in enumerate(zip(seg.starts, seg.lengths)):
        w = x[s:s + l + 1]
        alpha[m] = w.max() - w.min()
        u[m, :l + 1] = minmax(w)
    return {"u": u, "n": seg.lengths + 1, "alpha": alpha, "lengths": seg.lengths.copy(),
            "labels": seg.labels.copy(), "starts": seg.starts.copy()}


class SISC:
    """Greedy variable-length segmentation plus K-means style clustering under DTW.

    Patterns (centroids) have l_max + 1 points. Each iteration segments the whole series
    greedily with the current patterns (Eq. 1) and then moves every pattern to the DTW
    barycenter of its members.
    """

    def __init__(self, k, l_min=10, l_max=21, max_iters=20, dba_iters=10, n_init=5, band=None, seed=0,
                 device="cpu", verbose=False):
        self.k, self.l_min, self.l_max, self.band = k, l_min, l_max, band
        self.max_iters, self.dba_iters, self.n_init = max_iters, dba_iters, n_init
        self.seed, self.device, self.verbose = seed, device, verbose

    def fit(self, x):
        """Runs SISC from n_init initializations and keeps the one with the lowest inertia."""
        x = np.asarray(x, dtype=float)
        bank = window_bank(x, self.l_min, self.l_max)
        best = None
        for run in range(self.n_init):
            cents, seg, n_iter = self._fit_once(x, bank, np.random.default_rng(self.seed + run))
            if self.verbose:
                print(f"  SISC init {run + 1}/{self.n_init}: {n_iter} iterations, {len(seg.starts)} segments, "
                      f"inertia {seg.inertia:.4f}", flush=True)
            if best is None or seg.inertia < best[1].inertia:
                best = (cents, seg, n_iter)
        self.centroids_, self.segmentation_, self.n_iter_ = best
        return self

    def _fit_once(self, x, bank, rng):
        cents = self._init_centroids(bank, rng)
        prev = None
        for it in range(self.max_iters):
            seg = self._segment(x, bank, cents)
            key = (seg.starts.tobytes(), seg.labels.tobytes())
            if key == prev:
                break
            prev = key
            cents = self._update(x, seg, cents)
        else:
            seg = self._segment(x, bank, cents)
        return cents, seg, it + 1

    def segment(self, x):
        """Segments a new series with the learned patterns."""
        x = np.asarray(x, dtype=float)
        return self._segment(x, window_bank(x, self.l_min, self.l_max), self.centroids_)

    def _init_centroids(self, bank, rng):
        # Lines 1-10 of Algorithm 1: first pattern at random among l_max windows, the rest drawn
        # with probability proportional to the DTW distance to the closest chosen pattern.
        segs, lens, _, ls = bank
        cand, cand_lens = segs[ls == self.l_max], lens[ls == self.l_max]
        chosen = [cand[rng.integers(len(cand))]]
        dist = lambda c: dtw_to_centroids(cand, cand_lens, c[None], self.device, self.band)[:, 0]
        d = dist(chosen[-1])
        while len(chosen) < self.k:
            prob = d / d.sum() if d.sum() > 0 else None
            chosen.append(cand[rng.choice(len(cand), p=prob)])
            d = np.minimum(d, dist(chosen[-1]))
        return np.stack(chosen)

    def _segment(self, x, bank, cents):
        # Lines 13-19 of Algorithm 1 with Eq. (1): at each position keep the length whose
        # subsequence is closest to any pattern, then jump to the end of that subsequence.
        segs, lens, ts, ls = bank
        D = dtw_to_centroids(segs, lens, cents, self.device, self.band)
        n_l = self.l_max - self.l_min + 1
        dist = np.full((len(x), n_l), np.inf)
        lab = np.zeros((len(x), n_l), dtype=int)
        dist[ts, ls - self.l_min] = D.min(1)
        lab[ts, ls - self.l_min] = D.argmin(1)
        starts, lengths, labels, dists = [], [], [], []
        t = 0
        while t + self.l_min <= len(x) - 1:
            j = int(np.argmin(dist[t]))
            starts.append(t)
            lengths.append(j + self.l_min)
            labels.append(lab[t, j])
            dists.append(dist[t, j])
            t += j + self.l_min
        return Segmentation(np.array(starts), np.array(lengths), np.array(labels), np.array(dists))

    def _update(self, x, seg, cents):
        new = cents.copy()
        for k in range(self.k):
            idx = np.flatnonzero(seg.labels == k)
            if len(idx) == 0:
                # Re-seed an empty cluster with the worst-fitting segment.
                m = int(np.argmax(seg.dists))
                s, l = seg.starts[m], seg.lengths[m]
                new[k] = resample(minmax(x[s:s + l + 1]), self.l_max + 1)
                continue
            members = [minmax(x[seg.starts[m]:seg.starts[m] + seg.lengths[m] + 1]) for m in idx]
            new[k] = dba(members, cents[k], self.dba_iters, self.band)
        return new


def elbow(x, ks, **kwargs):
    """Inertia for each K, used with the elbow method to choose K (Appendix B.1)."""
    return {k: SISC(k, **kwargs).fit(x).segmentation_.inertia for k in ks}
