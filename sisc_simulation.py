"""SISC check on simulated series with known scale-invariant patterns (Appendix B.3).

Reports the per-unit DTW error between learned and ground-truth centroids and the Jaccard
similarity between learned and true segment boundaries.

    python sisc_simulation.py --patterns 1
    python sisc_simulation.py --patterns 4
"""
import argparse

import numpy as np
from scipy.optimize import linear_sum_assignment

from ftsd.dtw import dba, dtw_path, minmax, resample
from ftsd.sisc import SISC
from ftsd.utils import get_device

STANDARD = {
    "head_shoulders": [0, .5, .3, 1, .3, .5, 0],
    "v_shape": [1, .5, 0, .5, 1],
    "rise": [0, .2, .1, .6, .5, 1],
    "double_top": [0, 1, .4, 1, 0],
}


def simulate(n_patterns, T=10_000, l_min=10, l_max=21, noise=0.03, seed=0):
    rng = np.random.default_rng(seed)
    pats = [np.array(v, dtype=float) for v in list(STANDARD.values())[:n_patterns]]
    trans = rng.dirichlet(np.ones(n_patterns), size=n_patterns)
    x, bounds, labels, segs = [0.0], [0], [], []
    k = 0
    while len(x) < T:
        l = int(rng.integers(l_min, l_max + 1))
        shape = resample(pats[k], l + 1) + noise * rng.standard_normal(l + 1)
        seg = x[-1] + rng.uniform(0.5, 3.0) * (shape - shape[0])
        x.extend(seg[1:])
        bounds.append(len(x) - 1)
        labels.append(k)
        segs.append(minmax(seg))
        k = rng.choice(n_patterns, p=trans[k])
    return np.array(x), np.array(bounds), np.array(labels), segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patterns", type=int, default=4, choices=[1, 2, 3, 4])
    ap.add_argument("--band", type=float, help="relative DTW warping window (default: unconstrained)")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--dba_iters", type=int, default=10)
    ap.add_argument("--n_init", type=int, default=5)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    x, bounds, labels, segs = simulate(args.patterns)
    sisc = SISC(args.patterns, max_iters=args.iters, dba_iters=args.dba_iters, n_init=args.n_init,
                band=args.band, device=get_device(args.device), verbose=True).fit(x)
    L = sisc.l_max + 1
    truth = []
    for k in range(args.patterns):
        members = [s for s, lab in zip(segs, labels) if lab == k]
        truth.append(dba(members, resample(members[0], L), 5))
    err = np.array([[dtw_path(minmax(t), minmax(c))[1] / L for c in sisc.centroids_] for t in truth])
    rows, cols = linear_sum_assignment(err)
    seg = sisc.segmentation_
    learned = set(seg.starts.tolist()) | {int(seg.starts[-1] + seg.lengths[-1])}
    true = set(bounds.tolist())
    jac = len(true & learned) / len(true | learned)
    near = sum(min(abs(b - t) for t in true) <= 1 for b in learned) / len(learned)
    # Point-level Jaccard: IoU of every learned segment with its best-overlapping true segment,
    # weighted by length (the "share of correctly segmented points" reading of Appendix B.3).
    true_id = np.searchsorted(bounds, np.arange(bounds[-1]), side="right") - 1
    true_len = np.diff(bounds)
    ious = []
    for s, l in zip(seg.starts, seg.lengths):
        ids = true_id[s:s + l]
        best = np.bincount(ids).argmax()
        inter = np.sum(ids == best)
        ious.append(inter / (l + true_len[best] - inter))
    point_jac = np.average(ious, weights=seg.lengths)
    print(f"patterns: {args.patterns}, band {args.band}, series length {len(x)}, "
          f"true segments {len(bounds) - 1}, learned segments {len(seg.starts)}, SISC iterations {sisc.n_iter_}")
    print(f"per-unit DTW error (matched): {err[rows, cols].mean():.4f}   (paper: 0.009 / 0.01)")
    print(f"point-level Jaccard: {point_jac:.3f}   (paper: 0.938 / 0.784)")
    print(f"exact boundary Jaccard: {jac:.3f}   boundaries within 1 step: {near:.3f}")


if __name__ == "__main__":
    main()
