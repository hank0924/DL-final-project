"""Downstream prediction experiments of Sec. 5.3 (Fig. 6, Fig. 10).

An LSTM predicts the price `horizon` days ahead from the previous 64 prices; MAPE is measured
on the real test period.
  TMTR: a fixed-size training set mixes real and synthetic windows in given proportions.
  TATR: all real training windows plus a growing number of synthetic windows.

    python downstream.py --asset GOOG --setting both --horizon 1
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from evaluate import LABELS, available, md_table
from ftsd.data import load_split, results_dir
from ftsd.utils import get_device, set_seed


def windows(prices, lookback, horizon, first_target=None):
    """Inputs are the last `lookback` prices relative to the latest one; target is the relative change."""
    p = np.asarray(prices, dtype=float)
    t = np.arange(max(lookback, first_target or lookback), len(p) - horizon + 1)
    X = p[t[:, None] - lookback + np.arange(lookback)] / p[t - 1, None] - 1
    y = p[t + horizon - 1] / p[t - 1] - 1
    return X.astype(np.float32), y.astype(np.float32)


class LSTMForecaster(nn.Module):
    def __init__(self, hidden=64, layers=2):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
        self.out = nn.Linear(hidden, 1)

    def forward(self, x):
        return self.out(self.lstm(x[..., None])[0][:, -1]).squeeze(-1)


def fit_mape(Xtr, ytr, Xte, yte, args, seed):
    """Trains an LSTM and returns the test MAPE of the predicted price level."""
    set_seed(seed)
    dev = args.device
    model = LSTMForecaster().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xtr_t, ytr_t = torch.tensor(Xtr * 100, device=dev), torch.tensor(ytr * 100, device=dev)
    for _ in range(args.epochs):
        for b in torch.randperm(len(Xtr_t), device=dev).split(args.batch_size):
            loss = nn.functional.mse_loss(model(Xtr_t[b]), ytr_t[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = model(torch.tensor(Xte * 100, device=dev)).cpu().numpy() / 100
    return float(np.mean(np.abs(pred - yte) / (1 + yte)))


def synthetic_windows(logp, lookback, horizon):
    parts = [windows(np.exp(s), lookback, horizon) for s in logp]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="GOOG")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--setting", choices=["tmtr", "tatr", "both"], default="both")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--lookback", type=int, default=64)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--proportions", type=float, nargs="+", default=[0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--aug_sizes", type=int, nargs="+", default=[0, 2520, 5040, 10080, 15120, 20160, 25200],
                    help="synthetic windows added in TATR (25200 trading days = 100 years)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    args.device = get_device(args.device)

    out = results_dir(args.asset)
    df, train, _ = load_split(args.asset)
    prices = df.close.to_numpy()
    n_train = len(train)
    Xr, yr = windows(prices[:n_train], args.lookback, args.horizon)
    Xte, yte = windows(prices, args.lookback, args.horizon, first_target=n_train)
    naive = float(np.mean(np.abs(yte) / (1 + yte)))
    models = args.models or available(out)
    syn = {m: synthetic_windows(np.load(out / m / "samples.npz")["logp"], args.lookback, args.horizon)
           for m in models}
    print(f"{args.asset}: {len(Xr)} real train windows, {len(Xte)} test windows, horizon {args.horizon}; "
          f"naive MAPE {naive:.4f}", flush=True)

    rows = []
    rng = np.random.default_rng(0)
    if args.setting in ("tmtr", "both"):
        N = len(Xr)
        for m in models:
            Xs, ys = syn[m]
            for q in args.proportions:
                n_syn = int(round(q * N))
                for run in range(args.runs):
                    ir = rng.choice(len(Xr), N - n_syn, replace=False)
                    i_s = rng.choice(len(Xs), n_syn, replace=n_syn > len(Xs))
                    X = np.concatenate([Xr[ir], Xs[i_s]])
                    y = np.concatenate([yr[ir], ys[i_s]])
                    mape = fit_mape(X, y, Xte, yte, args, seed=run)
                    rows.append({"setting": "tmtr", "model": m, "x": q * 100, "run": run, "mape": mape})
                print(f"[tmtr] {m} syn {q:.0%}: MAPE {np.mean([r['mape'] for r in rows[-args.runs:]]):.5f}",
                      flush=True)
    if args.setting in ("tatr", "both"):
        for m in models:
            Xs, ys = syn[m]
            for size in args.aug_sizes:
                for run in range(args.runs):
                    i_s = rng.choice(len(Xs), size, replace=size > len(Xs))
                    X = np.concatenate([Xr, Xs[i_s]])
                    y = np.concatenate([yr, ys[i_s]])
                    mape = fit_mape(X, y, Xte, yte, args, seed=run)
                    rows.append({"setting": "tatr", "model": m, "x": size, "run": run, "mape": mape})
                print(f"[tatr] {m} +{size}: MAPE {np.mean([r['mape'] for r in rows[-args.runs:]]):.5f}",
                      flush=True)

    res = pd.DataFrame(rows)
    tag = f"h{args.horizon}"
    res.to_csv(out / f"downstream_{tag}.csv", index=False)
    summary = res.groupby(["setting", "model", "x"]).mape.agg(["mean", "std"]).reset_index()
    lines = [f"# {args.asset}: downstream MAPE, horizon {args.horizon} (naive {naive:.4f})", ""]
    for setting, g in summary.groupby("setting"):
        piv = g.pivot(index="model", columns="x", values="mean")
        lines += [f"## {setting.upper()}", "", md_table(piv, 5), ""]
        if setting == "tatr":
            base = piv.iloc[:, 0]
            lines += ["Relative MAPE reduction at the largest augmentation: "
                      + ", ".join(f"{LABELS[m]} {100 * (1 - piv.loc[m].iloc[-1] / base[m]):.1f}%"
                                  for m in piv.index), ""]
    (out / f"downstream_{tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    plot(res, naive, out / f"fig6_{tag}.png")


def plot(res, naive, path):
    settings = [s for s in ("tmtr", "tatr") if s in set(res.setting)]
    models = list(dict.fromkeys(res.model))
    fig, axes = plt.subplots(1, len(settings) * 1, figsize=(6 * len(settings), 4), squeeze=False)
    for ax, s in zip(axes[0], settings):
        g = res[res.setting == s].groupby(["model", "x"]).mape
        mean, sd, n = g.mean(), g.std().fillna(0), g.count()
        for m in models:
            x = mean[m].index.to_numpy()
            ci = 1.96 * sd[m].to_numpy() / np.sqrt(n[m].to_numpy())
            ax.plot(x, mean[m].to_numpy(), marker="o", label=LABELS[m])
            ax.fill_between(x, mean[m].to_numpy() - ci, mean[m].to_numpy() + ci, alpha=0.2)
        ax.axhline(naive, color="gray", ls=":", label="naive")
        ax.set_xlabel("Syn. proportion (%)" if s == "tmtr" else "Augmented size (synthetic windows)")
        ax.set_ylabel("MAPE")
        ax.set_title(s.upper())
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
