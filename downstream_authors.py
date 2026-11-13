"""Downstream experiments following the authors' supplementary code of the ICLR 2024 submission
(experiments/tatr.py, tmtr.py, predictor_lstm.py and utils_downstream.py).

What this takes from that code, unlike downstream.py:
  - Only the test period is used. Its first `init_days` (252 * 5) days are the real training data and
    the remaining days are the evaluation set.
  - Prices are min-max scaled to [-1, 1]. A rolling window of 64 days holds 64 - h inputs and h targets.
  - The predictor is an LSTM with 2 layers of 32 units, trained with MAE loss and Adam (lr 1e-2) for
    100 full-batch epochs.
  - TATR appends 252 synthetic days at a time, up to 100 times (100 years), to the same training set.
    TMTR mixes a real and a synthetic stretch into a 1260-day training series at 0%, 10%, ..., 100%;
    its scaler is fitted on the evaluation series, as in the authors' code.
  - MAPE is sklearn's, on de-scaled prices; runs are summarised by the mean with the lowest and highest
    2.5% removed.
Where this repo cannot follow the code:
  - Synthetic stretches are cut from the paths saved by train.py instead of being generated afresh, and
    re-levelled to start at the first test price (the authors start each one from the first test segment).
  - GOOG and ZC=F have fewer than 1260 + 64 test days, so the authors' code leaves them no evaluation data;
    there the first half of the test period is the real training data.
  - The authors run 100 runs at every one of the 100 augmentation steps; `--runs` and `--aug_counts`
    default to fewer to fit a Colab session.

    python downstream_authors.py --asset SP500 --horizon 1
"""
import argparse
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from downstream import plot
from evaluate import LABELS, available, md_table
from ftsd.data import load_split, results_dir
from ftsd.utils import get_device, set_seed


def scale(x, sc):
    return 2 * (np.asarray(x, dtype=float) - sc[0]) / (sc[1] - sc[0]) - 1


def unscale(s, sc):
    return (np.asarray(s, dtype=float) + 1) / 2 * (sc[1] - sc[0]) + sc[0]


def rolling(prices, window, sc):
    """Rolling windows of scaled prices (Timeseries2Dataset_Downstream)."""
    if len(prices) < window:
        return np.empty((0, window), dtype=np.float32)
    return np.lib.stride_tricks.sliding_window_view(scale(prices, sc), window).astype(np.float32)


class LSTMPredictor(nn.Module):
    def __init__(self, hidden=32, layers=2, ahead=1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
        self.fc = nn.Linear(hidden, ahead)

    def forward(self, x):
        return self.fc(self.lstm(x[..., None])[0][:, -1])


def fit_mape(data, test, sc, args, seed):
    """separate_train_lstm_predictor + test_on_real: full-batch training, MAPE on de-scaled prices."""
    set_seed(seed)
    dev, h = args.device, args.horizon
    model = LSTMPredictor(args.hidden, 2, h).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    D = torch.tensor(data, device=dev)
    X, y = D[:, :-h], D[:, -h:]
    for _ in range(args.epochs):
        loss = nn.functional.l1_loss(model(X), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = model(torch.tensor(test[:, :-h], device=dev)).cpu().numpy()
    real = unscale(test[:, -h:], sc)
    return float(np.mean(np.abs(real - unscale(pred, sc)) / np.abs(real)))


class SyntheticStretches:
    """Random stretches of a model's saved synthetic paths, started at a given price."""

    def __init__(self, logp, start_price, rng):
        self.r = np.diff(np.asarray(logp, dtype=float), axis=1)
        self.start, self.rng = start_price, rng

    def __call__(self, n):
        i = self.rng.integers(len(self.r))
        j = self.rng.integers(0, self.r.shape[1] - n + 2)
        return self.start * np.exp(np.concatenate([[0.0], np.cumsum(self.r[i, j:j + n - 1])]))


def contiguous(series, length, rng):
    """sample_tmtr_timeseries: a random contiguous stretch of the given length."""
    if length == len(series):
        return series
    s = rng.integers(0, len(series) - length)
    return series[s:s + length]


def trimmed(x):
    """summarize_results: mean without the lowest and highest 2.5% of runs."""
    x = np.sort(np.asarray(x))
    p = math.ceil(len(x) * 0.025)
    return x[p:-p].mean() if len(x) > 2 * p else x.mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="SP500")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--setting", choices=["tmtr", "tatr", "both"], default="both")
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--window", type=int, default=64, help="window length including the h targets")
    ap.add_argument("--init_days", type=int, default=252 * 5)
    ap.add_argument("--aug_length", type=int, default=252)
    ap.add_argument("--aug_counts", type=int, nargs="+", default=list(range(0, 101, 10)),
                    help="numbers of 252-day synthetic stretches at which TATR is evaluated")
    ap.add_argument("--proportions", type=int, nargs="+", default=list(range(0, 101, 10)))
    ap.add_argument("--runs", type=int, default=20, help="the authors use 100")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    args.device = get_device(args.device)

    out = results_dir(args.asset)
    _, _, test = load_split(args.asset)
    prices = test.close.to_numpy()
    n_init = args.init_days
    if len(prices) - n_init < args.window + 50:
        n_init = len(prices) // 2
        print(f"warning: {len(prices)} test days leave no evaluation data after {args.init_days} days; "
              f"using the first {n_init} days for training", flush=True)
    real, held = prices[:n_init], prices[n_init:]
    models = args.models or available(out)
    paths = {m: np.load(out / m / "samples.npz")["logp"] for m in models}
    h = args.horizon

    rows = []
    rng = np.random.default_rng(0)
    if args.setting in ("tatr", "both"):
        sc = (real.min(), real.max())
        base, test_set = rolling(real, args.window, sc), rolling(held, args.window, sc)
        last = unscale(test_set[:, -h - 1:-h], sc)
        naive = float(np.mean(np.abs(unscale(test_set[:, -h:], sc) - last) / unscale(test_set[:, -h:], sc)))
        print(f"{args.asset} TATR: {len(base)} real windows ({n_init} days), {len(test_set)} test windows; "
              f"naive MAPE {naive:.5f}", flush=True)
        for m in models:
            stretch = SyntheticStretches(paths[m], prices[0], rng)
            for run in range(args.runs):
                data, added = base, 0
                for k in sorted(args.aug_counts):
                    while added < k:
                        data = np.concatenate([data, rolling(stretch(args.aug_length), args.window, sc)])
                        added += 1
                    mape = fit_mape(data, test_set, sc, args, seed=run)
                    rows.append({"setting": "tatr", "model": m, "x": k * args.aug_length, "run": run, "mape": mape})
            at = lambda x: trimmed([r["mape"] for r in rows if r["model"] == m and r["setting"] == "tatr" and r["x"] == x])
            print(f"[tatr] {m}: MAPE {at(0):.5f} -> {at(max(args.aug_counts) * args.aug_length):.5f}", flush=True)
    if args.setting in ("tmtr", "both"):
        sc = (held.min(), held.max())
        test_set = rolling(held, args.window, sc)
        for m in models:
            stretch = SyntheticStretches(paths[m], prices[0], rng)
            for run in range(args.runs):
                syn = stretch(n_init)
                for q in args.proportions:
                    n_syn = int(n_init * q / 100)
                    parts = [contiguous(real, n_init - n_syn, rng) if n_syn < n_init else real[:0],
                             contiguous(syn, n_syn, rng) if n_syn > 0 else syn[:0]]
                    data = np.concatenate([rolling(p, args.window, sc) for p in parts])
                    mape = fit_mape(data, test_set, sc, args, seed=run)
                    rows.append({"setting": "tmtr", "model": m, "x": q, "run": run, "mape": mape})
            print(f"[tmtr] {m} done", flush=True)

    res = pd.DataFrame(rows)
    tag = f"authors_h{h}"
    res.to_csv(out / f"downstream_{tag}.csv", index=False)
    summary = res.groupby(["setting", "model", "x"]).mape.apply(trimmed).rename("mape").reset_index()
    lines = [f"# {args.asset}: downstream MAPE, horizon {h}, authors' protocol "
             f"({n_init} real days, {args.runs} runs, 2.5%-trimmed means)", ""]
    for setting, g in summary.groupby("setting"):
        piv = g.pivot(index="model", columns="x", values="mape")
        lines += [f"## {setting.upper()}", "", md_table(piv, 5), ""]
        if setting == "tatr":
            lines += [f"Naive MAPE on the same test windows: {naive:.5f}", "",
                      "Relative MAPE change at the largest augmentation: "
                      + ", ".join(f"{LABELS[m]} {100 * (piv.loc[m].iloc[-1] / piv.loc[m].iloc[0] - 1):+.1f}%"
                                  for m in piv.index), ""]
    (out / f"downstream_{tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    plot(res, naive if args.setting != "tmtr" else float("nan"), out / f"fig6_{tag}.png")


if __name__ == "__main__":
    main()
