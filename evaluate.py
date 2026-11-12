"""Fidelity evaluation: Tables 1, 3, 4 and Fig. 5 of the paper, plus stylized-fact summaries.

    python evaluate.py --asset GOOG
Writes results/<asset>/tables.md, fidelity.csv and figures.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ftsd.data import load_split, results_dir
from ftsd.metrics import acf, distribution_tests, log_returns, performance_ratios, stylized_facts

ORDER = ["rcgan", "timegan", "csdi", "fts", "fts_plus", "fts_plus_nogarch", "fts_plus_novol"]
LABELS = {"rcgan": "RCGAN", "timegan": "TimeGAN", "csdi": "CSDI (Generative)", "fts": "FTS-Diffusion",
          "fts_plus": "FTS-Diffusion+ (slides)", "fts_plus_nogarch": "FTS+ w/o GARCH",
          "fts_plus_novol": "FTS+ w/o volume"}


def md_table(df, digits=3):
    head = "| " + " | ".join([str(df.index.name or "")] + [str(c) for c in df.columns]) + " |"
    sep = "|" + "---|" * (len(df.columns) + 1)
    body = ["| " + " | ".join([str(i)] + [f"{v:.{digits}f}" for v in row]) + " |"
            for i, row in zip(df.index, df.to_numpy())]
    return "\n".join([head, sep] + body)


def available(out):
    return [m for m in ORDER if (out / m / "samples.npz").exists()]


def fidelity(syn_r, ref):
    """Per-path metrics against a reference sample; synthetic paths are cut to the reference length."""
    rows = []
    real_ratio = performance_ratios(ref)
    for r in syn_r:
        r = r[-len(ref):]
        row = distribution_tests(ref, r)
        ratio = performance_ratios(r)
        row.update({f"d{k}": abs(ratio[k] - real_ratio[k]) for k in ratio})
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(df, cols, digits=3):
    return {c: f"{df[c].mean():.{digits}f} ± {df[c].std():.{digits}f}" for c in cols}


def plot_fig5(real, syn, name, path):
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    lags = np.arange(1, 51)
    for row, (label, r) in enumerate([("Real", real), (LABELS[name], syn)]):
        x = np.concatenate(r) if isinstance(r, list) else r
        lim = np.quantile(np.abs(x), 0.999)
        grid = np.linspace(-lim, lim, 200)
        ax[row, 0].hist(x, bins=120, range=(-lim, lim), density=True, alpha=0.6, label=label)
        ax[row, 0].plot(grid, stats.norm.pdf(grid, x.mean(), x.std()), "k--", lw=1, label="Gaussian")
        ax[row, 0].set_xlabel("Returns")
        ax[row, 0].set_ylabel("Density")
        ax[row, 0].legend(fontsize=8)
        stats.probplot((x - x.mean()) / x.std(), dist="norm", plot=ax[row, 1])
        ax[row, 1].set_title("")
        ax[row, 1].set_xlabel("Theoretical")
        ax[row, 1].set_ylabel("Sample")
        series = r if isinstance(r, list) else [r]
        a = np.mean([acf(np.abs(s), lags) for s in series], axis=0)
        ax[row, 2].bar(lags, a, width=0.8)
        ax[row, 2].set_ylim(-0.05, 0.5)
        ax[row, 2].set_xlabel("Days")
        ax[row, 2].set_ylabel("Autocorr. |r|")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_acf_compare(real, syns, path):
    lags = np.arange(1, 101)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(lags, acf(np.abs(real), lags), "k", lw=2, label="Real (train)")
    for name, r in syns.items():
        ax.plot(lags, np.mean([acf(np.abs(s), lags) for s in r], axis=0), label=LABELS[name])
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("ACF of |returns|")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_patterns(centroids, path, title):
    k = len(centroids)
    cols = min(k, 7)
    rows = int(np.ceil(k / cols))
    fig, ax = plt.subplots(rows, cols, figsize=(2 * cols, 1.6 * rows), squeeze=False)
    for i, a in enumerate(ax.flat):
        a.axis("off")
        if i < k:
            a.plot(centroids[i], lw=2)
            a.set_title(f"p{i + 1}", fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_paths(real_logp, samples, path):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(np.exp(real_logp), "k", lw=1.5, label="Real (train)")
    for name, s in samples.items():
        ax.plot(np.exp(s[0]), lw=0.8, alpha=0.8, label=LABELS[name])
    ax.set_yscale("log")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="GOOG")
    args = ap.parse_args()
    out = results_dir(args.asset)
    df, train, test = load_split(args.asset)
    logp_train = np.log(train.close.to_numpy())
    r_train = log_returns(logp_train)
    r_test = log_returns(np.log(np.concatenate([[train.close.iloc[-1]], test.close.to_numpy()])))
    models = available(out)
    if not models:
        raise SystemExit(f"no samples in {out}; run train.py first")

    samples = {m: np.load(out / m / "samples.npz")["logp"] for m in models}
    syn_r = {m: list(log_returns(s)) for m, s in samples.items()}

    lines = [f"# {args.asset}: fidelity of synthetic returns",
             f"Train {train.index[0].date()}..{train.index[-1].date()} ({len(r_train)} returns), "
             f"test {test.index[0].date()}..{test.index[-1].date()} ({len(r_test)} returns). "
             f"Mean ± std over {len(syn_r[models[0]])} synthetic paths.", ""]
    records = []
    for ref_name, ref in [("test", r_test), ("train", r_train)]:
        tabs = {m: fidelity(syn_r[m], ref) for m in models}
        for m, t in tabs.items():
            records.append({"reference": ref_name, "model": m, **t.mean().to_dict()})
        lines += [f"## Reference: {ref_name} returns", "",
                  "Table 1/3 (KS, AD, ES are p-values: higher is better; W: lower is better)", "",
                  "| Model | KS | AD | ES | W |", "|---|---|---|---|---|"]
        for m in models:
            f = fmt(tabs[m], ["KS", "AD", "ES"])
            w = fmt(tabs[m], ["W"], 4)
            lines.append(f"| {LABELS[m]} | {f['KS']} | {f['AD']} | {f['ES']} | {w['W']} |")
        lines += ["", "Table 4 (absolute difference to the real ratios: lower is better)", "",
                  "| Model | ΔSR | ΔSoR | ΔCR |", "|---|---|---|---|"]
        for m in models:
            f = fmt(tabs[m], ["dSR", "dSoR", "dCR"])
            lines.append(f"| {LABELS[m]} | {f['dSR']} | {f['dSoR']} | {f['dCR']} |")
        lines.append("")

    sf_rows = {"Real train": stylized_facts(r_train), "Real test": stylized_facts(r_test)}
    for m in models:
        sf_rows[LABELS[m]] = pd.DataFrame([stylized_facts(r) for r in syn_r[m]]).mean().to_dict()
    sf = pd.DataFrame(sf_rows).T
    lines += ["## Stylized facts (mean over paths)", "", md_table(sf), ""]

    (out / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(records).to_csv(out / "fidelity.csv", index=False)
    sf.to_csv(out / "stylized_facts.csv")
    print("\n".join(lines))

    for m in models:
        plot_fig5(r_train, syn_r[m], m, out / f"fig5_{m}.png")
        z = np.load(out / m / "samples.npz")
        if "centroids" in z:
            plot_patterns(z["centroids"], out / f"patterns_{m}.png", f"{LABELS[m]}: SISC patterns")
    plot_acf_compare(r_train, syn_r, out / "acf_abs_returns.png")
    plot_paths(logp_train, samples, out / "sample_paths.png")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
