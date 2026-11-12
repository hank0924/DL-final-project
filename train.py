"""Train generators on the training split of an asset and save synthetic paths.

Example:
    python train.py --asset GOOG --models fts fts_plus rcgan timegan csdi
Outputs go to results/<asset>/<model>/ (samples.npz, model.pt, meta.json).
"""
import argparse
import json
import time

import numpy as np
import torch

from ftsd.baselines import CSDI, RCGAN, TimeGAN
from ftsd.data import asset_config, load_split, results_dir
from ftsd.fts_diffusion import FTSDiffusion
from ftsd.fts_plus import FTSPlus
from ftsd.utils import get_device, set_seed

MODELS = ["fts", "fts_plus", "rcgan", "timegan", "csdi"]
# Ablations of the improved model: each removes one change from the slides.
ABLATIONS = ["fts_plus_nogarch", "fts_plus_novol"]


def build(name, k, args):
    q = args.quick
    common = {"device": args.device, "seed": args.seed}
    sisc_iters = 3 if q else args.sisc_iters
    if name == "fts":
        return FTSDiffusion(k, sisc_iters=sisc_iters, gen_epochs=20 if q else args.gen_epochs,
                            evo_epochs=20 if q else 1000, evo_noise=not args.no_evo_noise, **common)
    if name.startswith("fts_plus"):
        return FTSPlus(k, sisc_iters=sisc_iters, dit_epochs=20 if q else args.gen_epochs,
                       evo_steps=50 if q else args.evo_steps, context=args.context,
                       use_garch=name != "fts_plus_nogarch", use_volume=name != "fts_plus_novol", **common)
    iters = 50 if q else args.baseline_iters
    return {"rcgan": RCGAN, "timegan": TimeGAN, "csdi": CSDI}[name](iters=iters, **common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="GOOG")
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS + ABLATIONS)
    ap.add_argument("--k", type=int, help="number of patterns (default: paper value for the asset)")
    ap.add_argument("--n_paths", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sisc_iters", type=int, default=20)
    ap.add_argument("--gen_epochs", type=int, default=2000)
    ap.add_argument("--evo_steps", type=int, default=4000)
    ap.add_argument("--context", type=int, default=16)
    ap.add_argument("--baseline_iters", type=int, default=5000)
    ap.add_argument("--no_evo_noise", action="store_true",
                    help="original FTS-Diffusion: use point predictions of length/magnitude")
    ap.add_argument("--quick", action="store_true", help="tiny budgets for a smoke test")
    ap.add_argument("--sample_only", action="store_true", help="reuse the saved model.pt and only resample")
    args = ap.parse_args()
    args.device = get_device(args.device)
    k = args.k or asset_config(args.asset)["k"]

    df, train, test = load_split(args.asset)
    logp = np.log(train.close.to_numpy())
    print(f"{args.asset}: {len(df)} days, train {train.index[0].date()}..{train.index[-1].date()} "
          f"({len(train)}), test {len(test)}; K={k}; device {args.device}")

    for name in args.models:
        set_seed(args.seed)
        out = results_dir(args.asset) / name
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        if args.sample_only:
            model = torch.load(out / "model.pt", weights_only=False)
            t_fit = json.loads((out / "meta.json").read_text())["fit_seconds"]
        else:
            model = build(name, k, args)
            model.fit(logp, volume=train.volume.to_numpy(), dates=train.index)
            t_fit = time.time() - t0
        t0 = time.time()
        paths = model.sample(args.n_paths, len(logp) - 1, seed=args.seed)
        t_gen = time.time() - t0
        extra = {}
        if hasattr(model, "sisc"):
            seg = model.sisc.segmentation_
            extra = {"centroids": model.sisc.centroids_, "seg_starts": seg.starts,
                     "seg_lengths": seg.lengths, "seg_labels": seg.labels}
        np.savez_compressed(out / "samples.npz", logp=paths, **extra)
        torch.save(model, out / "model.pt")
        n_params = sum(p.numel() for m in vars(model).values() if isinstance(m, torch.nn.Module)
                       for p in m.parameters())
        meta = {"model": name, "asset": args.asset, "k": k, "n_paths": args.n_paths, "seed": args.seed,
                "quick": args.quick, "train_days": len(train), "fit_seconds": round(t_fit, 1),
                "sample_seconds": round(t_gen, 1), "n_params": n_params}
        if hasattr(model, "garch"):
            meta["garch"] = model.garch.summary()
        (out / "meta.json").write_text(json.dumps(meta, indent=2))
        r = np.diff(paths, axis=1)
        print(f"== {name}: fit {t_fit:.0f}s, sample {t_gen:.0f}s, params {n_params}, "
              f"syn ret std {r.std():.4f} (real {np.diff(logp).std():.4f})\n", flush=True)


if __name__ == "__main__":
    main()
