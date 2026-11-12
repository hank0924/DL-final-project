"""Price loading and the chronological 80/20 split of Appendix E.1."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARCHIVE = ROOT.parent / "archive" / "sp500_stocks.csv"
RESULTS_DIR = ROOT / "results"

# Date ranges (Appendix E.1) and number of SISC clusters K (Appendix B.1) used in the paper.
# Assets without a Yahoo ticker are read from the S&P 500 stocks archive.
ASSETS = {
    "SP500": {"yahoo": "^GSPC", "start": "1980-01-01", "end": "2020-01-01", "k": 14},
    "GOOG": {"yahoo": None, "start": "2005-01-01", "end": "2020-01-01", "k": 11},
    "ZC=F": {"yahoo": "ZC=F", "start": "2001-01-01", "end": "2020-01-01", "k": 11},
}
DEFAULT_ASSET = {"yahoo": None, "start": "2005-01-01", "end": "2020-01-01", "k": 11}


def asset_config(asset):
    return ASSETS.get(asset, DEFAULT_ASSET)


def yahoo_path(ticker):
    return DATA_DIR / f"{ticker.replace('^', '').replace('=', '_')}.csv"


def results_dir(asset):
    path = RESULTS_DIR / asset.replace("=", "_")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _archive_frame(symbol):
    cache = DATA_DIR / f"archive_{symbol}.csv"
    if not cache.exists():
        DATA_DIR.mkdir(exist_ok=True)
        df = pd.read_csv(ARCHIVE, usecols=["date", "close", "volume", "symbol"])
        df = df[df.symbol == symbol]
        if df.empty:
            raise KeyError(f"{symbol} not found in {ARCHIVE}")
        df[["date", "close", "volume"]].to_csv(cache, index=False)
    return pd.read_csv(cache, parse_dates=["date"])


def load_frame(asset, start=None, end=None):
    """Daily close and volume of `asset`, indexed by date."""
    cfg = asset_config(asset)
    if cfg["yahoo"]:
        path = yahoo_path(cfg["yahoo"])
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; run download_data.py first")
        df = pd.read_csv(path, parse_dates=["date"])
    else:
        df = _archive_frame(asset)
    df = df.set_index("date")[["close", "volume"]].astype(float).sort_index()
    df = df[(df.index >= (start or cfg["start"])) & (df.index < (end or cfg["end"]))]
    df = df[df.close > 0].dropna(subset=["close"])
    df["volume"] = df.volume.fillna(0.0)
    return df


def load_split(asset, train_frac=0.8):
    """Full frame plus its chronological train and test parts (Appendix E.1)."""
    df = load_frame(asset)
    n = int(len(df) * train_frac)
    return df, df.iloc[:n], df.iloc[n:]
