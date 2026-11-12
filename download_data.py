"""Download the S&P 500 index and corn futures (ZC=F) daily prices from Yahoo Finance.

GOOG is read from ../archive/sp500_stocks.csv and needs no download.
"""
import json
import time
from urllib.request import Request, urlopen

import pandas as pd

from ftsd.data import ASSETS, DATA_DIR, yahoo_path

URL = ("https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
       "?period1={start}&period2={end}&interval=1d&events=div%2Csplit")


def download(ticker, start="1970-01-01", end="2026-01-01"):
    url = URL.format(ticker=ticker.replace("^", "%5E").replace("=", "%3D"),
                     start=int(pd.Timestamp(start).timestamp()), end=int(pd.Timestamp(end).timestamp()))
    with urlopen(Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as resp:
        res = json.load(resp)["chart"]["result"][0]
    tz = res["meta"]["exchangeTimezoneName"]
    quote = res["indicators"]["quote"][0]
    df = pd.DataFrame({
        "date": pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert(tz).date,
        "open": quote["open"], "high": quote["high"], "low": quote["low"],
        "close": quote["close"], "volume": quote["volume"],
    })
    adj = res["indicators"].get("adjclose")
    if adj:
        df["adjclose"] = adj[0]["adjclose"]
    return df.dropna(subset=["close"]).drop_duplicates("date", keep="last")


if __name__ == "__main__":
    DATA_DIR.mkdir(exist_ok=True)
    for asset, cfg in ASSETS.items():
        if not cfg["yahoo"]:
            continue
        df = download(cfg["yahoo"])
        df.to_csv(yahoo_path(cfg["yahoo"]), index=False)
        print(f"{asset}: {len(df)} rows {df.date.iloc[0]} .. {df.date.iloc[-1]} -> {yahoo_path(cfg['yahoo'])}")
        time.sleep(1)
