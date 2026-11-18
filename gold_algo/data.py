"""
Getting price data.

Every function here returns the same thing: a pandas DataFrame indexed by
timestamp (UTC), with lowercase columns: open, high, low, close, volume.
Keeping one format means the rest of the code never cares where data came from.

Three sources:
  1. load_yahoo()      - quick start, but Yahoo only gives the LAST ~60 DAYS of
                         15-minute data. That is far too little to judge a
                         strategy. Use it to get the code running, nothing more.
  2. load_csv()        - the serious option. Download years of XAUUSD 1-minute or
                         15-minute history (e.g. from Dukascopy or HistData) and
                         load it here.
  3. make_synthetic()  - fake random-walk "gold" prices. Has NO predictable
                         pattern by construction. Any strategy that looks
                         profitable on it is fooling you. Great for learning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["open", "high", "low", "close", "volume"]


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names, sort by time, drop duplicates and empty rows."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "volume" not in df.columns:
        df["volume"] = 0.0  # spot gold / forex data often has no real volume
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Data is missing columns: {missing}. Found: {list(df.columns)}")
    df = df[COLUMNS].astype(float)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    return df.dropna(subset=["open", "high", "low", "close"])


def resample(df: pd.DataFrame, rule: str = "15min") -> pd.DataFrame:
    """Turn 1-minute bars into 15-minute bars (or any other size).

    Labels each bar by its START time and drops periods with no trades
    (weekends, daily maintenance break).
    """
    out = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["close"])


def load_yahoo(ticker: str = "GC=F", interval: str = "15m", period: str = "60d") -> pd.DataFrame:
    """Download recent gold futures data from Yahoo Finance.

    GC=F = COMEX gold futures (continuous contract). Yahoo limits 15m data to
    ~60 days. Note: the continuous contract jumps when it rolls from one
    futures month to the next; that jump is not a real price move.
    """
    import yfinance as yf  # imported here so the rest works without it

    raw = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
    if raw.empty:
        raise RuntimeError(f"Yahoo returned no data for {ticker}. Check your internet connection.")
    if isinstance(raw.columns, pd.MultiIndex):  # newer yfinance returns (field, ticker) columns
        raw.columns = raw.columns.get_level_values(0)
    return _clean(raw)


def load_csv(path: str, time_col: str | None = None, resample_to: str | None = "15min") -> pd.DataFrame:
    """Load OHLC(V) bars from a CSV file.

    Expects a time column (first column by default) and open/high/low/close
    columns (any capitalisation). If your file has 1-minute bars, keep
    resample_to="15min". If it is already 15-minute, it does no harm.
    """
    df = pd.read_csv(path)
    time_col = time_col or df.columns[0]
    df = df.set_index(time_col)
    df = _clean(df)
    return resample(df, resample_to) if resample_to else df


def make_synthetic(n_days: int = 500, seed: int = 0, start_price: float = 2000.0) -> pd.DataFrame:
    """Generate fake 15-minute gold-like bars with NO exploitable edge.

    - ~23 trading hours a day, 5 days a week (like gold futures)
    - volatility is higher during London/New York hours (like real gold)
    - but returns are pure random noise: the future cannot be predicted.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days * 96, freq="15min", tz="UTC")
    idx = idx[(idx.dayofweek < 5) & (idx.hour != 22)]  # drop weekends + 1h daily break

    hour = idx.hour.to_numpy()
    busy = ((hour >= 7) & (hour < 17)).astype(float)       # London + NY overlap-ish
    vol = 0.0008 * (0.6 + 0.8 * busy)                       # ~0.05%-0.11% per 15 min
    rets = rng.standard_normal(len(idx)) * vol

    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.r_[start_price, close[:-1]]
    wiggle = np.abs(rng.standard_normal(len(idx))) * vol * close
    high = np.maximum(open_, close) + wiggle
    low = np.minimum(open_, close) - wiggle
    volume = rng.integers(100, 1000, len(idx)) * (1 + busy)

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )
    return _clean(df)
