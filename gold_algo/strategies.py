"""
Simple rule-based strategies = your BASELINES.

Why bother with dumb strategies when you want ML? Because if your ML model
cannot beat these after costs, the ML adds nothing but complexity.

Each function takes the price DataFrame and returns a position Series
(+1 long, -1 short, 0 flat), decided using data up to and including each bar.
"""

from __future__ import annotations

import pandas as pd


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    """Always long. The benchmark everything must beat (gold has risen a lot over time)."""
    return pd.Series(1.0, index=df.index)


def ma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 100, allow_short: bool = False) -> pd.Series:
    """Trend following: long when the fast moving average is above the slow one.

    On 15-min bars, fast=20 is ~5 hours and slow=100 is ~1 day of trading.
    """
    fast_ma = df["close"].rolling(fast).mean()
    slow_ma = df["close"].rolling(slow).mean()
    pos = (fast_ma > slow_ma).astype(float)
    if allow_short:
        pos = pos.where(fast_ma > slow_ma, -1.0)
    return pos.where(slow_ma.notna(), 0.0)  # no position until we have enough history


def mean_reversion(df: pd.DataFrame, lookback: int = 48, entry_z: float = 2.0) -> pd.Series:
    """Fade big moves: buy when price is far BELOW its recent average, sell when far above.

    z-score = how many standard deviations price is from its moving average.
    Holds the trade until price crosses back to the average.
    """
    ma = df["close"].rolling(lookback).mean()
    sd = df["close"].rolling(lookback).std()
    z = (df["close"] - ma) / sd

    pos = pd.Series(float("nan"), index=df.index)
    pos[z < -entry_z] = 1.0       # far below average -> buy
    pos[z > entry_z] = -1.0       # far above average -> sell
    pos[(z * z.shift(1)) < 0] = 0.0  # crossed back through the average -> exit
    return pos.ffill().fillna(0.0)  # hold the last decision until a new signal
