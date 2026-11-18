"""
How to judge a strategy. Accuracy is NOT on this list on purpose: a model can
be right 55% of the time and still lose money if its losers are bigger than its
winners or if costs eat the edge.

Key numbers:
  - Sharpe ratio: return per unit of risk, per year. Rough guide for a
    backtest: < 0.5 weak, ~1 decent, > 2 on intraday data = be suspicious
    (usually a bug or overfitting).
  - Max drawdown: the worst peak-to-bottom loss. Ask yourself: could I keep
    trading through that without panicking?
  - Cost drag: how much of the gross profit was paid to the broker. Intraday
    strategies often die here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult


def bars_per_year(index: pd.DatetimeIndex) -> float:
    """Estimate how many bars occur in a year FROM THE DATA itself.

    Gold trades ~23h/day, 5 days/week, so this is about 22,000 15-min bars.
    Measuring it avoids hard-coding a wrong number.
    """
    years = (index[-1] - index[0]).total_seconds() / (365.25 * 24 * 3600)
    return len(index) / years if years > 0 else np.nan


def summarize(result: BacktestResult) -> dict:
    r = result.returns
    n_per_year = bars_per_year(r.index)
    eq = result.equity

    total_return = eq.iloc[-1] - 1
    years = len(r) / n_per_year
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 and eq.iloc[-1] > 0 else np.nan
    ann_vol = r.std() * np.sqrt(n_per_year)
    sharpe = r.mean() / r.std() * np.sqrt(n_per_year) if r.std() > 0 else 0.0
    max_dd = (eq / eq.cummax() - 1).min()

    trades = int((result.position.diff().abs() > 0).sum())
    exposure = float((result.position != 0).mean())
    gross_total = result.gross.sum()
    cost_total = result.costs.sum()

    return {
        "total_return_%": 100 * total_return,
        "CAGR_%": 100 * cagr,
        "ann_vol_%": 100 * ann_vol,
        "sharpe": sharpe,
        "max_drawdown_%": 100 * max_dd,
        "trades": trades,
        "time_in_market_%": 100 * exposure,
        "gross_sum_%": 100 * gross_total,
        "costs_sum_%": 100 * cost_total,
    }


def compare(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Put several strategies side by side in one table."""
    return pd.DataFrame({name: summarize(res) for name, res in results.items()}).T.round(2)
