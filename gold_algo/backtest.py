"""
The backtester: "if I had followed these positions, what would have happened?"

THE ONE RULE THAT MATTERS MOST - no look-ahead:
    A position decided using bar t's CLOSE can only earn the return of bar t+1.
    You cannot trade at a price you used to make the decision.
That is why we do `position.shift(1)` below. Forgetting this single line is the
most common reason beginners "discover" amazing strategies.

Positions are numbers: +1 = fully long gold, -1 = fully short, 0 = flat.
(0.5 would mean half size.)

Costs: every time the position changes we pay `cost_bps` basis points
(1 bp = 0.01%) per unit traded. Going from -1 to +1 trades 2 units, so it
pays twice. For retail XAU/USD, a spread of ~$0.20-0.40 on a ~$2000-4000
price is roughly 0.5-2 bp per side, plus slippage. Default 1.0 bp is a
reasonable starting guess - check YOUR broker's real spread.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    returns: pd.Series      # strategy return per bar, after costs
    gross: pd.Series        # strategy return per bar, before costs
    costs: pd.Series        # cost paid per bar (as a return)
    position: pd.Series     # position held DURING each bar
    equity: pd.Series       # growth of $1


def run_backtest(close: pd.Series, position: pd.Series, cost_bps: float = 1.0) -> BacktestResult:
    """Simulate trading `position` on `close` prices.

    position[t] = what you decide at the close of bar t.
    It is held during bar t+1 and earns bar t+1's return.
    """
    position = position.reindex(close.index).fillna(0.0)

    bar_return = close.pct_change().fillna(0.0)           # return of each bar
    held = position.shift(1).fillna(0.0)                   # <-- the no-look-ahead line
    gross = held * bar_return

    traded = held.diff().abs().fillna(held.abs())          # units bought/sold to get there
    costs = traded * cost_bps / 10_000

    net = gross - costs
    equity = (1 + net).cumprod()
    return BacktestResult(returns=net, gross=gross, costs=costs, position=held, equity=equity)
