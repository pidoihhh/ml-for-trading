"""
Tests that protect against the bugs that create fake profits.
Run with:  python -m pytest -q
"""

import numpy as np
import pandas as pd

from gold_algo import data, ml, strategies
from gold_algo.backtest import run_backtest


def _prices():
    return data.make_synthetic(n_days=60, seed=1)


def test_no_lookahead_in_backtest():
    """A position decided at bar t must not earn bar t's own return."""
    idx = pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC")
    close = pd.Series([100.0, 110.0, 121.0, 121.0], index=idx)
    # Decide to go long at the close of bar 1 (price 110)...
    pos = pd.Series([0.0, 1.0, 1.0, 0.0], index=idx)
    res = run_backtest(close, pos, cost_bps=0)
    # ...so we earn bar 2's +10%, NOT bar 1's +10% that already happened.
    assert np.isclose(res.gross.iloc[1], 0.0)
    assert np.isclose(res.gross.iloc[2], 0.10)
    assert np.isclose(res.equity.iloc[-1], 1.10)


def test_costs_charged_per_unit_traded():
    idx = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    close = pd.Series(100.0, index=idx)  # flat price -> only costs matter
    pos = pd.Series([1.0, -1.0, -1.0, 0.0, 0.0], index=idx)  # enter 1, flip 2, exit 1 = 4 units
    res = run_backtest(close, pos, cost_bps=10)
    assert np.isclose(res.costs.sum(), 4 * 10 / 10_000)


def test_strategies_do_not_use_future_data():
    """Changing FUTURE prices must not change PAST positions."""
    df = _prices()
    cut = len(df) // 2
    altered = df.copy()
    altered.iloc[cut:, :4] *= 1.5  # wreck the second half

    for fn in (strategies.ma_crossover, strategies.mean_reversion):
        a, b = fn(df), fn(altered)
        pd.testing.assert_series_equal(a.iloc[:cut], b.iloc[:cut])

    fa, fb = ml.make_features(df), ml.make_features(altered)
    pd.testing.assert_frame_equal(fa.iloc[:cut], fb.iloc[:cut])


def test_walk_forward_runs_and_is_flat_before_first_model():
    df = _prices()
    pos, prob = ml.walk_forward(df, train_bars=2000, test_bars=500)
    assert (pos.iloc[:2000] == 0).all()
    assert prob.iloc[:2000].isna().all()
    assert prob.iloc[2100:].notna().mean() > 0.9
