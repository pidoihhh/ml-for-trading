"""
Run every strategy on gold 15-minute data and print a comparison table.

Examples:
    python run.py --source synthetic            # fake data, no internet needed
    python run.py --source yahoo                # last ~60 days of real gold futures
    python run.py --source csv --path data/XAUUSD_1m.csv
    python run.py --source csv --path data/XAUUSD_1m.csv --cost-bps 2   # stress-test costs
"""

from __future__ import annotations

import argparse

import pandas as pd

from gold_algo import data, ml, strategies
from gold_algo.backtest import run_backtest
from gold_algo.metrics import compare


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["synthetic", "yahoo", "csv"], default="synthetic")
    p.add_argument("--path", help="CSV file when --source csv")
    p.add_argument("--cost-bps", type=float, default=1.0, help="cost per unit traded, in basis points")
    p.add_argument("--no-ml", action="store_true", help="skip the (slower) ML strategy")
    args = p.parse_args()

    if args.source == "synthetic":
        df = data.make_synthetic()
        print("!! SYNTHETIC DATA: pure random walk. No strategy should make money here.")
        print("!! Positive results below = luck, not skill. That is the lesson.\n")
    elif args.source == "yahoo":
        df = data.load_yahoo()
        print("!! Yahoo gives only ~60 days of 15m data: too short to trust ANY result.\n")
    else:
        if not args.path:
            p.error("--path is required with --source csv")
        df = data.load_csv(args.path)

    print(f"Bars: {len(df):,}   from {df.index[0]}   to {df.index[-1]}")
    print(f"Cost assumption: {args.cost_bps} bp per unit traded\n")

    positions: dict[str, pd.Series] = {
        "buy_and_hold": strategies.buy_and_hold(df),
        "ma_cross_20_100": strategies.ma_crossover(df, 20, 100),
        "mean_rev_48_z2": strategies.mean_reversion(df, 48, 2.0),
    }

    if not args.no_ml:
        # Use smaller windows when there is little data (e.g. Yahoo's 60 days)
        train = min(22_000, len(df) // 2)
        test = max(500, min(2_000, len(df) // 10))
        print(f"ML walk-forward: train {train} bars, retrain every {test} bars ...")
        positions["ml_gbm"], _ = ml.walk_forward(df, train_bars=train, test_bars=test)
        positions["ml_SHUFFLED_check"], _ = ml.walk_forward(
            df, train_bars=train, test_bars=test, shuffle_labels=True
        )
        # Compare every strategy over the SAME period (when the ML had a model).
        df = df.iloc[train:]
        positions = {k: v.iloc[train:] for k, v in positions.items()}
        print(f"Evaluation period (out-of-sample only): {df.index[0]} -> {df.index[-1]}\n")

    results = {name: run_backtest(df["close"], pos, args.cost_bps) for name, pos in positions.items()}
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(compare(results))

    print(
        "\nHow to read this:\n"
        "  * Does anything beat buy_and_hold on sharpe AND max_drawdown?\n"
        "  * Compare gross_sum_% vs costs_sum_%: are costs eating the edge?\n"
        "  * ml_gbm must clearly beat ml_SHUFFLED_check, or the model learned nothing.\n"
        "  * Re-run with --cost-bps 2. A real edge survives higher costs."
    )


if __name__ == "__main__":
    main()
