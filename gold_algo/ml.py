"""
A machine-learning strategy done carefully.

The model: gradient-boosted trees (fast, strong on tabular data, little tuning).
The question it answers: "will gold be higher in `horizon` bars than now?"

Three traps this file is built to avoid:

1. LOOK-AHEAD IN FEATURES: every feature at bar t uses only data up to bar t.

2. LEAKAGE BETWEEN TRAIN AND TEST: the label at bar t uses prices up to
   t + horizon. So the last `horizon` training rows "see" into the test
   period. We drop them (called "purging").

3. TESTING ON THE PAST: a random train/test split lets the model learn
   from the future. We use WALK-FORWARD instead: train on the past, predict
   the next block, slide forward, retrain, repeat. This mimics real life.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn raw prices into numbers a model can learn from.

    All of these use only current and past bars.
    """
    c = df["close"]
    r = np.log(c).diff()
    f = pd.DataFrame(index=df.index)

    # Momentum: how much has price moved over the last 1, 4 (1h), 16 (4h), 96 (~1 day) bars?
    for n in (1, 4, 16, 96):
        f[f"ret_{n}"] = np.log(c).diff(n)

    # Volatility: how jumpy has it been recently?
    f["vol_16"] = r.rolling(16).std()
    f["vol_96"] = r.rolling(96).std()
    f["vol_ratio"] = f["vol_16"] / f["vol_96"]

    # Where is price relative to its recent average (in standard deviations)?
    ma = c.rolling(96).mean()
    f["z_96"] = (c - ma) / c.rolling(96).std()

    # Bar shape: size of this bar's range relative to recent volatility
    f["range"] = (df["high"] - df["low"]) / c / f["vol_96"]

    # Time of day matters for gold: Asia is quiet, London/NY opens are busy.
    hour = df.index.hour + df.index.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    return f.replace([np.inf, -np.inf], np.nan)


def make_labels(df: pd.DataFrame, horizon: int = 4) -> pd.Series:
    """1 if price is higher `horizon` bars from now, else 0.

    This LOOKS INTO THE FUTURE by design - it is the answer we train on.
    It must never be used as a feature.
    """
    future = df["close"].shift(-horizon)
    label = (future > df["close"]).astype(float)
    return label.where(future.notna())  # last `horizon` bars have no answer yet


def walk_forward(
    df: pd.DataFrame,
    horizon: int = 4,
    train_bars: int = 22_000,   # ~1 year of 15-min gold bars
    test_bars: int = 2_000,     # ~1 month; retrain this often
    threshold: float = 0.05,    # trade only if P(up) is > 0.5+threshold or < 0.5-threshold
    allow_short: bool = True,
    shuffle_labels: bool = False,
    seed: int = 0,
) -> tuple[pd.Series, pd.Series]:
    """Train-predict-slide over the whole history.

    Returns (position, probability). Bars before the first test block get
    position 0 because we had no trained model yet.

    shuffle_labels=True is a SANITY CHECK: it scrambles the answers so there
    is nothing real to learn. If the "real" model does not clearly beat
    the shuffled one, its results are luck.
    """
    X = make_features(df)
    y = make_labels(df, horizon)
    prob = pd.Series(np.nan, index=df.index)
    rng = np.random.default_rng(seed)

    start = train_bars
    while start < len(df):
        end = min(start + test_bars, len(df))

        # Training window: the `train_bars` before `start`, minus the last
        # `horizon` rows whose labels peek into the test block (purging).
        tr = slice(start - train_bars, start - horizon)
        X_tr, y_tr = X.iloc[tr], y.iloc[tr]
        ok = X_tr.notna().all(axis=1) & y_tr.notna()
        X_tr, y_tr = X_tr[ok], y_tr[ok]
        if shuffle_labels:
            y_tr = pd.Series(rng.permutation(y_tr.to_numpy()), index=y_tr.index)

        if len(y_tr) > 500 and y_tr.nunique() == 2:
            model = HistGradientBoostingClassifier(
                max_iter=200, learning_rate=0.05, max_depth=3,
                min_samples_leaf=200,   # big leaves = less memorising of noise
                random_state=seed,
            )
            model.fit(X_tr, y_tr)

            X_te = X.iloc[start:end]
            good = X_te.notna().all(axis=1)
            if good.any():
                prob.loc[X_te.index[good]] = model.predict_proba(X_te[good])[:, 1]

        start = end

    pos = pd.Series(0.0, index=df.index)
    pos[prob > 0.5 + threshold] = 1.0
    if allow_short:
        pos[prob < 0.5 - threshold] = -1.0
    return pos, prob
