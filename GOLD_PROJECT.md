# Gold 15-Minute Algo Trading — A Learning Project

A small, heavily commented Python project for researching trading strategies on
gold (XAU/USD) using 15-minute bars. Built for learning, so every file explains
*why* it does what it does.

> **This is a research tool, not a money machine.** Nothing here is a proven
> profitable strategy. The goal is to learn how to test ideas *honestly*.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest -q                # should say: 4 passed
python run.py --source synthetic   # runs with fake data, no internet needed
```

## Project layout

| File | What it does | Read it when you want to learn... |
|---|---|---|
| `gold_algo/data.py` | Loads prices (Yahoo, CSV, or fake) | Where data comes from, resampling 1m → 15m |
| `gold_algo/backtest.py` | Simulates trading, charges costs | **Look-ahead bias** — the #1 beginner bug |
| `gold_algo/metrics.py` | Sharpe, drawdown, cost drag | How to judge a strategy (not accuracy!) |
| `gold_algo/strategies.py` | Buy & hold, MA crossover, mean reversion | Baselines, trend vs mean-reversion |
| `gold_algo/ml.py` | Gradient-boosted trees, walk-forward | Features, labels, leakage, walk-forward |
| `run.py` | Runs everything, prints a table | Putting it together |
| `tests/` | Catches fake-profit bugs | Why tests matter in trading code |

## The first lesson: random data can look profitable

`python run.py --source synthetic` runs every strategy on a **random walk**.
By construction, nothing in it can be predicted. Here is the result from seed 0:

| strategy | sharpe | costs paid | note |
|---|---|---|---|
| buy_and_hold | 1.13 | 0.01% | the random walk happened to drift up |
| mean_rev_48_z2 | **1.53** | 7.1% | **looks great — but it's pure luck** |
| ma_cross_20_100 | -0.61 | 2.2% | |
| ml_gbm | -6.23 | **19.8%** | 1,962 trades; costs destroyed it |
| ml_SHUFFLED_check | -3.88 | 2.0% | |

Mean reversion on seeds 1, 2 and 3 gave Sharpe -0.34, -1.59 and -1.30.
**The 1.53 was noise.** If you had seen that number on real gold data, you would
probably have believed it. That is exactly how people lose money.

What to take from this:
1. **One good backtest proves nothing.** Test on several periods and several settings.
2. **Costs matter enormously on 15-min bars.** The ML model's gross loss was small.
   Trading too often is what killed it.
3. **Always run the shuffled-label check.** If the real model doesn't clearly beat
   a model trained on scrambled answers, it learned nothing.

## Getting real data

| Source | History | Notes |
|---|---|---|
| `--source yahoo` (GC=F futures) | ~60 days of 15m | Only for testing that the code runs. Far too short to trust. |
| Dukascopy (XAU/USD) | Many years, down to tick | Free. Download 1-minute bars as CSV. Best free option. |
| HistData.com (XAUUSD) | Years of 1-minute | Free. Semicolon-separated, so you may need to adjust `load_csv`. |
| Your broker (MT5, OANDA, etc.) | Varies | Best match to the prices you would actually trade |

Put the CSV in `data/` (it is git-ignored, so it won't bloat the repo), then:

```bash
python run.py --source csv --path data/XAUUSD_1m.csv
python run.py --source csv --path data/XAUUSD_1m.csv --cost-bps 2   # stress test
```

Aim for **at least 3–5 years** of 15-minute data. One year of gold contains only one
or two "market moods". Your model needs to see more than that.

## Gold-specific things to know

- **Trading hours:** gold trades about 23 hours a day, 5 days a week. The Asia session
  is quiet. The London open (~07:00–08:00 UTC) and the New York open / US data releases
  (~12:30–14:30 UTC) are where most movement happens. The `hour_sin`/`hour_cos`
  features exist for this reason.
- **News spikes:** US CPI, NFP (jobs report) and FOMC (Fed) decisions can move gold
  1–2% in minutes, and spreads widen a lot exactly then. Your backtest's fixed cost
  assumption is **too optimistic** around news.
- **Spot vs futures:** XAU/USD (spot, what most retail brokers offer) and GC=F
  (COMEX futures) are close but not the same. Backtest on the instrument you will trade.
- **Futures roll:** the continuous GC=F series jumps when the contract month changes.
  That jump is not a tradable move.

## Your learning path (do these in order)

**Week 1–2: understand the machine**
- [ ] Run the tests and `run.py --source synthetic`. Try `--cost-bps 0` and `--cost-bps 3`.
- [ ] Read `backtest.py`. Delete `.shift(1)`, re-run, and watch the fake profits appear.
      Then put it back. (`test_no_lookahead_in_backtest` should fail without it.)
- [ ] Change the seed in `make_synthetic` five times. Write down how much each strategy's
      Sharpe jumps around **on data with zero edge**. That spread is your "noise floor".

**Week 3–4: real data, baselines only**
- [ ] Get 3+ years of XAU/USD from Dukascopy and run the baselines.
- [ ] Split the data in half by time. Does a strategy that wins in the first half also
      win in the second?
- [ ] Find your broker's real spread for gold and set `--cost-bps` to match.

**Week 5+: ML, carefully**
- [ ] Add one new feature at a time to `make_features`. Keep it only if it helps
      out-of-sample **and** beats the shuffled check.
- [ ] Try `horizon=16` (4 hours) in `walk_forward`. Longer holds mean fewer trades and
      lower costs.
- [ ] Raise `threshold` (e.g. 0.10) so the model trades only when it is confident.
- [ ] Keep a log of **every** idea you test, including the failures. If you test 50
      ideas, expect 2–3 to look good by pure chance.

**Before ANY real money**
- [ ] Paper trade for 1–3 months. Compare live results with the backtest.
- [ ] Decide your maximum loss per day and per month *before* starting.
- [ ] Never trade with leverage you don't fully understand. Gold CFDs often allow
      50–100× leverage, and a 1% move can wipe out an account.

## Skills to build in parallel

Weak Python or statistics will hold you back more than any choice of model. Focus on:
- **pandas:** indexing, `shift`, `rolling`, `resample`. This project uses all of them.
- **Statistics:** mean, standard deviation, what a Sharpe ratio means, why
  testing many ideas creates false positives (the "multiple testing" problem).
- **Recommended reading:** *Advances in Financial Machine Learning* (López de Prado),
  chapters on backtesting pitfalls and cross-validation. *Machine Learning for
  Algorithmic Trading* (Stefan Jansen) has practical code.
