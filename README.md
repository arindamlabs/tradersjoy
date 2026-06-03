# tradersjoy

[![CI](https://github.com/arindamlabs/tradersjoy/actions/workflows/ci.yml/badge.svg)](https://github.com/arindamlabs/tradersjoy/actions/workflows/ci.yml)

An automated paper-trading system: daily-swing strategies on US equities,
executed against the Alpaca paper-trading API. Built to be a serious learning
project for quant infrastructure and ML-for-trading, not a get-rich-quick bot.

**Status: Phase 4** (ML strategy). The CLI works, the package installs, CI is
green. Daily bars for a 20-ticker watchlist back to 2005 ingest into a local
SQLite store via yfinance; an event-driven backtester replays them through
baseline strategies with realistic, no-look-ahead fills; the same strategies can
drive live orders against the Alpaca paper account (dry-run by default); and a
gradient-boosted-tree model can be trained and scored honestly with walk-forward
validation.

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
# install dependencies into .venv/
uv sync

# copy env template and fill in your Alpaca paper-trading keys
cp .env.example .env
nano .env

# run tests
uv run pytest

# see the CLI surface
uv run tradersjoy --help

# backfill ~20 years of daily bars for the watchlist into data/tradersjoy.sqlite
uv run tradersjoy ingest

# backtest a baseline strategy against the stored bars
uv run tradersjoy backtest --strategy buyhold --tickers SPY
uv run tradersjoy backtest --strategy sma --tickers SPY --short-window 20 --long-window 50

# see what a strategy WOULD trade live today (dry run; places nothing)
uv run tradersjoy trade --strategy buyhold

# actually place those orders on the Alpaca paper account
uv run tradersjoy trade --strategy buyhold --execute

# train an ML model and score it honestly with walk-forward validation
uv run tradersjoy train

# run the trained model as a strategy (dry run)
uv run tradersjoy trade --strategy ml --model data/models/ml.joblib
```

## Backtesting

The backtester replays stored daily bars one session at a time and reports the
standard scorecard (total return, CAGR, annualised Sharpe, max drawdown, hit
rate). Two assumptions keep results honest rather than flattering:

- **No look-ahead.** A strategy decides on day T's close; its orders fill at day
  T+1's *open*. It can never trade at a price it has already seen.
- **Adverse slippage.** Every fill moves against the trader by a configurable
  number of basis points (`--slippage-bps`, default 5). Real fills are
  uncertain; this is a deliberately pessimistic stand-in.

Baselines included: `buyhold` (equal-weight, the benchmark to beat) and `sma`
(long-only fast/slow moving-average crossover). On 2005-2026 SPY data the SMA
rule roughly halves the 2008 drawdown but underperforms buy-and-hold on total
return -- the expected, sobering result a realistic engine should produce.

## Live paper trading

The `trade` command runs one decision cycle of the *same* strategy against the
Alpaca paper account: it refreshes recent bars, reads the live account, lets the
strategy decide on the latest close, and (optionally) places the orders. Nothing
about the strategy changes between backtest and live; only the broker and the
source of positions do.

Safety and honesty:

- **Dry run by default.** Without `--execute`, `trade` reads state and prints
  exactly what it would do, but places no orders. Pass `--execute` to act.
- **Paper only.** The Alpaca client is pinned to the paper endpoint;
  real-money trading is deliberately not wired up.
- **Whole shares only** live (fractional quantities are floored), so live fills
  can differ slightly from a fractional backtest.
- **Market orders** mean real, uncontrolled slippage. Run `trade` once per day,
  ideally after the close, so orders queue for the next open and the timing
  matches the backtest's next-open assumption.

## Machine-learning strategy

The `train` command builds a learning table from the stored bars and fits a
gradient-boosted-tree classifier to predict a simple target: **will this stock
rise over the next 5 trading days?** The same features are computed live by the
`ml` strategy, so there is no train/serve skew, and every feature is a function
of the past only, so there is no look-ahead in the inputs.

The model is scored with **walk-forward validation**, the only honest way to
evaluate a trading model: train on the past, test on the next unseen year, roll
forward, repeat. A row's 5-day answer window is *purged* at each train/test
boundary so no sliver of the test year leaks into training. The naive
alternative (a random train/test split) would let the model learn from its own
future and is never used here.

Two deliberate honesty choices shape how results are read:

- **The baseline is the base rate, not 50%.** Because the market drifts up,
  roughly 56% of 5-day windows are up days. A model must beat *that*, not a coin
  flip, to mean anything; the scorecard prints accuracy next to the base rate.
- **AUC measures ranking skill.** It is the chance the model ranks a random
  up-day above a random down-day; 0.50 is pure luck. Ranking is what the strategy
  needs, since it buys the top-scored names.

On the 20-ticker watchlist with these starter features, the first honest result
is a near-coin-flip (AUC about 0.50, accuracy at or just below the base rate).
That is the expected, sober baseline, and it is far more useful than an
impressive number that turns out to be a leak. The next gains come from better
*features*, validated the same honest way, not from a fancier model.

The walk-forward report is the trustworthy track record. Running
`backtest --strategy ml` over the model's own training window is *in-sample* and
flatters it; the CLI prints a warning to that effect.

## API documentation

The code is documented with Google-style docstrings. Browse them as HTML with
[pdoc](https://pdoc.dev) (no config, live-reloads as you edit):

```bash
# serve interactively at http://localhost:8080
uv run pdoc -d google tradersjoy

# or build a static site into docs/api/ (gitignored)
uv run pdoc -d google tradersjoy -o docs/api
```

## Phased delivery

| Phase | What | Status |
|---|---|---|
| 0 | Scaffold (CLI, package, CI, tests) | done |
| 1 | Data ingest (yfinance -> SQLite) | done |
| 2 | Backtester + portfolio + baseline strategies | done |
| 3 | Live paper-trading loop | done |
| 4 | ML strategy with walk-forward validation | done |
| 5 | Risk management (position sizing, stops, circuit-breaker) | not started |
| 6 | Automation + Streamlit dashboard | not started |
| 7 | Disciplined retraining loop | not started |

## Design principles

- Same `Strategy` interface runs in backtest and live, no code-path divergence.
- Broker, data source, and clock are pluggable behind interfaces.
- Walk-forward validation is the only acceptable way to evaluate a model.
- The system retrains carefully on a quarterly cadence, never on its own live
  paper-trading data.
- Never trade real money until the system has paper-traded profitably
  out-of-sample for at least 6 months.
