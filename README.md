# tradersjoy

[![CI](https://github.com/arindamlabs/tradersjoy/actions/workflows/ci.yml/badge.svg)](https://github.com/arindamlabs/tradersjoy/actions/workflows/ci.yml)

An automated paper-trading system: daily-swing strategies on US equities,
executed against the Alpaca paper-trading API. Built to be a serious learning
project for quant infrastructure and ML-for-trading, not a get-rich-quick bot.

**Status: Phase 2** (backtester). The CLI works, the package installs, CI is
green. Daily bars for a 20-ticker watchlist back to 2005 ingest into a local
SQLite store via yfinance, and an event-driven backtester replays them through
baseline strategies with realistic, no-look-ahead fills.

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
| 3 | Live paper-trading loop | not started |
| 4 | ML strategy with walk-forward validation | not started |
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
