# tradersjoy

An automated paper-trading system: daily-swing strategies on US equities,
executed against the Alpaca paper-trading API. Built to be a serious learning
project for quant infrastructure and ML-for-trading — not a get-rich-quick bot.

**Status: Phase 0** (scaffolding). The CLI works, the package installs, CI is
green. No strategy or data layer yet.

## Setup

Requires Python 3.12+ and [uv](https://github.com/astral-sh/uv).

```bash
# install dependencies into .venv/
uv sync

# copy env template and fill in your Alpaca paper-trading keys
cp .env.example .env
$EDITOR .env

# run tests
uv run pytest

# see the CLI surface (subcommands are stubs until later phases)
uv run tradersjoy --help
```

## Phased delivery

| Phase | What | Status |
|---|---|---|
| 0 | Scaffold (CLI, package, CI, tests) | in progress |
| 1 | Data ingest (Alpaca + yfinance → SQLite) | not started |
| 2 | Backtester + portfolio + baseline strategies | not started |
| 3 | Live paper-trading loop | not started |
| 4 | ML strategy with walk-forward validation | not started |
| 5 | Risk management (position sizing, stops, circuit-breaker) | not started |
| 6 | Automation + Streamlit dashboard | not started |
| 7 | Disciplined retraining loop | not started |

## Design principles

- Same `Strategy` interface runs in backtest and live — no code-path divergence.
- Broker, data source, and clock are pluggable behind interfaces.
- Walk-forward validation is the only acceptable way to evaluate a model.
- The system retrains carefully on a quarterly cadence — never on its own live
  paper-trading data.
- Never trade real money until the system has paper-traded profitably
  out-of-sample for at least 6 months.
