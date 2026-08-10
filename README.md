# tradersjoy

[![CI](https://github.com/arindamlabs/tradersjoy/actions/workflows/ci.yml/badge.svg)](https://github.com/arindamlabs/tradersjoy/actions/workflows/ci.yml)

An automated paper-trading system: daily-swing strategies on US equities,
executed against the Alpaca paper-trading API. Built to be a serious learning
project for quant infrastructure and ML-for-trading, not a get-rich-quick bot.

**Status: Phase 6c** (rebalance cadence + walk-forward backtest). The CLI works, the package
installs, CI is green. Daily bars for a 20-ticker watchlist back to 2005 ingest
into a local SQLite store via yfinance; an event-driven backtester replays them
through baseline strategies with realistic, no-look-ahead fills; the same
strategies can drive live orders against the Alpaca paper account (dry-run by
default); a gradient-boosted-tree model can be trained and scored honestly with
walk-forward validation; any strategy can be wrapped in a stateless risk layer
(position sizing, exposure cap, stop-loss, circuit breaker) that behaves
identically in backtest and live; every live run is recorded to a local journal
that a read-only Streamlit dashboard reads back as an equity curve and a decision
log; and a guarded `autorun` command runs the whole thing unattended on a systemd
timer or GitHub Actions, refusing to trade on stale data and leaving a heartbeat
so a scheduler that dies is visible rather than assumed healthy. Trading cadence
is tied to the model's label horizon, and `evaluate` reports what the whole thing
would have earned out-of-sample after costs, which so far is roughly what holding
the universe would have earned.

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

# out-of-sample backtest: does it beat buy-and-hold once you pay to trade it?
uv run tradersjoy evaluate --horizon 5

# run the trained model as a strategy (dry run)
uv run tradersjoy trade --strategy ml --model data/models/ml.joblib

# install the dashboard extra, then launch the read-only web dashboard
uv sync --extra dashboard
uv run tradersjoy dashboard      # opens http://localhost:8501

# one guarded, unattended decision cycle (what the systemd timer calls)
uv run tradersjoy autorun --strategy ml --model data/models/ml.joblib --risk

# is the scheduler actually still firing?
uv run tradersjoy status
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
gradient-boosted-tree classifier. By default it predicts a **relative,
cross-sectional** target: **will this stock beat the universe median over the
next 5 trading days?** (Pass `--absolute` for the simpler "will it rise?"
target.) The relative framing subtracts the market-wide move out of the answer
and asks only what the top-K strategy actually needs, which name is better than
its peers. The benchmark (SPY) is the yardstick for that comparison, so it is
excluded from the ranked set and never becomes a training row. It uses a small
set of past-only features:
multi-horizon returns, distance from 20/50/200-day averages, recent volatility,
a volume ratio, an RSI oscillator, drawdown from the recent high, and crucially
*market-relative* returns (this stock's move minus the benchmark's), since most
of any one stock's daily move is just the whole market. The same features are
computed live by the `ml` strategy, so there is no train/serve skew.

The model is scored with **walk-forward validation**, the only honest way to
evaluate a trading model: train on the past, test on the next unseen year, roll
forward, repeat. A row's 5-day answer window is *purged* at each train/test
boundary so no sliver of the test year leaks into training. The naive
alternative (a random train/test split) would let the model learn from its own
future and is never used here.

Two deliberate honesty choices shape how results are read:

- **The baseline is the base rate, not 50%.** For the relative target about half
  the names beat the median each day by construction, so the base rate sits near
  50%; for the absolute target the market's upward drift pushes it to ~56%. A
  model must beat *its own* base rate, not a coin flip, to mean anything; the
  scorecard prints accuracy next to the base rate.
- **AUC measures ranking skill.** It is the chance the model ranks a random
  winner above a random loser; 0.50 is pure luck. Ranking is what the strategy
  needs, since it buys the top-scored names.

On the 20-ticker watchlist the honest result is still a near-coin-flip on raw
ranking: AUC around 0.51 either way. But moving from the absolute to the relative
target helped exactly where it should. Under the absolute label, accuracy (55.5%)
actually sat *below* its base rate (56.2%), the model knew nothing useful. Under
the relative label, accuracy (51.5%) sits a clear ~4 points *above* its ~48% base
rate, and the most-confident decile's forward-return lift roughly doubled, from
+0.18% to +0.41% over 5 days. That +0.41% is the number that maps to the top-K
strategy, and it moved the right way. It is still a faint, regime-dependent,
plausibly-real signal rather than a tradeable edge (thin returns, inconsistent
across years, and slippage would eat much of it), but it is progress measured and
earned honestly. Real gains, if they come, will be earned the same way.

The walk-forward report is the trustworthy track record. Running
`backtest --strategy ml` over the model's own training window is *in-sample* and
flatters it; the CLI prints a warning to that effect.

### Rebalance cadence

How often the ranking is *acted on* is a separate decision from how good the
ranking is, and it turned out to matter more. Measured over 2016-2026, acting on
it every session replaced 1.4 of 5 positions a day: roughly **70x annual
turnover, a 6.7%/year drag** at 5 bps a side. Half of all positions lasted
exactly one session, and 65% of sells were re-bought within five days, while the
model's label predicts *five sessions* ahead. The strategy was reliably selling
before the move it had predicted had time to happen.

`--rebalance` ties the trading cadence to that horizon:

| Cadence | Sessions | Turnover | Cost drag/yr | Avg hold |
|---|---|---|---|---|
| `daily` | 1 | 70x | 6.7% | 2.6 sessions |
| **`weekly`** (default) | ~5 | 20x | **2.0%** | 8.7 sessions |
| `fortnightly` | ~10 | 11x | 1.1% | 15.4 sessions |
| `monthly` | ~21 | 6x | 0.6% | 30.6 sessions |

A hysteresis buffer (hold until a name falls out of the top 7, say) was tried
first and barely helped: 6.7% to 5.6% even with a buffer of 5. Names are not
hovering near the cutoff, they make large rank moves, so only cadence fixes it.

Like the risk rails, the cadence is **stateless**: a rebalance day is the first
trading session of a calendar period, a pure function of the date. Counting
"sessions since last rebalance" would pass a backtest and then rebalance every
day live, because `run_once` starts fresh each session with no memory of previous
runs. Holidays need no special-casing either: if Monday is shut, the week's first
session is Tuesday.

### Does it beat buy-and-hold? (walk-forward backtest)

`train` says whether the *ranking* is good. It says nothing about what acting on
that ranking costs. `evaluate` answers the harder question: it runs the same
year-by-year walk-forward, keeps each fold's model, then replays the whole
out-of-sample span through the backtester with **every year driven by the model
that never saw it**. Unlike `backtest --strategy ml`, nothing in it is in-sample.

```bash
uv run tradersjoy evaluate --horizon 5      # cadence is matched automatically
```

Four arms, 2010-2026 out-of-sample, net of 5 bps per fill:

| Arm | AUC | CAGR | vs bench | Sharpe | maxDD | Fills/yr |
|---|---|---|---|---|---|---|
| h=5 daily *(pre-fix)* | 0.510 | 23.93% | -4.46% | 0.95 | -35.9% | 858 |
| **h=5 weekly** *(shipped)* | 0.510 | **29.92%** | **+1.53%** | **1.14** | -42.3% | 241 |
| h=10 fortnightly | 0.515 | 27.11% | -1.28% | 1.03 | -46.1% | 135 |
| h=21 monthly | 0.512 | 25.96% | -2.43% | 0.97 | -45.4% | 60 |
| equal-weight buy & hold | | 28.39% | | 1.04 | -54.9% | |

**What this does establish:** the cadence fix was right, and by more than
predicted. Moving from daily to weekly is worth **+6.0 percentage points of CAGR**
out of sample. That result was predicted in advance from an independent turnover
measurement, which is the kind of confirmation worth trusting.

**What it does not establish: a tradeable edge.** Read honestly:

- The +1.53% is the **best of four arms scored on the same data**. The maximum of
  four noisy draws is positive by construction.
- AUC across arms (0.510-0.515) does not track the return ranking. The best-AUC
  arm is not the best-returning one, so the return spread is mostly noise.
- Year by year it beat the benchmark **11 times in 17** (binomial p = 0.17, not
  distinguishable from a coin flip), with annual excess ranging from **-84%** to
  **+34%**.
- The **arithmetic** mean annual excess is **negative** (-1.14%). The positive
  CAGR comes from geometry, not from picking better: the strategy loses less in
  bad years (2022: -34% vs the benchmark's -50%) and badly lags strong bull years
  (2020: +49% vs +133%). Smaller drawdowns compound better. That is a real
  property, but it is a volatility profile, not alpha.

The sober summary: the engineering is sound and the cadence fix is a genuine
improvement, while the model itself remains at coin-flip ranking skill. That is
the expected state of an honest first ML trading system, and it is the number
that should gate any decision to trade for real.

## Risk management

Any strategy can be wrapped in a risk layer that sits between it and the broker:
the strategy proposes orders, the layer rewrites them, and only the rewritten set
reaches the market. Because the wrapper is itself a `Strategy`, it runs unchanged
in backtest and live. Add `--risk` to `backtest` or `trade`:

```bash
uv run tradersjoy backtest --strategy ml --model data/models/ml.joblib --risk
uv run tradersjoy trade    --strategy buyhold --risk        # dry run, with rails
```

Four rails, all **stateless**, recomputed each day from inputs the backtest and
the live account expose identically (current quantities, the broker-reported cost
basis, and price history). That is deliberate: a trailing stop or a peak-equity
breaker would need memory the live process loses when it restarts each day, and
would then behave differently live than in the backtest. We avoid that trap.

- **Position sizing.** No single name may exceed 20% of equity; oversized buys
  are trimmed.
- **Exposure cap.** Total invested never exceeds 100% of equity, so the account's
  2x margin is structurally never used.
- **Stop-loss.** A held name trading 10% or more below its cost basis is fully
  exited (and not re-bought that day). It is checked on the close and filled at
  the next open, matching the backtester's no-look-ahead rule rather than
  pretending we can fill intraday.
- **Circuit breaker.** While SPY sits 15% or more below its 60-day high, new buys
  are blocked (exits still go through), so we stop adding risk into a crash.

An honest caveat, straight from the backtest: the rails are not free. On
buy-and-hold over 2005-2026 they cut the worst drawdown from -52% to -31%,
exactly their job, but they also roughly halved CAGR (26% to 13%) and *lowered*
risk-adjusted return (Sharpe 1.00 to 0.77), because a naive stop sells into
weakness and the breaker keeps you out of the rebound. On a basket of secular
winners, holding through drawdowns historically won. Protection has a price; the
limits are knobs, not gospel, and the right setting depends on the universe and
your tolerance for drawdown versus give-up in return.

### Risk profiles: the rails are two different things

That caveat got sharper once the rails were measured against the *actual*
strategy rather than buy-and-hold. Out-of-sample, 2010-2026, ml/weekly:

| Arm | CAGR | vs bench | Sharpe | maxDD |
|---|---|---|---|---|
| ml only | 29.81% | +1.41% | 1.14 | -42.28% |
| ml + full rails | 23.45% | **-4.94%** | 1.00 | **-47.09%** |
| ml + caps only | 29.77% | +1.38% | 1.14 | -42.28% |
| buy & hold | 28.39% | | 1.04 | -54.92% |

The reactive rails cost **6.4 points of CAGR** and 0.14 of Sharpe while making
max drawdown **4.8 points worse**. They hurt in 12 of 17 years, including 2022,
the bear market they exist for (-34% became -41%). The mechanism is
understandable: a strategy that already re-ranks and exits weak names every week
does not need a second exit rule, so a cost-basis stop mostly sells transient
weakness just before the rebound, and the circuit breaker blocks re-entry during
exactly the stretches with the best forward returns.

The structural caps are a different animal. They are nearly free (29.77% vs
29.81%, identical drawdown) because a top-5 equal-weight book sits just under
them anyway, yet they still bound what a bug, a runaway position, or accidental
leverage could do. Note also that a 10% cost-basis stop is poor insurance against
the catastrophe people imagine it covers: a name that gaps down 40% exits at 40%
down, not 10%. The position cap is the rail that actually limits that.

So the two kinds are separable via `--risk-profile`:

```bash
uv run tradersjoy trade --risk --risk-profile caps   # structural caps only
uv run tradersjoy trade --risk --risk-profile full   # everything (default)
```

The deployed configuration uses `caps`.

## Dashboard and the run journal

Every live `trade` run is recorded to a local **run journal** (a table in the
same SQLite file as the price data): the decision date, the strategy, account
equity and cash at the time, whether orders were actually placed, and the orders
themselves. Dry runs are journaled too, so the record captures what the model
*wanted* on days nothing was placed, not just the days it acted. The journal is
ours, so unlike Alpaca's own history it survives a paper-account reset. Pass
`--no-journal` to skip recording a throwaway run.

The journal exists because the trading system is otherwise stateless: `trade`
reads the account, decides, and forgets. Alpaca remembers the account; the
journal is what lets the bot remember its own decisions.

A read-only **Streamlit dashboard** reads both sources and shows them on one
page: the live account snapshot, current positions and pending orders pulled
straight from Alpaca, an equity curve built from the journal, and a decision log.

```bash
uv sync --extra dashboard        # one-time: install the dashboard dependency
uv run tradersjoy dashboard      # serve at http://localhost:8501
```

It is deliberately **read-only**: the dashboard never places or cancels an order,
so opening it to watch is always safe. The order-placing path stays in the `trade`
command alone. Early on the equity curve is a single point and fills in one
session at a time as the bot runs each day, which is honest rather than a
back-filled illusion of history.

## Unattended operation

A strategy with a 5-day horizon that nobody runs for two months is not that
strategy any more; it is one stale snapshot held by accident. `autorun` is the
scheduled counterpart to `trade`, hardened for running with nobody watching.

```bash
uv run tradersjoy autorun --strategy ml --model data/models/ml.joblib --risk
uv run tradersjoy status         # is the scheduler actually still firing?
```

`trade` assumes a human reads the output and notices when something looks wrong.
Automation removes that human, so `autorun` makes those checks explicit and
**refuses to trade** rather than trading on bad state:

| Guard | Refuses when | Why it matters |
|---|---|---|
| Lock file | another run is in flight | two traders on one account is never right |
| Halt file | `data/HALT` exists | pause without touching timers or credentials |
| Market clock | a session is in progress | today's bar is not final until the close |
| Duplicate | this session already traded | a retry must not re-decide and churn |
| Freshness | stored data misses that session | a lagging feed would re-trade a stale opinion |

Every firing writes a **heartbeat** row, including the refusals. That is the
point of the table: a scheduler that quietly stopped leaves a visible gap instead
of looking exactly like a strategy that had nothing to do. `tradersjoy status`
and the dashboard's automation panel both read it, and the run exits non-zero
only for real failures, so `systemctl status` stays meaningful.

Two schedulers are supported, both shipping as a **dry run** that walks every
guard and places nothing:

- **GitHub Actions** (recommended) - `.github/workflows/trade.yml`, weekdays at
  23:00 UTC. Free and unlimited on a public repo, and it does not need your
  machine to be awake. CI schedulers are normally too imprecise for trading, but
  here the orders queue for the next open, so a late run places the identical
  trade. The runner is ephemeral: bars are re-ingested from scratch each run
  (~340 sessions in about eight seconds, against the 201 the model needs) while
  the journal lives in its own small file that the workflow commits back. That
  daily commit doubles as the keepalive that stops GitHub auto-disabling the
  schedule after 60 idle days.
- **systemd user timer** - `deploy/`, weekdays an hour after the close. Runs
  locally; note that a user timer only fires while WSL is up.

See [deploy/README.md](deploy/README.md) for setup, the trade-offs between the
two, and what to check before switching on `--execute`.

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
| 5 | Risk management (position sizing, stops, circuit-breaker) | done |
| 6 | Run journal + Streamlit dashboard | done |
| 6b | Scheduled automation (unattended daily run) | done |
| 6c | Rebalance cadence + walk-forward backtest | done |
| 7 | Disciplined retraining loop | not started |

## Design principles

- Same `Strategy` interface runs in backtest and live, no code-path divergence.
- Broker, data source, and clock are pluggable behind interfaces.
- Walk-forward validation is the only acceptable way to evaluate a model.
- The system retrains carefully on a quarterly cadence, never on its own live
  paper-trading data.
- Never trade real money until the system has paper-traded profitably
  out-of-sample for at least 6 months.
