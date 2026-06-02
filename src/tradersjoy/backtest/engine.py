"""The backtest event loop: replay history one day at a time, honestly.

The loop's ordering is the whole point, because it is what keeps a backtest from
quietly cheating. For each trading day, in this exact order:

1. **Settle.** Fill any orders the strategy submitted *yesterday*, priced at
   today's open (with slippage). This is where money actually moves.
2. **Mark.** Value the portfolio at today's close and append an equity point.
3. **Decide.** Let the strategy look at everything through today's close and
   submit orders, which the broker will only fill at tomorrow's open.

So a strategy never trades at a price it has already seen, and the equity curve
is marked on closes the strategy has, by then, been allowed to act on only for
*future* days. Orders submitted on the final day are never filled, which is
correct: there is no next open to fill them at.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from tradersjoy.backtest.data import BarHistory
from tradersjoy.backtest.metrics import Metrics, compute_metrics
from tradersjoy.backtest.portfolio import Portfolio
from tradersjoy.broker.base import Broker
from tradersjoy.core.types import Fill, Trade
from tradersjoy.strategy.base import BarContext, Strategy


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Everything produced by one run, plus its computed scorecard.

    Attributes:
        strategy_name: Name of the strategy that was run.
        tickers: The universe the run covered.
        equity_curve: ``(day, equity)`` points across the run.
        trades: Closed round trips, in the order they closed.
        fills: Every fill executed, in order.
        metrics: The summary :class:`~tradersjoy.backtest.metrics.Metrics`.
    """

    strategy_name: str
    tickers: list[str]
    equity_curve: list[tuple[date, float]]
    trades: list[Trade]
    fills: list[Fill]
    metrics: Metrics

    def summary(self) -> str:
        """Render a human-readable multi-line report of the run's results."""
        m = self.metrics
        span = (
            f"{m.start_day.isoformat()} -> {m.end_day.isoformat()}"
            if m.start_day and m.end_day
            else "no data"
        )
        lines = [
            f"Strategy:       {self.strategy_name}",
            f"Universe:       {', '.join(self.tickers)}",
            f"Period:         {span}  ({m.num_days} calendar days)",
            f"Start equity:   ${m.start_equity:,.2f}",
            f"Final equity:   ${m.final_equity:,.2f}",
            f"Total return:   {m.total_return * 100:+.2f}%",
            f"CAGR:           {m.cagr * 100:+.2f}%",
            f"Sharpe (ann.):  {m.sharpe:.2f}",
            f"Max drawdown:   {m.max_drawdown * 100:.2f}%",
            f"Closed trades:  {m.num_trades}",
            f"Hit rate:       {m.hit_rate * 100:.1f}%",
            f"Realized P&L:   ${m.realized_pnl:,.2f}",
        ]
        return "\n".join(lines)


def run_backtest(
    strategy: Strategy,
    data: BarHistory,
    broker: Broker,
    starting_cash: float,
    start: date | None = None,
    end: date | None = None,
) -> BacktestResult:
    """Replay ``data`` through ``strategy`` and ``broker``, returning the result.

    Args:
        strategy: The strategy to drive. Its ``on_bar`` is called once per day
            with data through that day's close.
        data: The loaded bar panel to replay.
        broker: Execution model (typically a
            :class:`~tradersjoy.broker.sim.SimBroker`).
        starting_cash: Opening cash balance for the portfolio.
        start: First day to simulate (inclusive), or ``None`` for the panel's
            earliest day.
        end: Last day to simulate (inclusive), or ``None`` for the panel's
            latest day.

    Returns:
        A :class:`BacktestResult` with the equity curve, trades, fills, and
        metrics.
    """
    portfolio = Portfolio(starting_cash)
    days = [
        d
        for d in data.trading_days
        if (start is None or d >= start) and (end is None or d <= end)
    ]

    for day in days:
        bars = data.bars_on(day)
        # 1. Fill yesterday's orders at today's open.
        broker.settle(day, bars, portfolio)
        # 2. Mark the book at today's close.
        closes = {ticker: bar.close for ticker, bar in bars.items()}
        portfolio.mark(day, closes)
        # 3. Let the strategy decide; orders fill at tomorrow's open.
        ctx = BarContext(day=day, bars=bars, history=data, portfolio=portfolio)
        broker.submit(strategy.on_bar(ctx))

    return BacktestResult(
        strategy_name=strategy.name,
        tickers=data.tickers,
        equity_curve=portfolio.equity_curve,
        trades=portfolio.trades,
        fills=portfolio.fills,
        metrics=compute_metrics(portfolio.equity_curve, portfolio.trades),
    )
