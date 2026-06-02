"""Performance metrics computed from an equity curve and the list of trades.

These are the standard yardsticks for judging a strategy. None of them is
sufficient alone, which is the point of reporting several: a high total return
with a brutal drawdown and a coin-flip hit rate is a very different thing from a
steadier curve with the same return.

Conventions used here:

- A trading year is 252 sessions; Sharpe is annualised by ``sqrt(252)``.
- The risk-free rate is assumed zero, so Sharpe is mean daily return over its
  standard deviation, annualised. This flatters Sharpe slightly in a positive-
  rate environment but keeps the metric dependency-free and easy to reason about.
- CAGR uses actual calendar time between the first and last equity points.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from datetime import date

from tradersjoy.core.types import Trade

#: Trading sessions per year, used to annualise the Sharpe ratio.
_TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class Metrics:
    """A summary scorecard for one backtest run.

    Attributes:
        start_equity: Equity at the first marked day (the starting cash).
        final_equity: Equity at the last marked day.
        total_return: Final over start minus one (e.g. ``0.42`` = +42%).
        cagr: Compound annual growth rate over the run's calendar span.
        sharpe: Annualised Sharpe ratio of daily returns (risk-free rate 0).
        max_drawdown: Worst peak-to-trough equity decline, as a negative
            fraction (e.g. ``-0.30`` = a 30% drawdown).
        hit_rate: Fraction of closed trades that were profitable.
        num_trades: Number of closed round trips.
        realized_pnl: Sum of realised profit and loss across closed trades.
        start_day: First day on the equity curve.
        end_day: Last day on the equity curve.
        num_days: Calendar days between ``start_day`` and ``end_day``.
    """

    start_equity: float
    final_equity: float
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    num_trades: int
    realized_pnl: float
    start_day: date | None
    end_day: date | None
    num_days: int


def _max_drawdown(values: list[float]) -> float:
    """Return the worst peak-to-trough decline in an equity series.

    Args:
        values: Equity values in chronological order.

    Returns:
        The minimum of ``value / running_peak - 1`` over the series, as a
        non-positive fraction; ``0.0`` if the series never declined.
    """
    peak = -math.inf
    worst = 0.0
    for v in values:
        peak = max(peak, v)
        if peak > 0:
            worst = min(worst, v / peak - 1.0)
    return worst


def compute_metrics(
    equity_curve: list[tuple[date, float]], trades: list[Trade]
) -> Metrics:
    """Reduce an equity curve and trade log into a :class:`Metrics` scorecard.

    Args:
        equity_curve: ``(day, equity)`` points in chronological order.
        trades: Closed round trips produced during the run.

    Returns:
        A populated :class:`Metrics`. With fewer than two equity points the
        return/risk fields are zero (nothing to measure), though trade-based
        fields are still reported.
    """
    num_trades = len(trades)
    wins = sum(1 for t in trades if t.pnl > 0)
    hit_rate = wins / num_trades if num_trades else 0.0
    realized_pnl = sum(t.pnl for t in trades)

    if len(equity_curve) < 2:
        start_eq = equity_curve[0][1] if equity_curve else 0.0
        day0 = equity_curve[0][0] if equity_curve else None
        return Metrics(
            start_equity=start_eq,
            final_equity=start_eq,
            total_return=0.0,
            cagr=0.0,
            sharpe=0.0,
            max_drawdown=0.0,
            hit_rate=hit_rate,
            num_trades=num_trades,
            realized_pnl=realized_pnl,
            start_day=day0,
            end_day=day0,
            num_days=0,
        )

    days = [d for d, _ in equity_curve]
    values = [v for _, v in equity_curve]
    start_eq, final_eq = values[0], values[-1]
    total_return = final_eq / start_eq - 1.0 if start_eq > 0 else 0.0

    span_days = (days[-1] - days[0]).days
    years = span_days / 365.25
    if years > 0 and start_eq > 0 and final_eq > 0:
        cagr = (final_eq / start_eq) ** (1.0 / years) - 1.0
    else:
        cagr = 0.0

    daily_returns = [
        values[i] / values[i - 1] - 1.0
        for i in range(1, len(values))
        if values[i - 1] > 0
    ]
    if len(daily_returns) > 1:
        sd = statistics.stdev(daily_returns)
        mean = statistics.fmean(daily_returns)
        sharpe = (mean / sd) * math.sqrt(_TRADING_DAYS) if sd > 0 else 0.0
    else:
        sharpe = 0.0

    return Metrics(
        start_equity=start_eq,
        final_equity=final_eq,
        total_return=total_return,
        cagr=cagr,
        sharpe=sharpe,
        max_drawdown=_max_drawdown(values),
        hit_rate=hit_rate,
        num_trades=num_trades,
        realized_pnl=realized_pnl,
        start_day=days[0],
        end_day=days[-1],
        num_days=span_days,
    )
