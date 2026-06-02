"""SMA crossover: a classic trend-following baseline, long-only.

For each ticker it tracks two simple moving averages of the adjusted close, a
fast one and a slow one. When the fast average is above the slow one the trend is
deemed up and the strategy holds; when it crosses below, it exits to cash. It is
a useful baseline precisely because it is so well known and so often *fails* to
beat buy-and-hold after slippage: a sober first lesson in how a plausible rule
can churn costs without adding return.

Signals use the adjusted close (continuous across splits/dividends); order sizing
uses the raw close, since that is closer to what will actually print at the next
open. This split is exactly why a :class:`~tradersjoy.core.types.Bar` keeps both
prices.
"""

from __future__ import annotations

from tradersjoy.core.types import Order, Side
from tradersjoy.strategy.base import BarContext, Strategy


class SMACrossover(Strategy):
    """Long-only fast/slow simple-moving-average crossover, one rule per ticker.

    Attributes:
        tickers: Symbols to trade independently.
        short_window: Look-back length of the fast moving average, in sessions.
        long_window: Look-back length of the slow moving average, in sessions.
        invest_fraction: Fraction of current equity to allocate per held name on
            entry, leaving a buffer against next-open slippage.
    """

    def __init__(
        self,
        tickers: list[str],
        short_window: int = 20,
        long_window: int = 50,
        invest_fraction: float = 0.95,
    ) -> None:
        """Configure the universe and the two moving-average look-backs.

        Args:
            tickers: Symbols to trade.
            short_window: Fast SMA length in sessions. Must be < ``long_window``.
            long_window: Slow SMA length in sessions.
            invest_fraction: Fraction of equity to spread across the tickers when
                sizing entries. Defaults to ``0.95``.

        Raises:
            ValueError: If ``short_window`` is not strictly less than
                ``long_window`` (the rule would be meaningless otherwise).
        """
        if short_window >= long_window:
            raise ValueError(
                f"short_window ({short_window}) must be < long_window ({long_window})"
            )
        self.tickers = tickers
        self.short_window = short_window
        self.long_window = long_window
        self.invest_fraction = invest_fraction

    @property
    def name(self) -> str:
        return f"sma({self.short_window}/{self.long_window})"

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Enter on a fast-above-slow cross, exit fully on the reverse cross."""
        orders: list[Order] = []
        per_name = ctx.portfolio.equity * self.invest_fraction / len(self.tickers)
        for ticker in self.tickers:
            bar = ctx.bars.get(ticker)
            if bar is None:
                continue  # not trading today; cannot price an order
            closes = ctx.history.adj_closes(ticker, ctx.day)
            if len(closes) < self.long_window:
                continue  # not enough history for the slow average yet
            short_ma = sum(closes[-self.short_window :]) / self.short_window
            long_ma = sum(closes[-self.long_window :]) / self.long_window
            holding = ctx.portfolio.qty(ticker)

            if short_ma > long_ma and holding == 0:
                qty = per_name / bar.close
                if qty > 0:
                    orders.append(Order(ticker, Side.BUY, qty, tag="sma-entry"))
            elif short_ma < long_ma and holding > 0:
                orders.append(Order(ticker, Side.SELL, holding, tag="sma-exit"))
        return orders
