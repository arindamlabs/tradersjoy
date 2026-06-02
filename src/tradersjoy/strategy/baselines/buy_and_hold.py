"""Buy-and-hold: the benchmark every active strategy must actually beat.

It buys an equal dollar slice of each ticker the first day that ticker trades,
then never trades again. Cheap, tax-efficient, and historically very hard to
beat after costs, which is exactly why it is the honest yardstick: a clever
strategy that underperforms buy-and-hold has, for all its activity, destroyed
value.

The equal slices are sized off the *initial* equity captured on the first call,
so tickers that IPO mid-backtest still receive a comparable nominal allocation
when they appear (staggered entry), rather than being sized off a by-then-larger
or -smaller book.
"""

from __future__ import annotations

from tradersjoy.core.types import Order, Side
from tradersjoy.strategy.base import BarContext, Strategy


class BuyAndHold(Strategy):
    """Equal-weight buy-and-hold across a fixed set of tickers.

    Attributes:
        tickers: Symbols to hold.
        invest_fraction: Fraction of starting equity to deploy in total, the
            remainder kept as a cash buffer so next-open slippage does not push
            a buy past available cash and get it rejected.
    """

    def __init__(self, tickers: list[str], invest_fraction: float = 0.98) -> None:
        """Configure the universe and how much of equity to deploy.

        Args:
            tickers: Symbols to buy and hold.
            invest_fraction: Total fraction of starting equity to invest across
                all tickers. Defaults to ``0.98`` to leave a small cash buffer.
        """
        self.tickers = tickers
        self.invest_fraction = invest_fraction
        self._start_equity: float | None = None
        self._bought: set[str] = set()

    @property
    def name(self) -> str:
        return "buyhold"

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Buy each not-yet-held ticker the first day it trades, then hold."""
        if self._start_equity is None:
            self._start_equity = ctx.portfolio.equity
        if len(self._bought) == len(self.tickers):
            return []  # fully invested; nothing left to do, ever

        target_each = self._start_equity * self.invest_fraction / len(self.tickers)
        orders: list[Order] = []
        for ticker in self.tickers:
            if ticker in self._bought:
                continue
            bar = ctx.bars.get(ticker)
            if bar is None:
                continue  # not trading yet; wait for its first session
            qty = target_each / bar.close
            if qty > 0:
                orders.append(Order(ticker, Side.BUY, qty, tag="buy-and-hold"))
                self._bought.add(ticker)
        return orders
