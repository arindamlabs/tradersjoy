"""The strategy interface every trading rule implements, baseline or ML.

A strategy is a pure decision function: given the market and portfolio state as
of a day's close, return the orders to place. It owns *signals and sizing only*,
never execution or accounting, which keeps the same strategy object usable
unchanged in a backtest and in live paper trading (Phase 3) where the only thing
that differs is the concrete broker behind it.

Strategies are handed a :class:`BarContext` and must treat it as read-only. In
particular they must read history through :class:`~tradersjoy.backtest.data.BarHistory`,
which only exposes bars up to the current day, so it is structurally impossible
for a strategy to look ahead.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradersjoy.backtest.data import BarHistory
    from tradersjoy.backtest.portfolio import Portfolio
    from tradersjoy.core.types import Bar, Order


@dataclass(frozen=True, slots=True)
class BarContext:
    """The read-only snapshot a strategy sees when deciding on one day.

    Attributes:
        day: The trading day being decided, i.e. "now". Its close is known.
        bars: Today's bar for each ticker that traded, keyed by ticker. A ticker
            absent here did not trade today (pre-IPO, halted, or delisted).
        history: Access to all bars up to and including ``day`` for indicators.
        portfolio: Current portfolio. Read it (``portfolio.equity``,
            ``portfolio.qty(ticker)``) to size orders; never mutate it.
    """

    day: date
    bars: dict[str, Bar]
    history: BarHistory
    portfolio: Portfolio


class Strategy(ABC):
    """Base class for all strategies: turn a :class:`BarContext` into orders."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for the strategy, used in reports and the CLI."""

    @abstractmethod
    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Decide the orders to place given everything known as of ``ctx.day``.

        Called once per trading day after the portfolio has been marked at that
        day's close. Returned orders are submitted to the broker and fill at the
        next session's open, so sizing should use ``ctx`` prices as estimates,
        not guarantees.

        Args:
            ctx: The read-only market-and-portfolio snapshot for today.

        Returns:
            Orders to submit. Return an empty list to do nothing this day.
        """
