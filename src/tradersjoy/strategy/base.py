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
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from tradersjoy.backtest.data import BarHistory
    from tradersjoy.core.types import Bar, Order


@runtime_checkable
class AccountView(Protocol):
    """The read-only slice of account state a strategy is allowed to see.

    This is the seam that lets one strategy run unchanged in a backtest and in
    live trading. In a backtest the concrete object is a
    :class:`~tradersjoy.backtest.portfolio.Portfolio`; live, it is an
    :class:`~tradersjoy.broker.alpaca.AlpacaAccount` reflecting the real paper
    account. A strategy depends only on this narrow contract, so it cannot tell
    (or care) which one it holds, and cannot reach execution or order history.
    """

    @property
    def equity(self) -> float:
        """Total account value: cash plus the marked value of open positions."""
        ...

    @property
    def cash(self) -> float:
        """Uninvested cash currently available."""
        ...

    def qty(self, ticker: str) -> float:
        """Shares currently held in ``ticker`` (``0.0`` if none)."""
        ...

    def avg_cost(self, ticker: str) -> float:
        """Average price paid per share for the open position (``0.0`` if none).

        Reported by the broker (live) or the portfolio (backtest), so a
        cost-basis stop-loss can be evaluated statelessly, identically in both,
        without the strategy having to remember its own entry prices.
        """
        ...


@dataclass(frozen=True, slots=True)
class BarContext:
    """The read-only snapshot a strategy sees when deciding on one day.

    Attributes:
        day: The trading day being decided, i.e. "now". Its close is known.
        bars: Today's bar for each ticker that traded, keyed by ticker. A ticker
            absent here did not trade today (pre-IPO, halted, or delisted).
        history: Access to all bars up to and including ``day`` for indicators.
        portfolio: Read-only account state (an :class:`AccountView`). Read it
            (``portfolio.equity``, ``portfolio.qty(ticker)``) to size orders;
            never mutate it. Backed by a simulated portfolio in a backtest and
            the live Alpaca account in live trading.
    """

    day: date
    bars: dict[str, Bar]
    history: BarHistory
    portfolio: AccountView


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
