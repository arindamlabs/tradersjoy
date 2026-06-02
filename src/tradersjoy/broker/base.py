"""The broker interface shared by the simulator and (later) live trading.

A broker is the only component that turns an :class:`~tradersjoy.core.types.Order`
into a real position change. Keeping it behind one small interface is what lets
the same strategy and engine run unchanged in a backtest and against the live
Alpaca paper account in Phase 3: only the concrete ``Broker`` swaps out.

The interface is deliberately two-step to model the no-look-ahead timing the
engine relies on:

1. ``submit`` receives the orders a strategy decided at day T's close. The
   broker only queues them; nothing fills yet.
2. ``settle`` is called on the *next* trading day and produces the fills for
   those queued orders, priced from that new day's bar.

A strategy therefore can never trade at a price it has already seen.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradersjoy.backtest.portfolio import Portfolio
    from tradersjoy.core.types import Bar, Fill, Order


class Broker(ABC):
    """Accepts orders and, one step later, reports the resulting fills.

    Implementations must honour the submit-then-settle split: orders handed to
    :meth:`submit` must not affect the portfolio until a subsequent
    :meth:`settle` call prices and applies them.
    """

    @abstractmethod
    def submit(self, orders: list[Order]) -> None:
        """Queue orders decided on the current bar for execution next bar.

        Args:
            orders: Orders a strategy produced from the latest close. May be
                empty. They take effect only on the following :meth:`settle`.
        """

    @abstractmethod
    def settle(
        self, day: date, bars: dict[str, Bar], portfolio: Portfolio
    ) -> list[Fill]:
        """Execute previously submitted orders against ``day``'s bars.

        Prices each pending order from the new day's bar, applies it to
        ``portfolio``, and returns the fills produced. Orders that cannot be
        priced or afforded are handled by the implementation (the simulator
        rejects them) and simply produce no fill.

        Args:
            day: The trading day now being entered.
            bars: That day's bars keyed by ticker. An order whose ticker is
                absent cannot be priced on this day.
            portfolio: Portfolio to apply accepted fills to, in place.

        Returns:
            The fills executed on ``day``, in submission order. Empty if nothing
            was pending or nothing could be filled.
        """
