"""A simulated broker that fills market orders at the next bar's open.

``SimBroker`` is the execution model used in backtests. It deliberately makes
two pessimistic, explicit assumptions so that simulated results lean
conservative rather than flattering:

- **Next-open fills.** Orders submitted from day T's close fill at day T+1's
  *open*, never at the close the strategy already observed. The engine drives
  this by calling :meth:`submit` then, on the following day, :meth:`settle`.
- **Adverse slippage.** Every fill moves against the trader by a fixed number
  of basis points: buys fill slightly above the open, sells slightly below.
  Real fills are uncertain; this is a stand-in, not a measurement, and it is a
  constructor knob so its effect can be stress-tested.

The simulator is long-only and never uses leverage: a buy it cannot fund from
cash, or a sell larger than the held position, is rejected and recorded rather
than silently clipped.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from tradersjoy.broker.base import Broker
from tradersjoy.core.types import Bar, Fill, Order, Side

if TYPE_CHECKING:
    from tradersjoy.backtest.portfolio import Portfolio

#: Cash/quantity tolerance so floating-point dust never spuriously rejects an
#: order that is affordable or coverable to the last cent/share.
_EPS = 1e-9


class SimBroker(Broker):
    """Fills queued market orders at the next session's open, with slippage.

    Attributes:
        slippage_bps: Adverse slippage in basis points applied to every fill
            (``5.0`` means 0.05%). Buys pay ``open * (1 + bps/10000)``; sells
            receive ``open * (1 - bps/10000)``.
        commission_per_share: Flat commission charged per share filled.
        commission_per_order: Flat commission charged once per filled order.
        rejections: Running log of ``(order, reason)`` pairs for orders that
            could not be filled (unpriceable, unaffordable, or uncovered).
    """

    def __init__(
        self,
        slippage_bps: float = 5.0,
        commission_per_share: float = 0.0,
        commission_per_order: float = 0.0,
    ) -> None:
        """Configure the simulated execution assumptions.

        Args:
            slippage_bps: Adverse slippage applied to each fill, in basis
                points. Defaults to ``5.0`` (0.05%).
            commission_per_share: Per-share commission. Defaults to ``0.0``
                (Alpaca is commission-free for US equities).
            commission_per_order: Per-order commission. Defaults to ``0.0``.
        """
        self.slippage_bps = slippage_bps
        self.commission_per_share = commission_per_share
        self.commission_per_order = commission_per_order
        self.rejections: list[tuple[Order, str]] = []
        self._pending: list[Order] = []

    def submit(self, orders: list[Order]) -> None:
        """Queue orders to be filled on the next :meth:`settle`."""
        self._pending.extend(orders)

    def settle(
        self, day: date, bars: dict[str, Bar], portfolio: Portfolio
    ) -> list[Fill]:
        """Fill every pending order priced from ``day``'s open, where possible.

        An order whose ticker has no bar on ``day`` stays pending and is retried
        on the next session (e.g. a trading halt). Orders that are unaffordable
        or uncovered are dropped and appended to :attr:`rejections`.
        """
        fills: list[Fill] = []
        still_pending: list[Order] = []
        for order in self._pending:
            bar = bars.get(order.ticker)
            if bar is None:
                still_pending.append(order)  # no price today; try again tomorrow
                continue
            fill = self._try_fill(order, bar, day, portfolio)
            if fill is not None:
                portfolio.apply_fill(fill)
                fills.append(fill)
        self._pending = still_pending
        return fills

    def _try_fill(
        self, order: Order, bar: Bar, day: date, portfolio: Portfolio
    ) -> Fill | None:
        """Price one order and check it is fundable, returning a fill or ``None``.

        Returns ``None`` (and records a rejection) when a buy exceeds available
        cash or a sell exceeds the held quantity. Pricing uses the raw open plus
        adverse slippage; the simulator never trades at the adjusted close.
        """
        slip = self.slippage_bps / 10_000.0
        # Slippage is always adverse: a buy pays up, a sell receives less.
        price = bar.open * (1.0 + slip if order.side is Side.BUY else 1.0 - slip)
        commission = self.commission_per_order + self.commission_per_share * order.quantity

        if order.side is Side.BUY:
            cost = order.quantity * price + commission
            if cost > portfolio.cash + _EPS:
                self.rejections.append((order, f"insufficient cash: need {cost:.2f}"))
                return None
        else:
            held = portfolio.qty(order.ticker)
            if order.quantity > held + _EPS:
                self.rejections.append(
                    (order, f"insufficient shares: have {held:.4f}")
                )
                return None

        return Fill(
            ticker=order.ticker,
            day=day,
            side=order.side,
            quantity=order.quantity,
            price=price,
            commission=commission,
        )
