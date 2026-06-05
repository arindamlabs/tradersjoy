"""Live order execution and account reads against the Alpaca paper API.

This is the live counterpart to :class:`~tradersjoy.broker.sim.SimBroker`. Where
the simulator invents fills, here Alpaca's real (paper) matching engine does, so
this module's job is only to (a) report the current account as an
:class:`AccountView` the strategy can read and (b) translate the strategy's
:class:`~tradersjoy.core.types.Order` objects into Alpaca market orders.

Three deliberate guardrails keep this honest and safe:

- **Paper only.** The client is pinned to ``paper=True``; placing real-money
  orders is intentionally not reachable from here.
- **Whole shares only.** Order quantities are floored to whole shares, dodging
  the constraints Alpaca places on fractional orders. Backtests keep fractional
  sizing, so live fills can differ slightly from a backtest; that gap is the
  price of this simplicity and is documented rather than hidden.
- **No duplicate orders.** Before submitting, any ticker that already has an
  open order at Alpaca is skipped, so an accidental re-run cannot double up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from tradersjoy.config import get_settings
from tradersjoy.core.types import Side

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tradersjoy.core.types import Order


@dataclass(frozen=True, slots=True)
class OrderPlan:
    """How one strategy order maps to a concrete whole-share live order.

    Attributes:
        ticker: Symbol to trade.
        side: Buy or sell.
        requested_qty: The fractional quantity the strategy actually asked for.
        shares: The whole-share quantity that will be sent (floored).
        note: Non-empty when the order will be skipped, explaining why.
    """

    ticker: str
    side: Side
    requested_qty: float
    shares: int
    note: str = ""


def plan_whole_share_orders(orders: Sequence[Order]) -> list[OrderPlan]:
    """Translate fractional strategy orders into whole-share live order plans.

    Pure and side-effect-free so it can drive both the dry-run preview and the
    real submission, and be unit-tested without touching Alpaca. Quantities are
    floored toward zero so a buy never overshoots the intended dollar amount; an
    order that floors to zero shares is marked to be skipped.

    Args:
        orders: The orders a strategy produced.

    Returns:
        One :class:`OrderPlan` per input order, in order.
    """
    plans: list[OrderPlan] = []
    for o in orders:
        shares = int(o.quantity)  # floor toward zero; never over-buy
        note = "" if shares > 0 else "rounds to <1 share; skipped"
        plans.append(OrderPlan(o.ticker, o.side, o.quantity, shares, note))
    return plans


class AlpacaAccount:
    """A point-in-time snapshot of the paper account, as an ``AccountView``.

    Satisfies :class:`~tradersjoy.strategy.base.AccountView` so a strategy reads
    it exactly as it would read a backtest portfolio.

    Attributes:
        equity: Total account value (cash plus marked positions).
        cash: Uninvested cash available.
    """

    def __init__(
        self,
        equity: float,
        cash: float,
        positions: dict[str, float],
        avg_costs: dict[str, float] | None = None,
    ) -> None:
        """Build a snapshot from already-fetched account values.

        Args:
            equity: Total account value.
            cash: Available cash.
            positions: Held share quantity keyed by ticker.
            avg_costs: Average entry price per share keyed by ticker, for a
                cost-basis stop-loss. Defaults to empty (stops simply skip names
                whose cost basis is unknown).
        """
        self.equity = equity
        self.cash = cash
        self._positions = positions
        self._avg_costs = avg_costs or {}

    def qty(self, ticker: str) -> float:
        """Shares currently held in ``ticker`` (``0.0`` if none)."""
        return self._positions.get(ticker, 0.0)

    def avg_cost(self, ticker: str) -> float:
        """Average entry price per share in ``ticker`` (``0.0`` if none)."""
        return self._avg_costs.get(ticker, 0.0)


class AlpacaBroker:
    """Reads the paper account and places whole-share market orders on it.

    The constructor pins the underlying client to the paper endpoint. Network
    calls happen only when :meth:`get_account` or :meth:`submit` are invoked, so
    constructing a broker is cheap and offline.
    """

    def __init__(self, api_key: str | None = None, secret_key: str | None = None) -> None:
        """Create a paper-only trading client.

        Args:
            api_key: Alpaca key ID. Defaults to the configured ``ALPACA_API_KEY``.
            secret_key: Alpaca secret. Defaults to ``ALPACA_API_SECRET``.
        """
        settings = get_settings()
        self._client = TradingClient(
            api_key or settings.alpaca_api_key,
            secret_key or settings.alpaca_api_secret,
            paper=True,
        )

    def get_account(self) -> AlpacaAccount:
        """Fetch the current account and open positions as a snapshot.

        Returns:
            An :class:`AlpacaAccount` with live equity, cash, and per-ticker
            share quantities.
        """
        acct = self._client.get_account()
        live_positions = self._client.get_all_positions()
        positions = {p.symbol: float(p.qty) for p in live_positions}
        avg_costs = {p.symbol: float(p.avg_entry_price) for p in live_positions}
        return AlpacaAccount(
            float(acct.equity), float(acct.cash), positions, avg_costs
        )

    def _open_order_symbols(self) -> set[str]:
        """Return the set of tickers that currently have an open order."""
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            orders = self._client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
        except Exception:  # noqa: BLE001 - any failure falls back to the default list
            orders = self._client.get_orders()
        return {o.symbol for o in orders}

    def submit(self, orders: Sequence[Order]) -> list[str]:
        """Place whole-share market orders for ``orders``, skipping unsafe ones.

        Each order is floored to whole shares; orders that round to zero, or
        whose ticker already has an open order, are skipped. Surviving orders are
        sent as ``DAY`` market orders, which queue for the next open if the
        market is currently closed.

        Args:
            orders: Orders a strategy produced this run.

        Returns:
            One human-readable result line per input order describing what was
            placed or why it was skipped.
        """
        pending = self._open_order_symbols()
        results: list[str] = []
        for plan in plan_whole_share_orders(orders):
            if plan.shares <= 0:
                results.append(f"skipped {plan.ticker}: {plan.note}")
                continue
            if plan.ticker in pending:
                results.append(f"skipped {plan.ticker}: an open order already exists")
                continue
            side = OrderSide.BUY if plan.side is Side.BUY else OrderSide.SELL
            self._client.submit_order(
                MarketOrderRequest(
                    symbol=plan.ticker,
                    qty=plan.shares,
                    side=side,
                    time_in_force=TimeInForce.DAY,
                )
            )
            results.append(f"placed {plan.side} {plan.shares} {plan.ticker}")
        return results
