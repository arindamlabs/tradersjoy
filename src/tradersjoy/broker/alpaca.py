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
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from tradersjoy.config import get_settings
from tradersjoy.core.types import Side

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import date

    from tradersjoy.core.types import Order

#: The exchange's own timezone. US market hours (and the DST shifts that move
#: them) are defined in New York time, so every session boundary is resolved
#: here rather than in whatever timezone the machine running the bot happens to
#: sit in.
MARKET_TZ = ZoneInfo("America/New_York")


def _session_close(session) -> datetime:  # noqa: ANN001 - Alpaca's Calendar model
    """Return an exchange-local, timezone-aware close instant for a calendar row.

    Alpaca reports session open/close as naive datetimes in exchange time; this
    attaches that timezone so the value can be compared against ``now`` safely.
    """
    close = session.close
    if not isinstance(close, datetime):
        # Older payloads carry a bare time; pair it with the session's date.
        close = datetime.combine(session.date, close)
    return close.replace(tzinfo=MARKET_TZ) if close.tzinfo is None else close


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

    def is_trading_day(self, day: date) -> bool:
        """Return whether ``day`` is a regular US equity session.

        Asks Alpaca's own calendar rather than guessing from the weekday, so
        market holidays (which no amount of ``Mon-Fri`` cron syntax can predict)
        are excluded correctly.

        Args:
            day: Calendar date to test.

        Returns:
            True if the exchange holds a session on ``day``.
        """
        from alpaca.trading.requests import GetCalendarRequest

        sessions = self._client.get_calendar(GetCalendarRequest(start=day, end=day))
        return any(s.date == day for s in sessions)

    def last_completed_session(self, now: datetime | None = None) -> date | None:
        """Return the most recent session whose closing bell has already rung.

        This is the session a strategy may legitimately decide on. Deriving it
        from the exchange calendar rather than from "today" gets every awkward
        case right for free: on a Saturday it is Friday, on a holiday it is the
        previous business day, and before today's close it is yesterday. The
        unattended runner then insists the stored data actually reaches this
        day before it will trade.

        Args:
            now: Instant to evaluate against, for testing. Defaults to the
                current time in the exchange's timezone.

        Returns:
            The date of the latest closed session, or ``None`` if the calendar
            returned nothing for the lookback window.
        """
        from alpaca.trading.requests import GetCalendarRequest

        moment = now or datetime.now(MARKET_TZ)
        if moment.tzinfo is None:
            moment = moment.replace(tzinfo=MARKET_TZ)
        today = moment.astimezone(MARKET_TZ).date()

        # Ten calendar days comfortably spans any weekend-plus-holiday stretch.
        sessions = self._client.get_calendar(
            GetCalendarRequest(start=today - timedelta(days=10), end=today)
        )
        closed = [s.date for s in sessions if _session_close(s) <= moment]
        return max(closed) if closed else None

    def is_market_open(self) -> bool:
        """Return whether the exchange is trading *right now*.

        The unattended runner uses this to refuse to decide while a session is
        still in progress: today's bar is not final until the close, and a
        strategy fed a half-formed bar is deciding on data that will change.

        Returns:
            True if the market is currently open.
        """
        return bool(self._client.get_clock().is_open)

    def _open_orders(self) -> list:
        """Return Alpaca's currently open orders (queued/partially filled)."""
        try:
            from alpaca.trading.enums import QueryOrderStatus
            from alpaca.trading.requests import GetOrdersRequest

            return self._client.get_orders(
                GetOrdersRequest(status=QueryOrderStatus.OPEN)
            )
        except Exception:  # noqa: BLE001 - any failure falls back to the default list
            return self._client.get_orders()

    def _open_order_symbols(self) -> set[str]:
        """Return the set of tickers that currently have an open order."""
        return {o.symbol for o in self._open_orders()}

    def positions_detail(self) -> list[dict]:
        """Return rich per-position data for display (not used by strategies).

        Unlike :meth:`get_account`, which exposes only what a strategy reads,
        this keeps the market value and unrealised P/L Alpaca reports, for the
        dashboard's positions table.

        Returns:
            One dict per open position with ticker, share quantity, average cost,
            current price, market value, and unrealised P/L (dollar and percent).
        """
        out: list[dict] = []
        for p in self._client.get_all_positions():
            out.append(
                {
                    "ticker": p.symbol,
                    "qty": float(p.qty),
                    "avg_cost": float(p.avg_entry_price),
                    "price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl),
                    "unrealized_plpc": float(p.unrealized_plpc),
                }
            )
        return out

    def open_orders(self) -> list[dict]:
        """Return the currently open (pending) orders for display.

        Returns:
            One dict per open order with ticker, side, quantity, and status.
            Empty when nothing is queued.
        """
        out: list[dict] = []
        for o in self._open_orders():
            out.append(
                {
                    "ticker": o.symbol,
                    "side": getattr(o.side, "value", str(o.side)),
                    "qty": float(o.qty) if o.qty is not None else 0.0,
                    "status": getattr(o.status, "value", str(o.status)),
                }
            )
        return out

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
