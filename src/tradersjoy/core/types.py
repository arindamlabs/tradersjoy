"""Core domain types shared across the system.

These are deliberately plain and storage-agnostic. The SQLAlchemy models in
:mod:`tradersjoy.data.store` map to and from these; strategies, the backtester,
and the live engine speak only in these types so nothing downstream is coupled
to how we happen to persist data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


@dataclass(frozen=True, slots=True)
class Bar:
    """A single daily OHLCV bar for one ticker.

    "OHLCV" is the standard summary of one trading day: the opening, highest,
    lowest, and closing prices, plus the volume (number of shares) traded.

    The class is ``frozen`` (immutable) and ``slots``-based: a bar is a fact
    about the past that should never be mutated in place, and slots keep these
    objects small since we hold many of them in memory during a backtest.

    Why two closing prices:
        ``close`` is the raw price that actually printed at the end of the day,
        which is what a realistic order simulation must use. ``adj_close`` is the
        same close retroactively adjusted for stock splits and dividends. A 2-for-1
        split halves the raw price overnight without any real loss to a holder, so
        return and signal math must use the adjusted series to avoid seeing a
        phantom -50% move. We store both: raw for execution, adjusted for analysis.

    Attributes:
        ticker: Symbol the bar belongs to, e.g. ``"AAPL"``. Upper-cased on ingest.
        day: Calendar date of the trading session.
        open: Raw price at the session open.
        high: Highest raw price during the session.
        low: Lowest raw price during the session.
        close: Raw price at the session close (used for order simulation).
        adj_close: Split- and dividend-adjusted close (used for returns/signals).
        volume: Shares traded during the session.
        source: Name of the :class:`~tradersjoy.data.sources.base.DataSource`
            that produced this bar (e.g. ``"yfinance"``), kept for provenance so
            we can tell where a given row came from.
    """

    ticker: str
    day: date
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int
    source: str


class Side(StrEnum):
    """Direction of an order or fill.

    A :class:`~enum.StrEnum` so a side serialises and compares as its plain
    value (``"BUY"``/``"SELL"``), which keeps logs, CSV exports, and equality
    checks readable without special-casing the enum.
    """

    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True, slots=True)
class Order:
    """A request to buy or sell a quantity of one ticker.

    Phase 2 models market orders only: an order names what to trade and how
    much, and the broker decides the fill price. Quantity is a float because
    Alpaca supports fractional shares, which lets baselines target an exact
    dollar allocation without integer-rounding noise.

    Attributes:
        ticker: Symbol to trade.
        side: :class:`Side.BUY` or :class:`Side.SELL`.
        quantity: Number of shares (always positive; ``side`` carries direction).
        tag: Optional free-text label for why the order was issued (e.g.
            ``"entry"``, ``"sma-cross"``), carried through for diagnostics.
    """

    ticker: str
    side: Side
    quantity: float
    tag: str = ""


@dataclass(frozen=True, slots=True)
class Fill:
    """An executed order: the price and quantity that actually traded.

    A strategy emits :class:`Order` objects; the broker turns each accepted one
    into a ``Fill`` once it decides an execution price (open price plus adverse
    slippage in the simulator). The portfolio applies fills to update cash and
    positions.

    Attributes:
        ticker: Symbol traded.
        day: Session the fill occurred on.
        side: Direction of the trade.
        quantity: Shares filled.
        price: Per-share execution price, already including slippage.
        commission: Total commission charged on this fill (often ``0``).
    """

    ticker: str
    day: date
    side: Side
    quantity: float
    price: float
    commission: float


@dataclass(slots=True)
class Position:
    """A currently held long position in one ticker.

    Mutable on purpose: the portfolio adjusts ``quantity`` and ``avg_cost`` in
    place as fills arrive. ``avg_cost`` is the average price paid per share
    *including* buy commissions, so realised profit on a later sale is simply
    ``(sale_price - avg_cost) * shares``.

    Attributes:
        ticker: Symbol held.
        quantity: Shares currently held (positive; Phase 2 is long-only).
        avg_cost: Commission-inclusive average cost basis per share.
    """

    ticker: str
    quantity: float
    avg_cost: float


@dataclass(frozen=True, slots=True)
class Trade:
    """A closed (or partially closed) round trip, recorded when shares are sold.

    One ``Trade`` is logged each time a sale reduces a position, capturing the
    realised profit or loss on the shares sold. These records drive per-trade
    metrics such as hit rate; the running equity curve, by contrast, also
    reflects unrealised gains on positions still open.

    Attributes:
        ticker: Symbol traded.
        quantity: Shares sold in this round trip.
        entry_price: Cost basis per share (the position's ``avg_cost`` at sale).
        exit_price: Sale price per share, including slippage.
        exit_day: Session the closing sale filled on.
        pnl: Realised profit or loss in dollars, net of the sale's commission.
    """

    ticker: str
    quantity: float
    entry_price: float
    exit_price: float
    exit_day: date
    pnl: float
