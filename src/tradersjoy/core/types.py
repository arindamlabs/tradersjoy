"""Core domain types shared across the system.

These are deliberately plain and storage-agnostic. The SQLAlchemy models in
:mod:`tradersjoy.data.store` map to and from these; strategies, the backtester,
and the live engine speak only in these types so nothing downstream is coupled
to how we happen to persist data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


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
