"""An in-memory panel of bars indexed for no-look-ahead access during a backtest.

A backtest replays many tickers day by day, so it needs two fast lookups:

- "what bars exist on this exact day?" (to price fills and mark the portfolio),
- "what is this ticker's history up to and including day T?" (for indicators).

:class:`BarHistory` precomputes both from bars already loaded out of the
:class:`~tradersjoy.data.store.Store`. The history accessor is strictly
inclusive of the requested day and never returns anything after it, which is the
mechanical guarantee that a strategy cannot peek at the future when it computes a
signal.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from datetime import date

from tradersjoy.core.types import Bar
from tradersjoy.data.store import Store


class BarHistory:
    """Date-indexed view over a fixed set of tickers' bars.

    Construct it from a mapping of ticker to its bars (already sorted ascending
    by day), or via :func:`load_history`. All accessors are read-only.

    Attributes:
        trading_days: The sorted union of every day on which any ticker has a
            bar. This is the calendar the backtest engine steps through.
    """

    def __init__(self, bars_by_ticker: dict[str, Sequence[Bar]]) -> None:
        """Index the supplied bars for per-day and per-ticker-history lookups.

        Args:
            bars_by_ticker: Each ticker's bars, sorted ascending by day. Tickers
                with no bars may be omitted or map to an empty sequence.
        """
        self._bars: dict[str, list[Bar]] = {
            t: list(bs) for t, bs in bars_by_ticker.items()
        }
        self._days: dict[str, list[date]] = {
            t: [b.day for b in bs] for t, bs in self._bars.items()
        }
        panel: dict[date, dict[str, Bar]] = {}
        for ticker, bars in self._bars.items():
            for bar in bars:
                panel.setdefault(bar.day, {})[ticker] = bar
        self._panel = panel
        self.trading_days: list[date] = sorted(panel)

    @property
    def tickers(self) -> list[str]:
        """The tickers this panel covers, in the order they were supplied."""
        return list(self._bars.keys())

    def bars_on(self, day: date) -> dict[str, Bar]:
        """Return every ticker's bar for ``day`` (empty dict if none traded)."""
        return self._panel.get(day, {})

    def history(self, ticker: str, up_to: date) -> list[Bar]:
        """Return ``ticker``'s bars from the start through ``up_to`` inclusive.

        Args:
            ticker: Symbol to read.
            up_to: Last day to include. Bars after this day are never returned,
                which is what prevents look-ahead in indicator calculations.

        Returns:
            Bars in ascending day order; empty if the ticker is unknown or has
            no bars on or before ``up_to``.
        """
        days = self._days.get(ticker)
        if not days:
            return []
        cut = bisect.bisect_right(days, up_to)
        return self._bars[ticker][:cut]

    def adj_closes(self, ticker: str, up_to: date) -> list[float]:
        """Return adjusted closes through ``up_to`` inclusive, for signal math.

        Adjusted closes (not raw closes) are the right series for indicators:
        they are continuous across splits and dividends, so a moving average is
        not jolted by a mechanical price change that cost a holder nothing.
        """
        return [b.adj_close for b in self.history(ticker, up_to)]


def load_history(
    store: Store,
    tickers: Sequence[str],
    start: date | None = None,
    end: date | None = None,
) -> BarHistory:
    """Load each ticker's bars from the store into a :class:`BarHistory`.

    Args:
        store: Market-data store to read from.
        tickers: Symbols to include in the backtest panel.
        start: Earliest day to load (inclusive), or ``None`` for no lower bound.
        end: Latest day to load (inclusive), or ``None`` for no upper bound.

    Returns:
        A :class:`BarHistory` over the requested tickers and window.
    """
    return BarHistory({t: store.get_bars(t, start, end) for t in tickers})
