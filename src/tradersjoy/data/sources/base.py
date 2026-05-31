"""The :class:`DataSource` interface.

Every concrete market-data provider (yfinance today, Alpaca later) implements
this single abstract method. The ingest layer and the rest of the system depend
only on this interface, never on a specific vendor's SDK, so adding or swapping
a provider is a new subclass and nothing else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

from tradersjoy.core.types import Bar


class DataSource(ABC):
    """Abstract base class for a provider of daily price bars.

    Subclasses set :attr:`name` and implement :meth:`get_daily_bars`.

    Attributes:
        name: Short identifier for the provider (e.g. ``"yfinance"``). It is
            recorded on every :class:`~tradersjoy.core.types.Bar` this source
            returns, so stored data carries its provenance.
    """

    name: str

    @abstractmethod
    def get_daily_bars(
        self, ticker: str, start: date, end: date | None = None
    ) -> list[Bar]:
        """Return daily bars for ``ticker`` within an inclusive date window.

        Implementations must treat an absence of data as a normal, empty result
        rather than an error: requesting dates before a company's IPO, or a
        weekend-only window, simply yields no bars. This lets the ingest layer
        iterate a watchlist where tickers have different histories without any
        single one aborting the run.

        Args:
            ticker: Symbol to fetch, e.g. ``"AAPL"``.
            start: Earliest date to include (inclusive).
            end: Latest date to include (inclusive). ``None`` means "up to the
                most recent available bar".

        Returns:
            Bars sorted ascending by :attr:`~tradersjoy.core.types.Bar.day`.
            Empty when the provider has no data for the window.

        Raises:
            NotImplementedError: If a subclass does not override this method.
        """
        raise NotImplementedError
