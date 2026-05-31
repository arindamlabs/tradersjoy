"""yfinance-backed data source.

yfinance is free and reaches back decades, which makes it our workhorse for the
historical backfill. It is not a contractual data feed (it scrapes Yahoo), so we
treat it as best-effort: empty results and the occasional missing field are
handled gracefully rather than crashing a multi-ticker ingest.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from tradersjoy.core.types import Bar
from tradersjoy.data.sources.base import DataSource


class YFinanceSource(DataSource):
    """Daily bars sourced from Yahoo Finance via the ``yfinance`` library.

    Suited to the historical backfill because it is free and offers decades of
    history. Because it is best-effort (scraped, not contractual), this class is
    defensive: it normalizes yfinance's shifting column shapes, tolerates missing
    fields, and returns an empty list instead of raising when there is no data.
    """

    name = "yfinance"

    def get_daily_bars(
        self, ticker: str, start: date, end: date | None = None
    ) -> list[Bar]:
        """Download and normalize daily bars for one ticker.

        Args:
            ticker: Symbol to fetch, e.g. ``"AAPL"``.
            start: Earliest date to include (inclusive).
            end: Latest date to include (inclusive), or ``None`` for "through the
                latest available bar". yfinance treats its own ``end`` as
                exclusive, so we add one day internally to keep this method's
                contract inclusive.

        Returns:
            Bars sorted ascending by day. Empty if yfinance returns nothing
            (e.g. dates before the IPO) or the response is missing the required
            OHLCV columns.

        Notes:
            ``auto_adjust=False`` is set deliberately so the response includes
            both the raw ``Close`` and the split/dividend-adjusted ``Adj Close``;
            we persist both (see :class:`~tradersjoy.core.types.Bar`). Rows whose
            close is ``NaN`` (non-trading days that occasionally slip through) are
            skipped, and a missing ``Adj Close`` falls back to the raw close.
        """
        # auto_adjust=False so we get both raw Close and split/dividend-adjusted
        # Adj Close. end is exclusive in yfinance, so add a day when provided.
        end_arg = None if end is None else (end + timedelta(days=1))
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=None if end_arg is None else end_arg.isoformat(),
            interval="1d",
            auto_adjust=False,
            actions=False,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            return []

        # With a single ticker, recent yfinance versions still return MultiIndex
        # columns like ("Close", "AAPL"). Flatten to the first level.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(df.columns):
            return []
        has_adj = "Adj Close" in df.columns

        bars: list[Bar] = []
        for idx, row in df.iterrows():
            close = row["Close"]
            if pd.isna(close):
                continue  # skip non-trading rows that slipped through
            adj = row["Adj Close"] if has_adj and not pd.isna(row["Adj Close"]) else close
            day = idx.date() if hasattr(idx, "date") else pd.Timestamp(idx).date()
            volume = row["Volume"]
            bars.append(
                Bar(
                    ticker=ticker,
                    day=day,
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(close),
                    adj_close=float(adj),
                    volume=0 if pd.isna(volume) else int(volume),
                    source=self.name,
                )
            )
        return bars
