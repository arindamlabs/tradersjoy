"""Ingest orchestration: pull bars for a universe and persist them.

Loops the watchlist, fetches from a DataSource, upserts into the Store, and
returns a per-ticker summary. One ticker failing (network blip, delisted symbol)
never aborts the run; it is reported in the summary instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

from tradersjoy.config import PROJECT_ROOT
from tradersjoy.data.sources.base import DataSource
from tradersjoy.data.store import Store

DEFAULT_UNIVERSE_PATH = PROJECT_ROOT / "config" / "universe.yaml"


@dataclass
class TickerResult:
    """Outcome of ingesting one ticker, for the run summary.

    Attributes:
        ticker: The symbol this result is for.
        rows: Number of bars fetched from the source this run.
        start: Earliest day now stored for the ticker, or ``None`` if no data.
        end: Latest day now stored for the ticker, or ``None`` if no data.
        error: Error message if the ticker failed, otherwise ``None``. A
            non-``None`` value marks a failure that did not abort the run.
    """

    ticker: str
    rows: int
    start: date | None
    end: date | None
    error: str | None = None


def load_universe(path: Path = DEFAULT_UNIVERSE_PATH) -> tuple[list[str], date]:
    """Load the watchlist and backfill start date from a universe YAML file.

    Args:
        path: Path to a ``universe.yaml``. Defaults to ``config/universe.yaml``
            at the repo root.

    Returns:
        A ``(tickers, start_date)`` tuple. Tickers are stripped and upper-cased;
        ``start_date`` is parsed from the file's ``start_date`` field.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)
    tickers = [str(t).strip().upper() for t in cfg["watchlist"]]
    start = date.fromisoformat(str(cfg["start_date"]))
    return tickers, start


def ingest(
    source: DataSource,
    store: Store,
    tickers: list[str],
    start: date,
    end: date | None = None,
) -> list[TickerResult]:
    """Fetch and persist bars for a list of tickers.

    Iterates tickers, fetching from ``source`` and upserting into ``store``. A
    failure on one ticker (network blip, delisted symbol) is caught and recorded
    in that ticker's :class:`TickerResult`, never aborting the whole run, so a
    single bad symbol cannot waste a long multi-ticker backfill.

    Args:
        source: Provider to fetch bars from.
        store: Destination store. Its tables are created if missing.
        tickers: Symbols to ingest.
        start: Earliest date to request (inclusive).
        end: Latest date to request (inclusive), or ``None`` for latest
            available.

    Returns:
        One :class:`TickerResult` per input ticker, in input order, each
        reporting rows fetched, the resulting stored date range, and any error.
    """
    store.init_db()
    results: list[TickerResult] = []
    for ticker in tickers:
        try:
            bars = source.get_daily_bars(ticker, start, end)
            store.upsert_bars(bars)
            rng = store.date_range(ticker)
            results.append(
                TickerResult(
                    ticker=ticker,
                    rows=len(bars),
                    start=rng[0] if rng else None,
                    end=rng[1] if rng else None,
                )
            )
        except Exception as exc:  # noqa: BLE001 - we want one bad ticker isolated
            results.append(
                TickerResult(ticker=ticker, rows=0, start=None, end=None, error=str(exc))
            )
    return results
