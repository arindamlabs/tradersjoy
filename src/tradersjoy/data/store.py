"""Local persistence for market data.

Backed by SQLite via SQLAlchemy. The public surface is the ``Store`` class:
construct it (optionally with a custom database URL for tests), call
``init_db()`` once, then ``upsert_bars`` / ``get_bars`` / ``count``.

Upserts are idempotent on the ``(ticker, day)`` primary key, so re-running an
ingest never duplicates rows; it overwrites with the latest values (useful when
a provider revises a bar, e.g. a late split adjustment).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

from sqlalchemy import (
    BigInteger,
    Date,
    Float,
    String,
    create_engine,
    func,
    select,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from tradersjoy.config import get_settings
from tradersjoy.core.types import Bar


class Base(DeclarativeBase):
    """Declarative base for all ORM models in the store."""


class DailyBar(Base):
    """ORM row for one daily bar, mirroring :class:`~tradersjoy.core.types.Bar`.

    The primary key is the composite ``(ticker, day)``: at most one bar per
    symbol per day. This is what makes ingest idempotent, the same key upserts
    in place instead of inserting a duplicate. ``volume`` uses ``BigInteger``
    because daily share volume for liquid names routinely exceeds the 32-bit
    integer range.
    """

    __tablename__ = "daily_bars"

    ticker: Mapped[str] = mapped_column(String, primary_key=True)
    day: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    adj_close: Mapped[float] = mapped_column(Float)
    volume: Mapped[int] = mapped_column(BigInteger)
    source: Mapped[str] = mapped_column(String)


def _to_bar(row: DailyBar) -> Bar:
    """Convert a persisted :class:`DailyBar` ORM row to a domain :class:`Bar`."""
    return Bar(
        ticker=row.ticker,
        day=row.day,
        open=row.open,
        high=row.high,
        low=row.low,
        close=row.close,
        adj_close=row.adj_close,
        volume=row.volume,
        source=row.source,
    )


class Store:
    """Read/write access to the local market-data database.

    Wraps a single SQLAlchemy engine. Construct once and reuse; the engine
    manages its own connection pool. Pass an explicit ``database_url`` in tests
    to point at a throwaway SQLite file, leave it ``None`` in normal use to read
    the configured location from :func:`~tradersjoy.config.get_settings`.

    Attributes:
        database_url: The SQLAlchemy URL this store is bound to.
        engine: The SQLAlchemy engine backing all operations.
    """

    def __init__(self, database_url: str | None = None) -> None:
        """Bind the store to a database, creating the SQLite directory if needed.

        Args:
            database_url: SQLAlchemy URL (e.g. ``"sqlite:///data/x.sqlite"``).
                Defaults to the configured ``DATABASE_URL``. Note this does not
                create the tables; call :meth:`init_db` for that.
        """
        self.database_url = database_url or get_settings().database_url
        self._ensure_sqlite_dir(self.database_url)
        self.engine = create_engine(self.database_url)

    @staticmethod
    def _ensure_sqlite_dir(database_url: str) -> None:
        """Create the parent directory for a file-based SQLite DB if needed.

        SQLite will not create missing parent directories itself, so a URL like
        ``sqlite:///data/tradersjoy.sqlite`` fails if ``data/`` does not exist.
        In-memory (``:memory:``) and non-SQLite URLs are left untouched.

        Args:
            database_url: The SQLAlchemy URL being opened.
        """
        prefix = "sqlite:///"
        if database_url.startswith(prefix):
            db_path = database_url[len(prefix) :]
            if db_path and db_path != ":memory:":
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def init_db(self) -> None:
        """Create any missing tables. Idempotent and safe to call repeatedly."""
        Base.metadata.create_all(self.engine)

    def upsert_bars(self, bars: Sequence[Bar]) -> int:
        """Insert bars, overwriting any existing row with the same key.

        Idempotent on the ``(ticker, day)`` primary key via SQLite's
        ``ON CONFLICT ... DO UPDATE``: re-ingesting an overlapping window never
        creates duplicates, and a revised bar (e.g. a late split adjustment from
        the provider) replaces the stored values. The batch is de-duplicated by
        key first because a single ``ON CONFLICT`` statement cannot reference the
        same primary key twice; when a key repeats in the input, the last
        occurrence wins.

        Args:
            bars: Bars to persist. May be empty, may contain multiple tickers,
                and may contain duplicate keys (the last wins).

        Returns:
            The number of distinct rows written (post de-duplication), i.e. the
            count of unique ``(ticker, day)`` keys in ``bars``.
        """
        if not bars:
            return 0

        deduped: dict[tuple[str, date], Bar] = {}
        for b in bars:
            deduped[(b.ticker, b.day)] = b
        rows = [
            {
                "ticker": b.ticker,
                "day": b.day,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "adj_close": b.adj_close,
                "volume": b.volume,
                "source": b.source,
            }
            for b in deduped.values()
        ]

        stmt = sqlite_insert(DailyBar).values(rows)
        update_cols = {
            c: stmt.excluded[c]
            for c in ("open", "high", "low", "close", "adj_close", "volume", "source")
        }
        stmt = stmt.on_conflict_do_update(
            index_elements=["ticker", "day"], set_=update_cols
        )
        with Session(self.engine) as session:
            session.execute(stmt)
            session.commit()
        return len(rows)

    def get_bars(
        self, ticker: str, start: date | None = None, end: date | None = None
    ) -> list[Bar]:
        """Read stored bars for one ticker within an optional date window.

        Args:
            ticker: Symbol to query, e.g. ``"AAPL"``.
            start: Earliest day to include (inclusive). ``None`` means no lower
                bound.
            end: Latest day to include (inclusive). ``None`` means no upper
                bound.

        Returns:
            Bars sorted ascending by day; empty if nothing matches.
        """
        stmt = select(DailyBar).where(DailyBar.ticker == ticker)
        if start is not None:
            stmt = stmt.where(DailyBar.day >= start)
        if end is not None:
            stmt = stmt.where(DailyBar.day <= end)
        stmt = stmt.order_by(DailyBar.day)
        with Session(self.engine) as session:
            return [_to_bar(r) for r in session.scalars(stmt)]

    def count(self, ticker: str | None = None) -> int:
        """Count stored bars, optionally for a single ticker.

        Args:
            ticker: If given, count only this symbol's bars; otherwise count
                every bar in the store.

        Returns:
            The number of matching rows (``0`` if none).
        """
        stmt = select(func.count()).select_from(DailyBar)
        if ticker is not None:
            stmt = stmt.where(DailyBar.ticker == ticker)
        with Session(self.engine) as session:
            return int(session.scalar(stmt) or 0)

    def date_range(self, ticker: str) -> tuple[date, date] | None:
        """Return the earliest and latest stored day for a ticker.

        Useful for reporting what history a backfill actually captured, which
        varies per symbol because of differing IPO dates.

        Args:
            ticker: Symbol to inspect.

        Returns:
            A ``(min_day, max_day)`` tuple, or ``None`` if no bars are stored for
            the ticker.
        """
        stmt = select(func.min(DailyBar.day), func.max(DailyBar.day)).where(
            DailyBar.ticker == ticker
        )
        with Session(self.engine) as session:
            lo, hi = session.execute(stmt).one()
        if lo is None or hi is None:
            return None
        return lo, hi
