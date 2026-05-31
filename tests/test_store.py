"""Offline tests for the Store layer. No network, so CI stays deterministic."""

from __future__ import annotations

from datetime import date

import pytest

from tradersjoy.core.types import Bar
from tradersjoy.data.store import Store


@pytest.fixture
def store(tmp_path) -> Store:
    s = Store(database_url=f"sqlite:///{tmp_path / 'test.sqlite'}")
    s.init_db()
    return s


def _bar(ticker: str, day: date, close: float, source: str = "test") -> Bar:
    return Bar(
        ticker=ticker,
        day=day,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        adj_close=close,
        volume=1000,
        source=source,
    )


def test_upsert_is_idempotent(store: Store) -> None:
    bars = [_bar("AAPL", date(2020, 1, 2), 100.0), _bar("AAPL", date(2020, 1, 3), 101.0)]
    store.upsert_bars(bars)
    store.upsert_bars(bars)  # re-ingest the same window
    assert store.count("AAPL") == 2  # no duplicates


def test_upsert_overwrites_on_conflict(store: Store) -> None:
    day = date(2020, 1, 2)
    store.upsert_bars([_bar("AAPL", day, 100.0)])
    store.upsert_bars([_bar("AAPL", day, 123.45, source="revised")])
    rows = store.get_bars("AAPL")
    assert len(rows) == 1
    assert rows[0].close == 123.45
    assert rows[0].source == "revised"


def test_dedup_within_single_batch(store: Store) -> None:
    day = date(2020, 1, 2)
    # Same key twice in one call must not raise and must keep the last value.
    store.upsert_bars([_bar("AAPL", day, 100.0), _bar("AAPL", day, 200.0)])
    rows = store.get_bars("AAPL")
    assert len(rows) == 1
    assert rows[0].close == 200.0


def test_get_bars_respects_date_range_and_order(store: Store) -> None:
    days = [date(2020, 1, d) for d in (2, 3, 6, 7, 8)]
    store.upsert_bars([_bar("MSFT", d, 50.0 + i) for i, d in enumerate(days)])
    rng = store.get_bars("MSFT", start=date(2020, 1, 3), end=date(2020, 1, 7))
    assert [b.day for b in rng] == [date(2020, 1, 3), date(2020, 1, 6), date(2020, 1, 7)]
    assert rng == sorted(rng, key=lambda b: b.day)


def test_count_and_date_range_isolate_tickers(store: Store) -> None:
    store.upsert_bars([_bar("AAPL", date(2020, 1, 2), 100.0)])
    store.upsert_bars(
        [_bar("MSFT", date(2019, 6, 1), 50.0), _bar("MSFT", date(2021, 6, 1), 60.0)]
    )
    assert store.count() == 3
    assert store.count("MSFT") == 2
    assert store.date_range("MSFT") == (date(2019, 6, 1), date(2021, 6, 1))
    assert store.date_range("NVDA") is None
