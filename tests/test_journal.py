"""Offline tests for the Phase 6 run journal and the dashboard curve helper.

No network and no Alpaca: a throwaway SQLite file under ``tmp_path`` is all the
journal needs, and the equity-curve helper is a pure function. The cases pin the
behaviours the dashboard relies on: runs round-trip (orders included), dry-run
versus executed is preserved, and a day with several runs collapses to its
latest equity on the curve.
"""

from __future__ import annotations

from datetime import date, datetime

from tradersjoy.core.types import Order, Side
from tradersjoy.dashboard.data import equity_curve
from tradersjoy.live.journal import Journal, RunEntry
from tradersjoy.live.trader import LivePlan


def _journal(tmp_path) -> Journal:
    j = Journal(database_url=f"sqlite:///{tmp_path / 'journal.sqlite'}")
    j.init_db()
    return j


def test_record_and_read_back_round_trips_orders(tmp_path) -> None:
    j = _journal(tmp_path)
    orders = [
        Order("AAPL", Side.BUY, 10.0, tag="entry"),
        Order("MSFT", Side.SELL, 3.0, tag="risk-stop"),
    ]
    j.record(
        run_at=datetime(2026, 6, 5, 16, 30),
        decision_day=date(2026, 6, 5),
        strategy="risk(ml(top5))",
        equity=101_000.0,
        cash=20_000.0,
        starting_equity=100_000.0,
        executed=True,
        orders=orders,
    )

    [entry] = j.recent()
    assert entry.decision_day == date(2026, 6, 5)
    assert entry.strategy == "risk(ml(top5))"
    assert entry.executed is True
    assert entry.pnl == 1_000.0
    assert entry.orders == orders  # side, qty, and tag all survive the JSON trip


def test_record_plan_stores_a_dry_run(tmp_path) -> None:
    j = _journal(tmp_path)
    plan = LivePlan(
        day=date(2026, 6, 5),
        strategy_name="buyhold",
        equity=100_000.0,
        cash=100_000.0,
        starting_equity=100_000.0,
        orders=[Order("SPY", Side.BUY, 5.0)],
        executed=False,
    )
    j.record_plan(plan, run_at=datetime(2026, 6, 5, 12, 0))

    [entry] = j.recent()
    assert entry.executed is False  # a dry run is journaled as intent, not action
    assert entry.orders == [Order("SPY", Side.BUY, 5.0)]


def test_recent_is_newest_first(tmp_path) -> None:
    j = _journal(tmp_path)
    for i, day in enumerate((date(2026, 6, 3), date(2026, 6, 4), date(2026, 6, 5))):
        j.record(
            run_at=datetime(2026, 6, 3 + i, 16, 0),
            decision_day=day,
            strategy="buyhold",
            equity=100_000.0 + i,
            cash=0.0,
            starting_equity=100_000.0,
            executed=False,
            orders=[],
        )
    days = [e.decision_day for e in j.recent()]
    assert days == [date(2026, 6, 5), date(2026, 6, 4), date(2026, 6, 3)]


def _entry(run_at: datetime, day: date, equity: float) -> RunEntry:
    return RunEntry(
        run_at=run_at,
        decision_day=day,
        strategy="x",
        equity=equity,
        cash=0.0,
        starting_equity=100_000.0,
        executed=False,
        orders=[],
    )


def test_equity_curve_keeps_one_point_per_day_latest_wins() -> None:
    # Two runs on 06-05: the later one (higher equity) is the day's value.
    entries = [
        _entry(datetime(2026, 6, 5, 9, 0), date(2026, 6, 5), 100_500.0),
        _entry(datetime(2026, 6, 5, 16, 0), date(2026, 6, 5), 101_000.0),
        _entry(datetime(2026, 6, 4, 16, 0), date(2026, 6, 4), 100_000.0),
    ]
    curve = equity_curve(entries)
    assert curve == [(date(2026, 6, 4), 100_000.0), (date(2026, 6, 5), 101_000.0)]


def test_equity_curve_is_empty_for_no_runs() -> None:
    assert equity_curve([]) == []
