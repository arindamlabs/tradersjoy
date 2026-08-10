"""Offline tests for the Phase 6b unattended runner.

No network and no Alpaca: a fake broker supplies the account, the market clock,
and the session calendar, and a throwaway SQLite file under ``tmp_path`` backs
the store and journal. The cases pin the behaviour that makes automation safe to
leave alone, which is almost entirely about the runner *refusing* to trade:
stale data, a live session, a halt file, a duplicate firing, and a broken feed
must each stop it, and every one of them must still leave a heartbeat behind.
"""

from __future__ import annotations

from datetime import date, datetime

import pytest

from tradersjoy.core.types import Bar, Order, Side
from tradersjoy.data.store import Store
from tradersjoy.live.autorun import run_daily
from tradersjoy.live.journal import Journal

SESSION = date(2026, 8, 7)
TICKERS = ["AAA", "BBB"]


class FakeAccount:
    """Minimal ``AccountView``: flat account, no positions."""

    equity = 100_000.0
    cash = 100_000.0

    def qty(self, ticker: str) -> float:
        return 0.0

    def avg_cost(self, ticker: str) -> float:
        return 0.0


class FakeBroker:
    """Broker stub with a controllable clock, calendar, and submission log."""

    def __init__(self, *, market_open: bool = False, session: date | None = SESSION):
        self._market_open = market_open
        self._session = session
        self.submitted: list[Order] = []

    def get_account(self) -> FakeAccount:
        return FakeAccount()

    def is_market_open(self) -> bool:
        return self._market_open

    def last_completed_session(self, now=None) -> date | None:  # noqa: ANN001
        return self._session

    def submit(self, orders) -> list[str]:  # noqa: ANN001
        self.submitted.extend(orders)
        return [f"placed {o.side} {o.quantity} {o.ticker}" for o in orders]


class OneOrderStrategy:
    """Strategy stub that always wants to buy one share of the first ticker."""

    name = "fake"

    def on_bar(self, ctx) -> list[Order]:  # noqa: ANN001
        return [Order(TICKERS[0], Side.BUY, 1.0, tag="test")]


class NoOrderStrategy:
    """Strategy stub that never trades."""

    name = "quiet"

    def on_bar(self, ctx) -> list[Order]:  # noqa: ANN001
        return []


@pytest.fixture
def wiring(tmp_path, monkeypatch):
    """A store and journal on a temp DB, with strategy building stubbed out."""
    url = f"sqlite:///{tmp_path / 'test.sqlite'}"
    store = Store(database_url=url)
    store.init_db()
    journal = Journal(database_url=url)
    journal.init_db()

    strategy_holder = {"strategy": OneOrderStrategy()}
    monkeypatch.setattr(
        "tradersjoy.strategy.registry.build_strategy",
        lambda *a, **k: strategy_holder["strategy"],
    )
    # The runner must never reach the network in these tests.
    monkeypatch.setattr("tradersjoy.live.autorun._refresh", lambda *a, **k: None)

    return {
        "store": store,
        "journal": journal,
        "tmp_path": tmp_path,
        "strategy": strategy_holder,
    }


def _seed_bars(store: Store, day: date) -> None:
    """Put one bar per ticker into the store for ``day``."""
    store.upsert_bars(
        [
            Bar(t, day, 100.0, 101.0, 99.0, 100.0, 100.0, 1_000, "test")
            for t in TICKERS
        ]
    )


def _run(wiring, **kwargs):
    """Call ``run_daily`` with the fixture's wiring and sensible test defaults."""
    params = {
        "tickers": TICKERS,
        "store": wiring["store"],
        "journal": wiring["journal"],
        "halt_file": wiring["tmp_path"] / "HALT",
        "lock_file": wiring["tmp_path"] / "autorun.lock",
        "broker": FakeBroker(),
    }
    params.update(kwargs)
    return run_daily(**params)


def test_executes_when_every_guard_passes(wiring) -> None:
    _seed_bars(wiring["store"], SESSION)
    broker = FakeBroker()

    result = _run(wiring, broker=broker, execute=True)

    assert result.status == "traded"
    assert result.session == SESSION
    assert result.exit_code == 0
    assert [o.ticker for o in broker.submitted] == [TICKERS[0]]


def test_dry_run_decides_but_places_nothing(wiring) -> None:
    _seed_bars(wiring["store"], SESSION)
    broker = FakeBroker()

    result = _run(wiring, broker=broker, execute=False)

    assert result.status == "dry_run"
    assert result.exit_code == 0
    assert broker.submitted == []  # the whole point of the default


def test_stale_data_refuses_to_trade(wiring) -> None:
    # The store only reaches the day *before* the session we intend to trade,
    # exactly what a lagging or failed data feed looks like.
    _seed_bars(wiring["store"], date(2026, 8, 6))
    broker = FakeBroker()

    result = _run(wiring, broker=broker, execute=True)

    assert result.status == "failed"
    assert "stale data" in result.reason
    assert result.exit_code == 1  # systemd should see this one
    assert broker.submitted == []


def test_open_market_refuses_to_trade(wiring) -> None:
    _seed_bars(wiring["store"], SESSION)
    broker = FakeBroker(market_open=True)

    result = _run(wiring, broker=broker, execute=True)

    assert result.status == "skipped"
    assert "still open" in result.reason
    assert result.exit_code == 0  # a skip is not a failure
    assert broker.submitted == []


def test_halt_file_pauses_trading(wiring) -> None:
    _seed_bars(wiring["store"], SESSION)
    halt = wiring["tmp_path"] / "HALT"
    halt.touch()
    broker = FakeBroker()

    result = _run(wiring, broker=broker, execute=True, halt_file=halt)

    assert result.status == "skipped"
    assert "halt file" in result.reason
    assert broker.submitted == []


def test_second_firing_for_same_session_is_skipped(wiring) -> None:
    _seed_bars(wiring["store"], SESSION)
    first = FakeBroker()
    second = FakeBroker()

    _run(wiring, broker=first, execute=True)
    result = _run(wiring, broker=second, execute=True)

    assert result.status == "skipped"
    assert "already traded" in result.reason
    assert second.submitted == []  # no double-dip on a retry


def test_force_overrides_the_duplicate_guard(wiring) -> None:
    _seed_bars(wiring["store"], SESSION)
    _run(wiring, broker=FakeBroker(), execute=True)

    second = FakeBroker()
    result = _run(wiring, broker=second, execute=True, force=True)

    assert result.status == "traded"
    assert len(second.submitted) == 1


def test_dry_run_does_not_block_a_later_real_run(wiring) -> None:
    _seed_bars(wiring["store"], SESSION)
    _run(wiring, broker=FakeBroker(), execute=False)

    live = FakeBroker()
    result = _run(wiring, broker=live, execute=True)

    assert result.status == "traded"  # a dry run committed to nothing
    assert len(live.submitted) == 1


def test_no_orders_is_a_clean_outcome(wiring) -> None:
    _seed_bars(wiring["store"], SESSION)
    wiring["strategy"]["strategy"] = NoOrderStrategy()

    result = _run(wiring, execute=True)

    assert result.status == "no_orders"
    assert result.exit_code == 0


def test_broken_data_feed_fails_without_trading(wiring, monkeypatch) -> None:
    _seed_bars(wiring["store"], SESSION)

    def boom(*a, **k):
        raise RuntimeError("yfinance exploded")

    monkeypatch.setattr("tradersjoy.live.autorun._refresh", boom)
    broker = FakeBroker()

    result = _run(wiring, broker=broker, execute=True)

    assert result.status == "failed"
    assert "yfinance exploded" in result.reason
    assert broker.submitted == []


def test_every_outcome_leaves_a_heartbeat(wiring) -> None:
    """The gap-detection property: refusals are recorded, not silent."""
    _seed_bars(wiring["store"], date(2026, 8, 6))  # forces a stale-data failure
    _run(wiring, execute=True)

    halt = wiring["tmp_path"] / "HALT"
    halt.touch()
    _run(wiring, execute=True, halt_file=halt)

    runs = wiring["journal"].recent_auto_runs()
    assert len(runs) == 2
    assert {r.status for r in runs} == {"failed", "skipped"}
    assert all(not r.ok for r in runs)
    assert wiring["journal"].last_auto_run().status == "skipped"


def test_unresolvable_session_fails(wiring) -> None:
    _seed_bars(wiring["store"], SESSION)
    broker = FakeBroker(session=None)

    result = _run(wiring, broker=broker, execute=True)

    assert result.status == "failed"
    assert broker.submitted == []


def test_concurrent_run_is_refused(wiring) -> None:
    """A second runner must not trade while the first holds the lock."""
    from tradersjoy.live.autorun import _exclusive_lock

    _seed_bars(wiring["store"], SESSION)
    lock = wiring["tmp_path"] / "autorun.lock"
    broker = FakeBroker()

    with _exclusive_lock(lock) as held:
        assert held
        result = _run(wiring, broker=broker, execute=True, lock_file=lock)

    assert result.status == "failed"
    assert "another run" in result.reason
    assert broker.submitted == []


def test_journal_records_the_decision_alongside_the_heartbeat(wiring) -> None:
    _seed_bars(wiring["store"], SESSION)

    _run(wiring, execute=True, now=datetime(2026, 8, 7, 17, 0))

    [entry] = wiring["journal"].recent()
    assert entry.decision_day == SESSION
    assert entry.executed is True
    assert [o.ticker for o in entry.orders] == [TICKERS[0]]


# --------------------------------------------------------------------------
# Equity floor
# --------------------------------------------------------------------------


class _EquityBroker(FakeBroker):
    """Broker stub whose account equity can be set per test."""

    def __init__(self, equity: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self._equity = equity

    def get_account(self):
        account = FakeAccount()
        account.equity = self._equity
        return account


def test_equity_at_or_below_the_floor_withholds_all_orders(wiring) -> None:
    """The safeguard as specified: nothing is placed, buys or sells."""
    _seed_bars(wiring["store"], SESSION)
    broker = _EquityBroker(99_000.0)

    result = _run(wiring, broker=broker, execute=True, min_equity=100_000.0)

    assert result.status == "floored"
    assert broker.submitted == []
    assert result.exit_code == 0  # configured behaviour, not a fault


def test_floor_is_inclusive(wiring) -> None:
    """"<= threshold" as asked: exactly at the floor still suspends."""
    _seed_bars(wiring["store"], SESSION)
    broker = _EquityBroker(100_000.0)

    result = _run(wiring, broker=broker, execute=True, min_equity=100_000.0)

    assert result.status == "floored"
    assert broker.submitted == []


def test_above_the_floor_trades_normally(wiring) -> None:
    _seed_bars(wiring["store"], SESSION)
    broker = _EquityBroker(100_000.01)

    result = _run(wiring, broker=broker, execute=True, min_equity=100_000.0)

    assert result.status == "traded"
    assert len(broker.submitted) == 1


def test_floor_resumes_automatically_without_intervention(wiring) -> None:
    """No state to clear: recovery is just the next run seeing higher equity."""
    _seed_bars(wiring["store"], SESSION)

    down = _EquityBroker(95_000.0)
    assert _run(wiring, broker=down, execute=True, min_equity=100_000.0).status == "floored"
    assert down.submitted == []

    up = _EquityBroker(101_000.0)
    recovered = _run(wiring, broker=up, execute=True, min_equity=100_000.0, force=True)
    assert recovered.status == "traded"
    assert len(up.submitted) == 1


def test_floor_disabled_by_default(wiring) -> None:
    """Omitting the floor must not silently suspend a poor account."""
    _seed_bars(wiring["store"], SESSION)
    broker = _EquityBroker(1_000.0)

    result = _run(wiring, broker=broker, execute=True)

    assert result.status == "traded"
    assert len(broker.submitted) == 1


def test_floored_run_still_records_what_it_wanted(wiring) -> None:
    """Suspended is not the same as blind: the record shows the withheld plan."""
    _seed_bars(wiring["store"], SESSION)
    broker = _EquityBroker(50_000.0)

    _run(wiring, broker=broker, execute=True, min_equity=100_000.0)

    [entry] = wiring["journal"].recent()
    assert entry.executed is False
    assert [o.ticker for o in entry.orders] == [TICKERS[0]]  # wanted it, withheld it

    beat = wiring["journal"].last_auto_run()
    assert beat.status == "floored"
    assert beat.n_orders == 1
    assert beat.ok  # configured behaviour, so not flagged as broken
    assert "floor" in beat.reason


def test_floored_run_does_not_block_a_later_recovery_run(wiring) -> None:
    """A withheld run committed to nothing, so it must not consume the session."""
    _seed_bars(wiring["store"], SESSION)

    _run(wiring, broker=_EquityBroker(90_000.0), execute=True, min_equity=100_000.0)
    up = _EquityBroker(110_000.0)
    result = _run(wiring, broker=up, execute=True, min_equity=100_000.0)

    assert result.status == "traded"  # no --force needed
    assert len(up.submitted) == 1


def test_dry_run_reports_the_floor_the_same_way_a_live_run_would(wiring) -> None:
    """A rehearsal must not disagree with the performance.

    A dry run is how you ask "what would it do today?". If it answered
    "would have placed 5 orders" while the scheduled run would have withheld
    them, the rehearsal would be actively misleading.
    """
    _seed_bars(wiring["store"], SESSION)

    dry = _run(wiring, broker=_EquityBroker(90_000.0), execute=False, min_equity=100_000.0)
    live = _run(
        wiring, broker=_EquityBroker(90_000.0), execute=True, min_equity=100_000.0
    )

    assert dry.status == "floored"
    assert live.status == "floored"
