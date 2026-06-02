"""Offline tests for the Phase 3 live path: the decision cycle and order planning.

No Alpaca and no network: a fake broker stands in for the real one, so these
tests pin down the logic that matters (dry-run places nothing, execute submits,
buy-and-hold never double-buys what the account already holds, fractional orders
floor to whole shares) without depending on a live account.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from tradersjoy.broker.alpaca import plan_whole_share_orders
from tradersjoy.core.types import Bar, Order, Side
from tradersjoy.data.store import Store
from tradersjoy.live.trader import LiveTrader
from tradersjoy.strategy.baselines.buy_and_hold import BuyAndHold


class FakeAccount:
    """Minimal stand-in satisfying the AccountView contract."""

    def __init__(
        self, equity: float, cash: float, positions: dict[str, float] | None = None
    ) -> None:
        self.equity = equity
        self.cash = cash
        self._positions = positions or {}

    def qty(self, ticker: str) -> float:
        return self._positions.get(ticker, 0.0)


class FakeBroker:
    """Records submissions instead of hitting Alpaca."""

    def __init__(self, account: FakeAccount) -> None:
        self._account = account
        self.submitted: list[Order] = []

    def get_account(self) -> FakeAccount:
        return self._account

    def submit(self, orders: Sequence[Order]) -> list[str]:
        self.submitted.extend(orders)
        return [f"placed {o.side} {o.quantity} {o.ticker}" for o in orders]


def _bar(ticker: str, day: date, close: float) -> Bar:
    return Bar(
        ticker=ticker,
        day=day,
        open=close,
        high=close + 1,
        low=close - 1,
        close=close,
        adj_close=close,
        volume=1000,
        source="test",
    )


def _store_with(tmp_path, bars: list[Bar]) -> Store:
    store = Store(database_url=f"sqlite:///{tmp_path / 'live.sqlite'}")
    store.init_db()
    store.upsert_bars(bars)
    return store


def test_dry_run_decides_but_submits_nothing(tmp_path) -> None:
    day = date(2026, 1, 5)
    store = _store_with(tmp_path, [_bar("AAPL", day, 100.0), _bar("MSFT", day, 200.0)])
    broker = FakeBroker(FakeAccount(equity=1000.0, cash=1000.0))
    trader = LiveTrader(broker, store)

    plan = trader.run_once(BuyAndHold(["AAPL", "MSFT"]), ["AAPL", "MSFT"], execute=False)

    assert len(plan.orders) == 2  # wants to buy both
    assert plan.executed is False
    assert broker.submitted == []  # but placed nothing


def test_execute_submits_the_decided_orders(tmp_path) -> None:
    day = date(2026, 1, 5)
    store = _store_with(tmp_path, [_bar("AAPL", day, 100.0), _bar("MSFT", day, 200.0)])
    broker = FakeBroker(FakeAccount(equity=1000.0, cash=1000.0))
    trader = LiveTrader(broker, store)

    plan = trader.run_once(BuyAndHold(["AAPL", "MSFT"]), ["AAPL", "MSFT"], execute=True)

    assert plan.executed is True
    assert {o.ticker for o in broker.submitted} == {"AAPL", "MSFT"}
    assert len(plan.results) == 2


def test_buy_and_hold_is_idempotent_against_existing_positions(tmp_path) -> None:
    day = date(2026, 1, 5)
    store = _store_with(tmp_path, [_bar("AAPL", day, 100.0), _bar("MSFT", day, 200.0)])
    # The account already holds AAPL (as if a prior run bought it).
    broker = FakeBroker(FakeAccount(equity=1000.0, cash=500.0, positions={"AAPL": 5.0}))
    trader = LiveTrader(broker, store)

    plan = trader.run_once(BuyAndHold(["AAPL", "MSFT"]), ["AAPL", "MSFT"], execute=True)

    # Only MSFT is bought; AAPL is skipped because it is already held.
    assert {o.ticker for o in plan.orders} == {"MSFT"}
    assert {o.ticker for o in broker.submitted} == {"MSFT"}


def test_pnl_reflects_equity_over_starting_balance(tmp_path) -> None:
    day = date(2026, 1, 5)
    store = _store_with(tmp_path, [_bar("AAPL", day, 100.0)])
    broker = FakeBroker(FakeAccount(equity=112_500.0, cash=50_000.0))
    trader = LiveTrader(broker, store, starting_equity=100_000.0)

    plan = trader.run_once(BuyAndHold(["AAPL"]), ["AAPL"], execute=False)
    assert plan.pnl == 12_500.0


def test_run_once_raises_when_store_is_empty(tmp_path) -> None:
    store = _store_with(tmp_path, [])  # no bars
    broker = FakeBroker(FakeAccount(equity=1000.0, cash=1000.0))
    trader = LiveTrader(broker, store)

    try:
        trader.run_once(BuyAndHold(["AAPL"]), ["AAPL"], execute=False)
    except ValueError as exc:
        assert "No bars" in str(exc)
    else:  # pragma: no cover - the call must raise
        raise AssertionError("expected ValueError on an empty store")


def test_plan_whole_share_orders_floors_and_flags_dust() -> None:
    plans = plan_whole_share_orders(
        [
            Order("AAPL", Side.BUY, 9.7, "x"),
            Order("MSFT", Side.BUY, 0.4, "x"),
            Order("SPY", Side.SELL, 3.0, "x"),
        ]
    )
    by_ticker = {p.ticker: p for p in plans}
    assert by_ticker["AAPL"].shares == 9  # floored, not rounded
    assert by_ticker["MSFT"].shares == 0 and by_ticker["MSFT"].note  # flagged skip
    assert by_ticker["SPY"].shares == 3
