"""Offline tests for the Phase 2 backtest stack: portfolio, broker, engine, metrics.

All synthetic data and no network, so CI stays deterministic. The cases lean on
hand-checkable numbers (round prices, small share counts) so an assertion failure
points at a real accounting bug rather than floating-point noise.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tradersjoy.backtest.data import BarHistory
from tradersjoy.backtest.engine import run_backtest
from tradersjoy.backtest.metrics import compute_metrics
from tradersjoy.backtest.portfolio import Portfolio
from tradersjoy.broker.sim import SimBroker
from tradersjoy.core.types import Bar, Fill, Order, Side
from tradersjoy.strategy.base import BarContext, Strategy
from tradersjoy.strategy.baselines.buy_and_hold import BuyAndHold


def _bar(ticker: str, day: date, o: float, c: float) -> Bar:
    return Bar(
        ticker=ticker,
        day=day,
        open=o,
        high=max(o, c) + 1,
        low=min(o, c) - 1,
        close=c,
        adj_close=c,
        volume=1000,
        source="test",
    )


def _series(ticker: str, start: date, opens_closes: list[tuple[float, float]]) -> list[Bar]:
    return [
        _bar(ticker, start + timedelta(days=i), o, c)
        for i, (o, c) in enumerate(opens_closes)
    ]


# --- Portfolio accounting ---------------------------------------------------


def test_portfolio_buy_then_sell_realizes_pnl() -> None:
    p = Portfolio(starting_cash=1000.0)
    day = date(2020, 1, 2)
    p.apply_fill(Fill("AAPL", day, Side.BUY, 10, 50.0, 0.0))
    assert p.cash == pytest.approx(500.0)
    assert p.qty("AAPL") == 10

    sell_day = date(2020, 1, 3)
    p.apply_fill(Fill("AAPL", sell_day, Side.SELL, 10, 60.0, 0.0))
    assert p.cash == pytest.approx(1100.0)
    assert p.qty("AAPL") == 0
    assert len(p.trades) == 1
    assert p.trades[0].pnl == pytest.approx(100.0)  # (60 - 50) * 10


def test_partial_sell_keeps_position_and_avg_cost() -> None:
    p = Portfolio(starting_cash=1000.0)
    day = date(2020, 1, 2)
    p.apply_fill(Fill("AAPL", day, Side.BUY, 10, 50.0, 0.0))
    p.apply_fill(Fill("AAPL", date(2020, 1, 3), Side.SELL, 4, 55.0, 0.0))
    assert p.qty("AAPL") == 6
    assert p.positions["AAPL"].avg_cost == pytest.approx(50.0)
    assert p.trades[0].pnl == pytest.approx(20.0)  # (55 - 50) * 4


def test_commission_folds_into_basis_and_pnl() -> None:
    p = Portfolio(starting_cash=1000.0)
    p.apply_fill(Fill("AAPL", date(2020, 1, 2), Side.BUY, 10, 50.0, 5.0))
    # 500 spent on shares + 5 commission
    assert p.cash == pytest.approx(495.0)
    assert p.positions["AAPL"].avg_cost == pytest.approx(50.5)
    p.apply_fill(Fill("AAPL", date(2020, 1, 3), Side.SELL, 10, 50.5, 5.0))
    # exit at basis, but the 5.0 sell commission is the realized loss
    assert p.trades[0].pnl == pytest.approx(-5.0)


def test_equity_tracks_marked_prices() -> None:
    p = Portfolio(starting_cash=1000.0)
    p.apply_fill(Fill("AAPL", date(2020, 1, 2), Side.BUY, 10, 50.0, 0.0))
    p.mark(date(2020, 1, 2), {"AAPL": 70.0})
    assert p.equity == pytest.approx(500.0 + 10 * 70.0)  # cash + unrealized


# --- SimBroker fills, slippage, rejections ----------------------------------


def test_broker_fills_at_next_open_with_slippage() -> None:
    p = Portfolio(starting_cash=10_000.0)
    broker = SimBroker(slippage_bps=10.0)  # 0.10%
    day = date(2020, 1, 2)
    broker.submit([Order("AAPL", Side.BUY, 10, "x")])
    fills = broker.settle(day, {"AAPL": _bar("AAPL", day, 100.0, 105.0)}, p)
    assert len(fills) == 1
    assert fills[0].price == pytest.approx(100.0 * 1.001)  # buy pays open + slippage
    assert p.qty("AAPL") == 10


def test_broker_rejects_unaffordable_buy() -> None:
    p = Portfolio(starting_cash=100.0)
    broker = SimBroker(slippage_bps=0.0)
    day = date(2020, 1, 2)
    broker.submit([Order("AAPL", Side.BUY, 10, "x")])  # needs ~1000
    fills = broker.settle(day, {"AAPL": _bar("AAPL", day, 100.0, 100.0)}, p)
    assert fills == []
    assert p.qty("AAPL") == 0
    assert len(broker.rejections) == 1


def test_broker_rejects_oversized_sell() -> None:
    p = Portfolio(starting_cash=10_000.0)
    broker = SimBroker(slippage_bps=0.0)
    day = date(2020, 1, 2)
    broker.submit([Order("AAPL", Side.SELL, 5, "x")])  # nothing held
    fills = broker.settle(day, {"AAPL": _bar("AAPL", day, 100.0, 100.0)}, p)
    assert fills == []
    assert len(broker.rejections) == 1


def test_broker_keeps_order_pending_when_no_bar() -> None:
    p = Portfolio(starting_cash=10_000.0)
    broker = SimBroker(slippage_bps=0.0)
    broker.submit([Order("AAPL", Side.BUY, 1, "x")])
    # ticker has no bar today -> stays pending, fills when its bar appears
    assert broker.settle(date(2020, 1, 2), {}, p) == []
    fills = broker.settle(date(2020, 1, 3), {"AAPL": _bar("AAPL", date(2020, 1, 3), 50.0, 50.0)}, p)
    assert len(fills) == 1


# --- Metrics ----------------------------------------------------------------


def test_metrics_on_known_curve() -> None:
    curve = [
        (date(2020, 1, 1), 100.0),
        (date(2020, 1, 2), 120.0),
        (date(2020, 1, 3), 90.0),  # drawdown from the 120 peak
        (date(2020, 1, 4), 150.0),
    ]
    m = compute_metrics(curve, trades=[])
    assert m.total_return == pytest.approx(0.5)  # 150 / 100 - 1
    assert m.max_drawdown == pytest.approx(90.0 / 120.0 - 1.0)  # -0.25


def test_metrics_hit_rate_from_trades() -> None:
    from tradersjoy.core.types import Trade

    trades = [
        Trade("AAPL", 1, 10.0, 12.0, date(2020, 1, 2), 2.0),
        Trade("AAPL", 1, 10.0, 9.0, date(2020, 1, 3), -1.0),
        Trade("AAPL", 1, 10.0, 11.0, date(2020, 1, 4), 1.0),
    ]
    m = compute_metrics([(date(2020, 1, 1), 100.0)], trades)
    assert m.num_trades == 3
    assert m.hit_rate == pytest.approx(2 / 3)
    assert m.realized_pnl == pytest.approx(2.0)


# --- Engine end-to-end ------------------------------------------------------


def test_engine_no_lookahead_fills_next_open() -> None:
    """An order decided on day T must fill at day T+1's open, not T's close."""

    class BuyOnceOnFirstDay(Strategy):
        def __init__(self) -> None:
            self.done = False

        @property
        def name(self) -> str:
            return "buy-once"

        def on_bar(self, ctx: BarContext) -> list[Order]:
            if self.done or "AAPL" not in ctx.bars:
                return []
            self.done = True
            return [Order("AAPL", Side.BUY, 1, "x")]

    start = date(2020, 1, 1)
    # Day 0 close=100 (decision); day 1 open=200 is the price we must pay.
    bars = _series("AAPL", start, [(100.0, 100.0), (200.0, 210.0)])
    data = BarHistory({"AAPL": bars})
    broker = SimBroker(slippage_bps=0.0)
    result = run_backtest(BuyOnceOnFirstDay(), data, broker, starting_cash=1000.0)
    assert result.fills[0].price == pytest.approx(200.0)  # next open, not 100
    assert result.fills[0].day == start + timedelta(days=1)


def test_buy_and_hold_runs_and_invests() -> None:
    start = date(2020, 1, 1)
    a = _series("AAPL", start, [(10.0, 10.0)] * 5)
    b = _series("MSFT", start, [(20.0, 20.0)] * 5)
    data = BarHistory({"AAPL": a, "MSFT": b})
    broker = SimBroker(slippage_bps=0.0)
    result = run_backtest(
        BuyAndHold(["AAPL", "MSFT"]), data, broker, starting_cash=1000.0
    )
    # Both names bought once, no exits, so no closed trades but cash deployed.
    assert result.metrics.num_trades == 0
    assert result.fills, "buy-and-hold should have placed buys"
    assert {f.ticker for f in result.fills} == {"AAPL", "MSFT"}
