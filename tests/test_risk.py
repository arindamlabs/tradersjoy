"""Offline tests for the Phase 5 risk layer: sizing, exposure, stops, breaker.

No network and no real data. Each case is a hand-checkable number on a tiny
portfolio so a failed assertion points at a real rule, not float noise. The rails
are deliberately stateless, so a :class:`Portfolio` (which satisfies the same
``AccountView`` contract the live account does) is all the state any test needs.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from tradersjoy.backtest.data import BarHistory
from tradersjoy.backtest.portfolio import Portfolio
from tradersjoy.core.types import Bar, Fill, Order, Side
from tradersjoy.risk.limits import RiskLimits
from tradersjoy.risk.manager import RiskManagedStrategy
from tradersjoy.strategy.base import BarContext, Strategy


class _Stub(Strategy):
    """A strategy that always proposes a fixed list of orders."""

    def __init__(self, orders: list[Order]) -> None:
        self._orders = orders

    @property
    def name(self) -> str:
        return "stub"

    def on_bar(self, ctx: BarContext) -> list[Order]:
        return list(self._orders)


def _history(closes_by_ticker: dict[str, list[float]]) -> BarHistory:
    """Build a panel where each ticker's opens equal its closes, aligned by day."""
    start = date(2020, 1, 1)
    bars: dict[str, list[Bar]] = {}
    for ticker, closes in closes_by_ticker.items():
        bars[ticker] = [
            Bar(
                ticker=ticker,
                day=start + timedelta(days=i),
                open=c,
                high=c + 1,
                low=c - 1,
                close=c,
                adj_close=c,
                volume=1000,
                source="test",
            )
            for i, c in enumerate(closes)
        ]
    return BarHistory(bars)


def _ctx(history: BarHistory, portfolio: Portfolio) -> BarContext:
    day = history.trading_days[-1]
    return BarContext(day=day, bars=history.bars_on(day), history=history, portfolio=portfolio)


def _by_ticker(orders: list[Order]) -> dict[str, Order]:
    return {o.ticker: o for o in orders}


def test_avg_cost_is_exposed_for_the_stop() -> None:
    p = Portfolio(100_000.0)
    p.apply_fill(Fill("AAPL", date(2020, 1, 2), Side.BUY, 100, 100.0, 0.0))
    assert p.avg_cost("AAPL") == pytest.approx(100.0)
    assert p.avg_cost("MSFT") == 0.0  # nothing held


def test_position_cap_trims_an_oversized_buy() -> None:
    # All cash, $100k equity, 20% per-name cap -> at most $20k in one name.
    p = Portfolio(100_000.0)
    history = _history({"AAPL": [100.0, 100.0]})
    risk = RiskManagedStrategy(
        ["AAPL"], _Stub([Order("AAPL", Side.BUY, 1000, tag="x")])  # wants $100k
    )
    orders = _by_ticker(risk.on_bar(_ctx(history, p)))
    assert orders["AAPL"].quantity == pytest.approx(200.0)  # $20k / $100 = 200 sh


def test_gross_exposure_cap_never_uses_margin() -> None:
    # Per-name cap relaxed so the *gross* cap is what binds: total <= 100% equity.
    p = Portfolio(100_000.0)
    history = _history({"AAPL": [100.0, 100.0], "MSFT": [100.0, 100.0]})
    limits = RiskLimits(max_position_weight=1.0, max_gross_exposure=1.0, crash_drawdown=None)
    risk = RiskManagedStrategy(
        ["AAPL", "MSFT"],
        _Stub([Order("AAPL", Side.BUY, 800), Order("MSFT", Side.BUY, 800)]),  # $160k
        limits,
    )
    orders = _by_ticker(risk.on_bar(_ctx(history, p)))
    # AAPL (first) gets its full $80k; MSFT is trimmed to the remaining $20k.
    assert orders["AAPL"].quantity == pytest.approx(800.0)
    assert orders["MSFT"].quantity == pytest.approx(200.0)
    deployed = sum(o.quantity * 100.0 for o in orders.values())
    assert deployed == pytest.approx(100_000.0)  # exactly 100%, never above


def test_cost_basis_stop_exits_and_suppresses_rebuy() -> None:
    p = Portfolio(100_000.0)
    p.apply_fill(Fill("AAPL", date(2020, 1, 2), Side.BUY, 100, 100.0, 0.0))  # cost 100
    history = _history({"AAPL": [100.0, 85.0]})  # now 85, i.e. -15% < -10% stop
    # The inner strategy stubbornly wants to add more AAPL; the stop must win.
    risk = RiskManagedStrategy(["AAPL"], _Stub([Order("AAPL", Side.BUY, 50)]))
    orders = risk.on_bar(_ctx(history, p))
    assert len(orders) == 1
    assert orders[0] == Order("AAPL", Side.SELL, 100, tag="risk-stop")


def test_stop_does_not_fire_above_the_threshold() -> None:
    p = Portfolio(100_000.0)
    p.apply_fill(Fill("AAPL", date(2020, 1, 2), Side.BUY, 100, 100.0, 0.0))
    history = _history({"AAPL": [100.0, 95.0]})  # only -5%, inside the 10% stop
    risk = RiskManagedStrategy(["AAPL"], _Stub([]))
    assert risk.on_bar(_ctx(history, p)) == []  # no exit forced


def test_circuit_breaker_blocks_buys_but_allows_exits() -> None:
    p = Portfolio(100_000.0)
    p.apply_fill(Fill("AAPL", date(2020, 1, 2), Side.BUY, 100, 100.0, 0.0))
    # SPY 20% off its high -> deeper than the 15% breaker.
    history = _history(
        {
            "AAPL": [100.0, 100.0, 100.0, 100.0, 100.0],
            "MSFT": [50.0, 50.0, 50.0, 50.0, 50.0],
            "SPY": [100.0, 100.0, 100.0, 100.0, 80.0],
        }
    )
    risk = RiskManagedStrategy(
        ["AAPL", "MSFT", "SPY"],
        _Stub([Order("MSFT", Side.BUY, 100), Order("AAPL", Side.SELL, 40, tag="ml-exit")]),
    )
    orders = risk.on_bar(_ctx(history, p))
    assert orders == [Order("AAPL", Side.SELL, 40, tag="ml-exit")]  # buy dropped, exit kept


def test_within_limits_orders_pass_through_unchanged() -> None:
    p = Portfolio(100_000.0)
    history = _history({"AAPL": [100.0, 100.0], "SPY": [90.0, 100.0]})  # SPY at its high
    risk = RiskManagedStrategy(
        ["AAPL", "SPY"], _Stub([Order("AAPL", Side.BUY, 100, tag="entry")])  # $10k, fine
    )
    orders = risk.on_bar(_ctx(history, p))
    assert orders == [Order("AAPL", Side.BUY, 100, tag="entry")]


# --------------------------------------------------------------------------
# Named risk profiles
# --------------------------------------------------------------------------


def test_caps_profile_keeps_structural_limits_and_drops_reactive_rails() -> None:
    """The two kinds of rail are separable, and that separation is the point.

    Measured out-of-sample, the reactive rails (stop-loss, circuit breaker) cost
    6.4pp of CAGR and made max drawdown *worse*, while the structural caps were
    close to free. Keeping them in one all-or-nothing switch forced a bad trade.
    """
    from tradersjoy.risk.limits import limits_for

    full, caps = limits_for("full"), limits_for("caps")

    # Structural caps survive: these bound what one bug or blow-up can do.
    assert caps.max_position_weight == full.max_position_weight
    assert caps.max_gross_exposure == full.max_gross_exposure
    # Reactive rails are off.
    assert caps.stop_loss is None
    assert caps.crash_drawdown is None
    assert full.stop_loss is not None and full.crash_drawdown is not None


def test_unknown_risk_profile_is_rejected() -> None:
    import pytest

    from tradersjoy.risk.limits import limits_for

    with pytest.raises(ValueError, match="Unknown risk profile"):
        limits_for("yolo")


def test_caps_profile_does_not_stop_out_a_losing_position() -> None:
    """The behavioural difference, not just the config difference."""
    from tradersjoy.risk.limits import limits_for

    # AAA drifts down; a holder whose cost basis is 100 is far past a 10% stop.
    history = _history({"AAA": [100.0, 95.0, 90.0, 80.0], "SPY": [100.0] * 4})

    class _Underwater:
        equity = 100_000.0
        cash = 0.0

        def qty(self, ticker: str) -> float:
            return 100.0 if ticker == "AAA" else 0.0

        def avg_cost(self, ticker: str) -> float:
            return 100.0 if ticker == "AAA" else 0.0

    day = history.trading_days[-1]
    ctx = BarContext(
        day=day, bars=history.bars_on(day), history=history, portfolio=_Underwater()
    )
    tickers = ["AAA"]

    full = RiskManagedStrategy(tickers, _Stub([]), limits=limits_for("full"))
    caps = RiskManagedStrategy(tickers, _Stub([]), limits=limits_for("caps"))

    assert [o.side for o in full.on_bar(ctx)] == [Side.SELL]  # stop fires
    assert caps.on_bar(ctx) == []  # no reactive rail, position rides
