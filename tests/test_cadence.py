"""Offline tests for the rebalance cadence.

Two things are being pinned. First, the calendar arithmetic: a rebalance day is
the first *trading* session of a period, which has to survive holidays, weekends,
and the year boundary. Second, and more important, the property the whole design
exists for: the decision is a pure function of the date and the surrounding
sessions, with no memory. A cadence that counted "sessions since last rebalance"
would pass a backtest and then rebalance every single day live, because
``run_once`` starts fresh each session with no recollection of previous runs.
"""

from __future__ import annotations

from datetime import date

import pytest

from tradersjoy.core.types import Bar, Side
from tradersjoy.strategy.cadence import (
    CADENCES,
    is_rebalance_day,
    period_key,
)


def _sessions(days: list[str]) -> list[date]:
    return [date.fromisoformat(d) for d in days]


# A normal week, then a week whose Monday is a holiday.
WEEK = _sessions(
    [
        "2026-08-03",  # Mon
        "2026-08-04",
        "2026-08-05",
        "2026-08-06",
        "2026-08-07",  # Fri
        "2026-08-11",  # Tue: Monday 08-10 is a holiday in this fixture
        "2026-08-12",
    ]
)


def test_weekly_fires_once_per_week() -> None:
    fired = [d for d in WEEK if is_rebalance_day(d, WEEK, "weekly")]
    assert fired == [date(2026, 8, 3), date(2026, 8, 11)]


def test_weekly_uses_the_first_open_session_when_monday_is_a_holiday() -> None:
    """Holidays need no special-casing: the week's first session is whichever
    day the market actually opened."""
    assert is_rebalance_day(date(2026, 8, 11), WEEK, "weekly") is True
    assert is_rebalance_day(date(2026, 8, 12), WEEK, "weekly") is False


def test_daily_always_fires() -> None:
    assert all(is_rebalance_day(d, WEEK, "daily") for d in WEEK)


def test_first_known_session_always_rebalances() -> None:
    """Nothing precedes it, so it must be allowed to open the initial book."""
    assert is_rebalance_day(WEEK[0], WEEK, "weekly") is True
    assert is_rebalance_day(WEEK[0], WEEK, "monthly") is True


def test_monthly_fires_on_the_first_session_of_the_month() -> None:
    days = _sessions(["2026-07-30", "2026-07-31", "2026-08-03", "2026-08-04"])
    fired = [d for d in days if is_rebalance_day(d, days, "monthly")]
    assert fired == [date(2026, 7, 30), date(2026, 8, 3)]


def test_fortnightly_does_not_collapse_at_the_year_boundary() -> None:
    """ISO week numbers restart each year, so week-parity bucketing would make
    weeks 52 and 1 adjacent and emit a one-week 'fortnight'. Ordinal bucketing
    does not."""
    dec = date(2026, 12, 28)
    jan = date(2027, 1, 4)
    # One week apart: they must still fall in the same fortnight bucket, or in
    # adjacent buckets -- never more than one apart.
    assert abs(period_key(jan, "fortnightly") - period_key(dec, "fortnightly")) <= 1


def test_unknown_cadence_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown rebalance cadence"):
        period_key(date(2026, 8, 3), "hourly")


def test_decision_is_stateless() -> None:
    """The property that makes backtest and live agree.

    Same date and same surrounding sessions must give the same answer no matter
    how much history the caller happened to load, and no matter how many times
    it has been asked. This is what a counter-based implementation could not do.
    """
    for cadence in CADENCES:
        for day in WEEK:
            first = is_rebalance_day(day, WEEK, cadence)
            # Asking again must not change the answer: no counter is ticking.
            assert is_rebalance_day(day, WEEK, cadence) is first, (cadence, day)

    # A caller holding only a short window (as the live path does: ~500 days,
    # not 20 years) must get the same answers as one holding everything, so
    # long as the window reaches back one session.
    for day in WEEK[-2:]:
        for cadence in CADENCES:
            window = [d for d in WEEK if d <= day][-3:]
            assert is_rebalance_day(day, window, cadence) is is_rebalance_day(
                day, WEEK, cadence
            ), (cadence, day)


# --------------------------------------------------------------------------
# The strategy actually honouring the cadence
# --------------------------------------------------------------------------


class _FlatAccount:
    equity = 100_000.0
    cash = 100_000.0

    def qty(self, ticker: str) -> float:
        return 0.0

    def avg_cost(self, ticker: str) -> float:
        return 0.0


class _AlwaysBullishModel:
    """Stand-in model that likes every name equally."""

    feature_names = ["ret_1"]

    def predict_proba(self, rows) -> list[float]:  # noqa: ANN001
        return [0.9] * len(rows)


def _history(tickers: list[str], days: list[date]):
    """Build a BarHistory covering the given tickers and sessions."""
    from tradersjoy.backtest.data import BarHistory

    return BarHistory(
        {
            t: [
                Bar(t, d, 100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i,
                    100.0 + i, 1_000, "test")
                for i, d in enumerate(days)
            ]
            for t in tickers
        }
    )


def test_strategy_proposes_nothing_off_cadence(monkeypatch) -> None:
    """The core behaviour change: no orders on a non-rebalance session."""
    from tradersjoy.strategy.base import BarContext
    from tradersjoy.strategy.ml.strategy import MLStrategy

    tickers = ["AAA", "BBB"]
    history = _history(tickers, WEEK)
    strat = MLStrategy(tickers, _AlwaysBullishModel(), top_k=2, rebalance="weekly")
    # Bypass feature computation; this test is about the cadence gate only.
    monkeypatch.setattr(
        MLStrategy, "_scores", lambda self, ctx: {"AAA": 0.9, "BBB": 0.8}
    )

    def orders_on(day: date):
        ctx = BarContext(
            day=day,
            bars=history.bars_on(day),
            history=history,
            portfolio=_FlatAccount(),
        )
        return strat.on_bar(ctx)

    assert orders_on(date(2026, 8, 3))  # Monday: rebalance
    assert orders_on(date(2026, 8, 4)) == []  # Tuesday: hold
    assert orders_on(date(2026, 8, 5)) == []
    assert orders_on(date(2026, 8, 11))  # next week's first session


def test_cadence_appears_in_the_strategy_name() -> None:
    """So a journal row or scorecard can never be misread as the other cadence."""
    from tradersjoy.strategy.ml.strategy import MLStrategy

    weekly = MLStrategy(["AAA"], _AlwaysBullishModel(), top_k=5, rebalance="weekly")
    daily = MLStrategy(["AAA"], _AlwaysBullishModel(), top_k=5, rebalance="daily")
    assert weekly.name == "ml(top5,weekly)"
    assert daily.name == "ml(top5,daily)"


def test_strategy_rejects_an_unknown_cadence() -> None:
    from tradersjoy.strategy.ml.strategy import MLStrategy

    with pytest.raises(ValueError, match="Unknown rebalance cadence"):
        MLStrategy(["AAA"], _AlwaysBullishModel(), rebalance="hourly")


def test_stop_loss_still_fires_on_a_hold_day(monkeypatch) -> None:
    """Suppressing strategy churn must not suppress the risk rails.

    A stop-loss that waited for the next rebalance would let a position run
    another four sessions past its limit, which is precisely the protection the
    risk layer exists to provide.
    """
    from tradersjoy.risk.manager import RiskManagedStrategy
    from tradersjoy.strategy.base import BarContext
    from tradersjoy.strategy.ml.strategy import MLStrategy

    tickers = ["AAA"]
    history = _history(tickers, WEEK)
    inner = MLStrategy(tickers, _AlwaysBullishModel(), top_k=1, rebalance="weekly")
    monkeypatch.setattr(MLStrategy, "_scores", lambda self, ctx: {"AAA": 0.9})
    managed = RiskManagedStrategy(tickers, inner)

    class _Underwater:
        """Holds AAA, bought far above the current price."""

        equity = 100_000.0
        cash = 0.0

        def qty(self, ticker: str) -> float:
            return 100.0

        def avg_cost(self, ticker: str) -> float:
            return 1_000.0  # current price is ~104: way past any sane stop

    tuesday = date(2026, 8, 4)  # deliberately NOT a rebalance day
    assert inner.on_bar(
        BarContext(day=tuesday, bars=history.bars_on(tuesday),
                   history=history, portfolio=_Underwater())
    ) == []

    orders = managed.on_bar(
        BarContext(day=tuesday, bars=history.bars_on(tuesday),
                   history=history, portfolio=_Underwater())
    )
    assert [(o.ticker, o.side) for o in orders] == [("AAA", Side.SELL)]
