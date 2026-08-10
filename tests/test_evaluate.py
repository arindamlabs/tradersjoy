"""Offline tests for the walk-forward backtest.

The property under test is the one that makes the whole evaluation trustworthy:
each calendar year must be traded by a model that never saw that year. If the
year-to-model dispatch is wrong the backtest silently becomes in-sample, which
looks like a spectacular result rather than like a bug.
"""

from __future__ import annotations

from datetime import date

from tradersjoy.core.types import Bar
from tradersjoy.ml.evaluate import WalkForwardMLStrategy
from tradersjoy.strategy.base import BarContext

TICKERS = ["AAA", "BBB"]
DAYS = [date(2020, 1, 6), date(2021, 1, 4), date(2022, 1, 3)]


class _TaggedModel:
    """Model stand-in that reports which fold it came from via its scores."""

    feature_names = ["ret_1"]

    def __init__(self, year: int) -> None:
        self.year = year
        self.calls = 0

    def predict_proba(self, rows) -> list[float]:  # noqa: ANN001
        self.calls += 1
        return [0.9] * len(rows)


class _Flat:
    equity = 100_000.0
    cash = 100_000.0

    def qty(self, ticker: str) -> float:
        return 0.0

    def avg_cost(self, ticker: str) -> float:
        return 0.0


def _history(days: list[date]):
    from tradersjoy.backtest.data import BarHistory

    return BarHistory(
        {
            t: [
                Bar(t, d, 100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i,
                    100.0 + i, 1_000, "test")
                for i, d in enumerate(days)
            ]
            for t in TICKERS
        }
    )


def _ctx(day: date, history):
    return BarContext(
        day=day, bars=history.bars_on(day), history=history, portfolio=_Flat()
    )


def test_each_year_is_traded_by_its_own_fold_model(monkeypatch) -> None:
    """The out-of-sample guarantee, mechanically."""
    from tradersjoy.strategy.ml.strategy import MLStrategy

    models = {2021: _TaggedModel(2021), 2022: _TaggedModel(2022)}
    strat = WalkForwardMLStrategy(TICKERS, models, top_k=2, rebalance="daily")
    history = _history(DAYS)

    # Record which inner strategy (hence which model) actually scored.
    seen: list[int] = []

    def spy(self, ctx):  # noqa: ANN001
        seen.append(self.model.year)
        return {"AAA": 0.9, "BBB": 0.8}

    monkeypatch.setattr(MLStrategy, "_scores", spy)

    strat.on_bar(_ctx(date(2021, 1, 4), history))
    strat.on_bar(_ctx(date(2022, 1, 3), history))
    assert seen == [2021, 2022]


def test_years_before_the_first_fold_propose_nothing(monkeypatch) -> None:
    """No model is entitled to an opinion about a year it was trained on."""
    from tradersjoy.strategy.ml.strategy import MLStrategy

    monkeypatch.setattr(
        MLStrategy, "_scores", lambda self, ctx: {"AAA": 0.9, "BBB": 0.8}
    )
    models = {2021: _TaggedModel(2021), 2022: _TaggedModel(2022)}
    strat = WalkForwardMLStrategy(TICKERS, models, top_k=2, rebalance="daily")
    history = _history(DAYS)

    # 2020 predates every fold: it was training data for the 2021 model.
    assert strat.on_bar(_ctx(date(2020, 1, 6), history)) == []
    assert strat.on_bar(_ctx(date(2021, 1, 4), history)) != []


def test_cadence_is_honoured_across_the_year_dispatch(monkeypatch) -> None:
    """Swapping models must not accidentally re-enable daily trading."""
    from tradersjoy.strategy.ml.strategy import MLStrategy

    monkeypatch.setattr(
        MLStrategy, "_scores", lambda self, ctx: {"AAA": 0.9, "BBB": 0.8}
    )
    week = [date(2021, 1, 4), date(2021, 1, 5), date(2021, 1, 6)]
    history = _history(week)
    strat = WalkForwardMLStrategy(
        TICKERS, {2021: _TaggedModel(2021)}, top_k=2, rebalance="weekly"
    )

    assert strat.on_bar(_ctx(week[0], history)) != []  # Monday
    assert strat.on_bar(_ctx(week[1], history)) == []  # Tuesday: hold
    assert strat.on_bar(_ctx(week[2], history)) == []


def test_name_records_the_cadence() -> None:
    strat = WalkForwardMLStrategy(
        TICKERS, {2021: _TaggedModel(2021)}, top_k=5, rebalance="monthly"
    )
    assert strat.name == "wf-ml(top5,monthly)"


def test_walk_forward_retains_one_model_per_test_year() -> None:
    """evaluate_horizon depends on these being kept; assert the contract."""
    from tradersjoy.ml.dataset import Sample
    from tradersjoy.ml.features import FEATURE_NAMES
    from tradersjoy.ml.labels import Label
    from tradersjoy.ml.walkforward import walk_forward

    samples = []
    for year in (2015, 2016, 2017, 2018, 2019, 2020, 2021):
        for i in range(120):
            day = date(year, 1 + i % 12, 1 + i % 27)
            samples.append(
                Sample(
                    ticker="AAA",
                    day=day,
                    features={n: float((i + j) % 7) for j, n in enumerate(FEATURE_NAMES)},
                    label=Label(
                        value=i % 2,
                        fwd_return=0.01 * (i % 3 - 1),
                        end_day=day,
                    ),
                )
            )

    result = walk_forward(samples, train_years=3)
    assert result.models, "fold models must be retained"
    assert set(result.models) == {f.test_year for f in result.folds}
    # Each fold's model is a distinct object: no accidental sharing.
    assert len(set(map(id, result.models.values()))) == len(result.models)
