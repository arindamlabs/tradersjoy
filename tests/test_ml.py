"""Offline tests for the Phase 4 ML pipeline: features, labels, purge, model.

No network and no real data: synthetic bars and a tiny planted signal pin down
the properties that actually keep the model honest, the ones a flattering bug
would quietly break:

- features depend only on the past (changing a future bar cannot move an earlier
  day's features),
- the label looks the right number of days ahead and the most-recent rows stay
  unlabelled,
- walk-forward *purges* training rows whose answer window reaches into the test
  year (the subtle boundary leak), and
- the model wrapper actually learns a planted signal and round-trips to disk.
"""

from __future__ import annotations

from datetime import date, timedelta

from tradersjoy.backtest.data import BarHistory
from tradersjoy.core.types import Bar
from tradersjoy.ml.dataset import Sample, _relativize, build_dataset, labelled, matrix
from tradersjoy.ml.features import FEATURE_NAMES, MIN_BARS, features_from_bars
from tradersjoy.ml.labels import Label, forward_label
from tradersjoy.ml.model import GBMModel
from tradersjoy.ml.walkforward import walk_forward


def _bars(ticker: str, n: int, base: date = date(2020, 1, 1)) -> list[Bar]:
    """Deterministic, gently trending bars: enough history to form features."""
    out: list[Bar] = []
    for i in range(n):
        price = 100.0 + i * 0.5 + (i % 5)  # smooth uptrend with a little wiggle
        out.append(
            Bar(
                ticker=ticker,
                day=base + timedelta(days=i),
                open=price,
                high=price + 1,
                low=price - 1,
                close=price,
                adj_close=price,
                volume=1000 + (i % 7) * 10,
                source="test",
            )
        )
    return out


# --- features -------------------------------------------------------------


def test_features_need_minimum_history() -> None:
    assert features_from_bars(_bars("X", MIN_BARS - 1)) is None
    feats = features_from_bars(_bars("X", MIN_BARS))
    assert feats is not None
    assert set(feats) == set(FEATURE_NAMES)


def test_ret_1_matches_hand_calculation() -> None:
    bars = _bars("X", MIN_BARS)
    feats = features_from_bars(bars)
    assert feats is not None
    expected = bars[-1].adj_close / bars[-2].adj_close - 1.0
    assert abs(feats["ret_1"] - expected) < 1e-12


def test_relative_feature_subtracts_the_benchmark() -> None:
    bars = _bars("X", MIN_BARS)
    feats = features_from_bars(bars)  # no benchmark -> neutral
    assert feats is not None
    assert feats["rel_ret_5"] == 0.0 and feats["rel_ret_20"] == 0.0

    rel = features_from_bars(bars, benchmark={"ret_5": 0.01, "ret_20": 0.03})
    assert rel is not None
    assert abs(rel["rel_ret_5"] - (feats["ret_5"] - 0.01)) < 1e-12
    assert abs(rel["rel_ret_20"] - (feats["ret_20"] - 0.03)) < 1e-12


def test_features_do_not_depend_on_future_bars() -> None:
    """Corrupting the LAST bar must not change any earlier day's features."""
    tickers = ["X"]
    n = MIN_BARS + 30
    clean = BarHistory({"X": _bars("X", n)})
    bars2 = _bars("X", n)
    last = bars2[-1]
    bars2[-1] = Bar(  # absurd values on the final day only
        ticker=last.ticker,
        day=last.day,
        open=9e9,
        high=9e9,
        low=9e9,
        close=9e9,
        adj_close=9e9,
        volume=10**9,
        source="test",
    )
    corrupted = BarHistory({"X": bars2})

    clean_s = {s.day: s.features for s in build_dataset(clean, tickers)}
    corrupt_s = {s.day: s.features for s in build_dataset(corrupted, tickers)}
    # Every day except the final one must be byte-for-byte identical.
    for day in clean_s:
        if day == last.day:
            continue
        assert clean_s[day] == corrupt_s[day], f"future bar leaked into {day}"


# --- labels ---------------------------------------------------------------


def test_forward_label_direction_and_end_day() -> None:
    closes = [10.0, 11.0, 12.0, 9.0]
    days = [date(2020, 1, d) for d in (1, 2, 3, 4)]
    up = forward_label(closes, days, index=0, horizon=2, threshold=0.0)
    assert up == Label(value=1, fwd_return=12.0 / 10.0 - 1.0, end_day=days[2])
    down = forward_label(closes, days, index=1, horizon=2, threshold=0.0)
    assert down is not None and down.value == 0  # 11 -> 9
    # No future left for the last index at this horizon.
    assert forward_label(closes, days, index=3, horizon=2) is None


def test_recent_rows_are_unlabelled_and_matrix_rejects_them() -> None:
    history = BarHistory({"X": _bars("X", MIN_BARS + 20)})
    samples = build_dataset(history, ["X"], horizon=5)
    # The final `horizon` rows have no known future yet.
    tail = sorted(samples, key=lambda s: s.day)[-5:]
    assert all(s.label is None for s in tail)
    assert all(s.label is not None for s in labelled(samples))
    try:
        matrix(samples)  # contains unlabelled rows
    except ValueError as exc:
        assert "unlabelled" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("matrix must reject unlabelled rows")


# --- relative (cross-sectional) label -------------------------------------


def test_relative_label_splits_each_day_at_the_universe_median() -> None:
    """A stock is a 1 iff it beat that day's median; the benchmark is excluded."""
    day, end = date(2021, 3, 1), date(2021, 3, 8)

    def s(ticker: str, fwd: float) -> Sample:
        feats = {name: 0.0 for name in FEATURE_NAMES}
        return Sample(ticker, day, feats, Label(value=0, fwd_return=fwd, end_day=end))

    # Non-benchmark forward returns {A: .05, B: .01, C: -.02} -> median .01.
    samples = [s("A", 0.05), s("B", 0.01), s("C", -0.02), s("SPY", 0.10)]
    by = {x.ticker: x.label for x in _relativize(samples, threshold=0.0, benchmark="SPY")}

    assert by["A"].value == 1  # above the median
    assert by["B"].value == 0  # exactly at the median is not "beating" it
    assert by["C"].value == 0  # below the median
    assert by["SPY"] is None  # benchmark is the yardstick, never a training row


def test_relative_label_is_balanced_and_excludes_the_benchmark() -> None:
    """End to end: divergent tickers get a roughly 50/50, benchmark-free target."""

    def sloped(ticker: str, slope: float, n: int) -> list[Bar]:
        out: list[Bar] = []
        for i in range(n):
            price = 100.0 + slope * i
            out.append(
                Bar(
                    ticker=ticker,
                    day=date(2020, 1, 1) + timedelta(days=i),
                    open=price,
                    high=price + 1,
                    low=price - 1,
                    close=price,
                    adj_close=price,
                    volume=1000,
                    source="test",
                )
            )
        return out

    n = MIN_BARS + 40
    # Distinct slopes -> a strict, stable forward-return ordering A>B>C>D each day.
    panel = {
        "A": sloped("A", 2.0, n),
        "B": sloped("B", 1.0, n),
        "C": sloped("C", 0.5, n),
        "D": sloped("D", 0.25, n),
        "SPY": sloped("SPY", 1.0, n),
    }
    history = BarHistory(panel)
    samples = build_dataset(history, list(panel), horizon=5, relative=True)
    rows = labelled(samples)

    # The benchmark is never a labelled training row.
    assert all(s.ticker != "SPY" for s in rows)
    # Among A,B,C,D the two fastest beat the median, the two slowest do not.
    per_ticker = {t: {s.label.value for s in rows if s.ticker == t} for t in "ABCD"}
    assert per_ticker["A"] == {1} and per_ticker["B"] == {1}
    assert per_ticker["C"] == {0} and per_ticker["D"] == {0}


# --- walk-forward purge ---------------------------------------------------


class _RecordingModel:
    """Captures the size of each fold's training set instead of really fitting."""

    def __init__(self, sizes: list[int]) -> None:
        self._sizes = sizes

    def fit(self, X: list[list[float]], y: list[int]) -> _RecordingModel:
        self._sizes.append(len(X))
        return self

    def predict_proba(self, X: list[list[float]]) -> list[float]:
        return [0.5] * len(X)


def _sample(ticker: str, day: date, value: int, end_day: date) -> Sample:
    feats = {name: 0.0 for name in FEATURE_NAMES}
    return Sample(ticker, day, feats, Label(value=value, fwd_return=0.01, end_day=end_day))


def test_walk_forward_purges_boundary_spillover() -> None:
    samples: list[Sample] = []
    # 2020: five rows whose label finishes in 2020, plus one late-December row
    # whose 5-day window spills into 2021 (must be purged from the 2021 fold).
    for d in range(1, 6):
        samples.append(_sample("X", date(2020, 6, d), d % 2, date(2020, 6, d + 5)))
    samples.append(_sample("X", date(2020, 12, 30), 1, date(2021, 1, 4)))  # spill
    # 2021: four clean rows, plus one spilling into 2022.
    for d in range(1, 5):
        samples.append(_sample("X", date(2021, 6, d), d % 2, date(2021, 6, d + 5)))
    samples.append(_sample("X", date(2021, 12, 30), 0, date(2022, 1, 3)))  # spill
    # 2022: three rows so the 2022 fold has something to test on.
    for d in range(1, 4):
        samples.append(_sample("X", date(2022, 6, d), d % 2, date(2022, 6, d + 5)))

    sizes: list[int] = []
    walk_forward(samples, train_years=1, model_factory=lambda: _RecordingModel(sizes))

    # Fold 2021 trains only on 2020 rows finishing in 2020 -> 5 (spill excluded).
    # Fold 2022 trains on everything finishing before 2022 -> 6 + 4 = 10
    # (the 2021 December spill is excluded).
    assert sizes == [5, 10]


# --- model wrapper --------------------------------------------------------


def test_model_learns_a_planted_signal_and_round_trips(tmp_path) -> None:
    # Feature 0 deterministically drives the label; the rest is noise.
    X: list[list[float]] = []
    y: list[int] = []
    for i in range(400):
        signal = 1.0 if i % 2 == 0 else -1.0
        row = [signal] + [float((i % 11) - 5) for _ in FEATURE_NAMES[1:]]
        X.append(row)
        y.append(1 if signal > 0 else 0)

    model = GBMModel().fit(X, y)
    probs = model.predict_proba(X)
    # High-signal rows should score clearly above low-signal ones.
    hi = sum(p for p, label in zip(probs, y, strict=True) if label == 1)
    lo = sum(p for p, label in zip(probs, y, strict=True) if label == 0)
    assert hi / 200 > 0.7 and lo / 200 < 0.3

    path = model.save(tmp_path / "m.joblib")
    reloaded = GBMModel.load(path)
    assert reloaded.predict_proba(X[:3]) == probs[:3]
