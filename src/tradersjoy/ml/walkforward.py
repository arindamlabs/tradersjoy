"""Walk-forward validation: the only honest way to score a trading model.

The idea in one sentence: **always test on dates that come strictly after the
dates the model learned from**, mimicking how it would have run in real time.

Concretely we step year by year. To score year Y we train on every row from
before Y and predict year Y's days; then we roll forward and do the same for
Y+1, and so on. Stitching all those out-of-sample years together gives a track
record that is the closest thing to "what if I had actually run this live".

Why not just shuffle all the days and hold out a random 20%? Because that puts
2024 in the training set while testing on 2019, letting the model "predict" the
past from its own future. It is the single most common way beginners produce a
gorgeous backtest that means nothing. We never do it.

One subtlety this module gets right: the **purge**. Our label for day T looks 5
days ahead, so a row dated late-December "knows" something about early January.
If that row stayed in the training set while January is the test year, a sliver
of the future would leak across the boundary. So before each fold we drop any
training row whose label window (its ``end_day``) reaches into the test year.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date

from tradersjoy.ml.dataset import Sample
from tradersjoy.ml.metrics import ClassificationMetrics, evaluate
from tradersjoy.ml.model import GBMModel

#: A factory for a fresh, unfitted model per fold. Default builds the standard
#: GBM; tests and experiments can substitute a cheaper or different estimator.
ModelFactory = Callable[[], GBMModel]


@dataclass(frozen=True, slots=True)
class Prediction:
    """One out-of-sample prediction, kept for scoring and inspection.

    Attributes:
        ticker: Symbol predicted.
        day: The day predicted on.
        score: Model probability of the up-label.
        label: The true ``0/1`` outcome.
        fwd_return: The actual forward return for that row.
    """

    ticker: str
    day: date
    score: float
    label: int
    fwd_return: float


@dataclass(frozen=True, slots=True)
class Fold:
    """Bookkeeping for one train-past / test-future split.

    Attributes:
        test_year: The calendar year held out and predicted.
        n_train: Rows used to train (after purging the boundary spillover).
        n_test: Rows scored in the test year.
        metrics: The scorecard for this year alone, or ``None`` if it could not
            be scored (e.g. the test year had only one class).
    """

    test_year: int
    n_train: int
    n_test: int
    metrics: ClassificationMetrics | None


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """The full out-of-sample evaluation across every fold.

    Attributes:
        folds: Per-year bookkeeping and scorecards.
        predictions: Every out-of-sample prediction, across all years, in order.
        overall: The scorecard computed over all predictions stitched together,
            or ``None`` if nothing could be scored.
    """

    folds: list[Fold]
    predictions: list[Prediction]
    overall: ClassificationMetrics | None = field(default=None)

    def summary(self) -> str:
        """Render the per-year table plus the stitched overall scorecard."""
        lines = ["Walk-forward (out-of-sample) results", ""]
        lines.append(f"{'year':>6}  {'train':>8}  {'test':>7}  {'acc':>7}  {'AUC':>6}")
        for f in self.folds:
            if f.metrics is None:
                lines.append(
                    f"{f.test_year:>6}  {f.n_train:>8,}  {f.n_test:>7,}  "
                    f"{'n/a':>7}  {'n/a':>6}"
                )
            else:
                lines.append(
                    f"{f.test_year:>6}  {f.n_train:>8,}  {f.n_test:>7,}  "
                    f"{f.metrics.accuracy:>6.2%}  {f.metrics.auc:>6.3f}"
                )
        if self.overall is not None:
            lines.append("")
            lines.append("Overall, all test years stitched together:")
            lines.append(self.overall.summary())
        return "\n".join(lines)


def _default_factory() -> GBMModel:
    return GBMModel()


def walk_forward(
    samples: list[Sample],
    train_years: int = 5,
    model_factory: ModelFactory = _default_factory,
) -> WalkForwardResult:
    """Run an expanding-window, year-by-year walk-forward over labelled samples.

    Args:
        samples: The learning table. Only labelled rows are used; pass the full
            dataset and unlabelled (most-recent) rows are ignored here.
        train_years: How many initial years of history to require before the
            first test year. The first ``train_years`` of data are training-only.
        model_factory: Builds a fresh, unfitted model for each fold.

    Returns:
        A :class:`WalkForwardResult` with per-year folds, every out-of-sample
        prediction, and the stitched overall scorecard.

    Raises:
        ValueError: If there are no labelled samples to evaluate.
    """
    rows = [s for s in samples if s.label is not None]
    if not rows:
        raise ValueError("no labelled samples to evaluate")

    years = sorted({s.day.year for s in rows})
    first_test_year = years[0] + train_years
    test_years = [y for y in years if y >= first_test_year]

    folds: list[Fold] = []
    predictions: list[Prediction] = []

    for test_year in test_years:
        boundary = date(test_year, 1, 1)
        # Train on rows whose entire label window finished before the test year
        # (this is the purge: late-December rows that peek into January are out).
        train = [s for s in rows if s.label.end_day < boundary]
        test = [s for s in rows if s.day.year == test_year]
        if not train or not test:
            continue

        model = model_factory()
        model.fit([s.row for s in train], [s.label.value for s in train])
        scores = model.predict_proba([s.row for s in test])

        fold_preds = [
            Prediction(
                ticker=s.ticker,
                day=s.day,
                score=score,
                label=s.label.value,
                fwd_return=s.label.fwd_return,
            )
            for s, score in zip(test, scores, strict=True)
        ]
        predictions.extend(fold_preds)

        fold_metrics = _score(fold_preds)
        folds.append(
            Fold(
                test_year=test_year,
                n_train=len(train),
                n_test=len(test),
                metrics=fold_metrics,
            )
        )

    overall = _score(predictions)
    return WalkForwardResult(folds=folds, predictions=predictions, overall=overall)


def _score(preds: list[Prediction]) -> ClassificationMetrics | None:
    """Score a batch of predictions, or ``None`` if it cannot be scored."""
    if not preds:
        return None
    try:
        return evaluate(
            [p.label for p in preds],
            [p.score for p in preds],
            [p.fwd_return for p in preds],
        )
    except ValueError:
        return None
