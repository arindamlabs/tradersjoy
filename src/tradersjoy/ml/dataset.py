"""Assemble the learning table: one row per (ticker, day) with features + label.

This is where the feature inputs (:mod:`tradersjoy.ml.features`) and the future
answer (:mod:`tradersjoy.ml.labels`) are joined into the flat table a model
learns from. Each :class:`Sample` is one row: "on this day, this ticker looked
like *these numbers*, and over the next few days it did *this*."

Two honesty properties are enforced here, not left to the caller:

- **No look-ahead in the inputs.** Features for day T are computed from
  ``history(ticker, T)``, which the :class:`~tradersjoy.backtest.data.BarHistory`
  guarantees contains nothing after T.
- **No label without a real future.** A row only gets a label if ``horizon`` more
  sessions actually exist after it. The most recent days therefore yield
  *unlabelled* samples, which are exactly what we feed the model to predict
  today, never to train on.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from tradersjoy.backtest.data import BarHistory
from tradersjoy.ml.features import MIN_BARS, feature_row, features_from_bars
from tradersjoy.ml.labels import (
    DEFAULT_HORIZON,
    DEFAULT_THRESHOLD,
    Label,
    forward_label,
)


@dataclass(frozen=True, slots=True)
class Sample:
    """One row of the learning table.

    Attributes:
        ticker: The symbol this row describes.
        day: The day the features are measured as of (its close is known).
        features: The feature mapping (see :data:`~tradersjoy.ml.features.FEATURE_NAMES`).
        label: The future outcome, or ``None`` for the most recent days whose
            ``horizon``-ahead future has not happened yet. Unlabelled rows are
            prediction inputs only and must never be used to train or score.
    """

    ticker: str
    day: date
    features: dict[str, float]
    label: Label | None

    @property
    def row(self) -> list[float]:
        """The feature values in fixed model order (see :func:`feature_row`)."""
        return feature_row(self.features)


def samples_for_ticker(
    history: BarHistory,
    ticker: str,
    horizon: int = DEFAULT_HORIZON,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Sample]:
    """Build every labelled-or-not sample for one ticker over its full history.

    Args:
        history: The loaded bar panel.
        ticker: Symbol to build rows for.
        horizon: Forward look, in sessions, used to label each row.
        threshold: Forward-return threshold separating up (``1``) from down.

    Returns:
        Samples in ascending day order, one per day that has at least
        :data:`~tradersjoy.ml.features.MIN_BARS` bars of prior history. Rows in
        the final ``horizon`` days are present but carry ``label=None``.
    """
    bars = history.history(ticker, history.trading_days[-1]) if history.trading_days else []
    if len(bars) < MIN_BARS:
        return []

    closes = [b.adj_close for b in bars]
    days = [b.day for b in bars]
    out: list[Sample] = []
    for i in range(MIN_BARS - 1, len(bars)):
        feats = features_from_bars(bars[: i + 1])
        if feats is None:  # defensive; MIN_BARS guarantees it is not
            continue
        label = forward_label(closes, days, i, horizon=horizon, threshold=threshold)
        out.append(Sample(ticker=ticker, day=days[i], features=feats, label=label))
    return out


def build_dataset(
    history: BarHistory,
    tickers: list[str],
    horizon: int = DEFAULT_HORIZON,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[Sample]:
    """Assemble the full learning table across every ticker, sorted by day.

    Args:
        history: The loaded bar panel.
        tickers: Symbols to include.
        horizon: Forward look, in sessions, used to label each row.
        threshold: Forward-return threshold separating up from down.

    Returns:
        All samples across tickers, sorted by ``(day, ticker)`` so a walk-forward
        split can slice cleanly along the time axis.
    """
    samples: list[Sample] = []
    for ticker in tickers:
        samples.extend(
            samples_for_ticker(history, ticker, horizon=horizon, threshold=threshold)
        )
    samples.sort(key=lambda s: (s.day, s.ticker))
    return samples


def labelled(samples: list[Sample]) -> list[Sample]:
    """Return only the samples that have a known future label (for train/test)."""
    return [s for s in samples if s.label is not None]


def matrix(samples: list[Sample]) -> tuple[list[list[float]], list[int]]:
    """Split labelled samples into the feature matrix ``X`` and target list ``y``.

    Args:
        samples: Labelled samples (call :func:`labelled` first).

    Returns:
        ``(X, y)`` where ``X`` is a list of feature rows and ``y`` the matching
        ``0/1`` labels, ready to hand to a model.

    Raises:
        ValueError: If any sample is unlabelled (a programming error: train/score
            must never see a row whose future is unknown).
    """
    X: list[list[float]] = []
    y: list[int] = []
    for s in samples:
        if s.label is None:
            raise ValueError(f"unlabelled sample for {s.ticker} on {s.day} cannot train")
        X.append(s.row)
        y.append(s.label.value)
    return X, y
