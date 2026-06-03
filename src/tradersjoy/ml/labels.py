"""Define the *answer* the model is trained to predict: the label, or "y".

We chose a deliberately simple, honest target: **did this stock rise over the
next few trading days?** For a given day T we look ``horizon`` sessions ahead,
measure the return, and turn it into a yes/no label (1 if it beat a small
threshold, else 0). Framing it as up/down (a *classification*) instead of "guess
the exact return" keeps evaluation grounded: there is an obvious baseline to
beat, the base rate of up-days, so we can never fool ourselves that random
guessing looks clever.

The label is the one quantity that, by definition, can only be known in the
future. It is used *only* to train and score the model on history; it is never
available when we predict for today (that future has not happened yet). The
forward look here is therefore legitimate, the opposite of look-ahead leakage,
which is when future information sneaks into the *inputs*.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

#: Default horizon: how many trading days ahead the label looks. Matches the
#: project's 1-10 day swing timeframe.
DEFAULT_HORIZON: int = 5

#: Default decision threshold on the forward return. ``0.0`` means "any rise at
#: all counts as up". Because the market drifts upward, the resulting base rate
#: is usually a little above 50%, and *that* base rate, not 50%, is the honest
#: bar the model must clear.
DEFAULT_THRESHOLD: float = 0.0


@dataclass(frozen=True, slots=True)
class Label:
    """The future outcome attached to one (ticker, day) training row.

    Attributes:
        value: ``1`` if the forward return beat the threshold, else ``0``. This
            is the target the model learns to predict.
        fwd_return: The raw forward return over ``horizon`` days, kept so we can
            measure not just hit-rate but how much the model's picks actually
            moved (the bridge from accuracy to money).
        end_day: The calendar day of the bar ``horizon`` sessions ahead, i.e. the
            last day this label "knows about". The walk-forward split uses it to
            purge any training row whose answer window reaches into the test
            period, closing a subtle leak across the train/test boundary.
    """

    value: int
    fwd_return: float
    end_day: date


def forward_label(
    closes: list[float],
    days: list[date],
    index: int,
    horizon: int = DEFAULT_HORIZON,
    threshold: float = DEFAULT_THRESHOLD,
) -> Label | None:
    """Build the label for day ``index`` by looking ``horizon`` sessions ahead.

    Args:
        closes: A ticker's adjusted closes in ascending day order.
        days: The matching calendar days, same length and order as ``closes``.
        index: Position of the day being labelled ("T").
        horizon: How many trading days ahead to measure the return over.
        threshold: Forward return above which the day is labelled up (``1``).

    Returns:
        A :class:`Label`, or ``None`` if there are not ``horizon`` more sessions
        after ``index`` (the most recent days have no known future yet, so they
        cannot be training rows, only prediction inputs).
    """
    future = index + horizon
    if future >= len(closes):
        return None
    base = closes[index]
    if not base:
        return None
    fwd_return = closes[future] / base - 1.0
    value = 1 if fwd_return > threshold else 0
    return Label(value=value, fwd_return=fwd_return, end_day=days[future])
