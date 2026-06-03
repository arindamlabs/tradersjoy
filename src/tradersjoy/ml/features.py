"""Turn a ticker's price history into the numeric inputs a model learns from.

A *feature* is just a number describing what a stock looked like on one day, the
model's "X". This module computes a small, deliberately plain set of features
from a ticker's bars *up to and including* a given day, and nothing after it.
That last clause is the whole point: every feature here is a function of the past
only, so it is structurally impossible to leak future information into the inputs.

The exact same function is called in two places:

- at *training* time, once per historical day, to build the learning table, and
- at *prediction* time, on today's bars, to ask the model what it thinks now.

Using one implementation for both is what prevents "train/serve skew", the
classic bug where the features a model is trained on subtly differ from the ones
it is shown live, quietly destroying its real-world accuracy.

Features use the adjusted close (continuous across splits and dividends), for the
same reason indicators do: a mechanical split should not look like a price move.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradersjoy.core.types import Bar

#: Ordered names of the features produced, the columns of the learning table.
#: The order is fixed so a saved model and live inputs always line up.
FEATURE_NAMES: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_20",
    "mom_60",
    "dist_sma20",
    "dist_sma50",
    "vol_20",
    "volume_ratio",
)

#: Minimum bars of history required before any feature can be computed. Set by
#: the longest look-back used below (the 60-day momentum needs ``c[-61]``), so a
#: ticker's first ~60 sessions produce no rows. Skipping them is correct: we
#: simply have no opinion until there is enough history to form one.
MIN_BARS: int = 61


def features_from_bars(bars: list[Bar]) -> dict[str, float] | None:
    """Compute the feature vector for the *last* day in ``bars``.

    Args:
        bars: A ticker's bars in ascending day order, ending on the day to
            describe (i.e. "today"). Only this list is read, so the result can
            depend on nothing after that day.

    Returns:
        A mapping from each name in :data:`FEATURE_NAMES` to its value, or
        ``None`` if there is not yet enough history (fewer than :data:`MIN_BARS`
        bars) to compute every feature honestly.

    Notes:
        The features, in plain English:

        - ``ret_1`` / ``ret_5`` / ``ret_20``: the simple return over the last 1,
          5, and 20 trading days (recent momentum at three speeds).
        - ``mom_60``: the return over the last 60 days (slower trend).
        - ``dist_sma20`` / ``dist_sma50``: how far today's price sits above (+)
          or below (-) its own 20- and 50-day average, as a fraction. A measure
          of "stretched" vs "depressed" relative to the recent norm.
        - ``vol_20``: the standard deviation of the last 20 daily returns, a
          plain gauge of how jumpy the stock has been (its recent risk).
        - ``volume_ratio``: today's volume divided by the average of the last 20
          days', so >1 means unusually heavy trading.
    """
    if len(bars) < MIN_BARS:
        return None

    closes = [b.adj_close for b in bars]
    volumes = [b.volume for b in bars]
    last = closes[-1]

    def ret(n: int) -> float:
        prior = closes[-1 - n]
        return last / prior - 1.0 if prior else 0.0

    def dist_from_sma(n: int) -> float:
        sma = sum(closes[-n:]) / n
        return last / sma - 1.0 if sma else 0.0

    daily_returns = [
        closes[i] / closes[i - 1] - 1.0
        for i in range(len(closes) - 20, len(closes))
        if closes[i - 1]
    ]
    vol_20 = statistics.pstdev(daily_returns) if len(daily_returns) > 1 else 0.0

    avg_volume = sum(volumes[-20:]) / 20
    volume_ratio = volumes[-1] / avg_volume if avg_volume else 1.0

    return {
        "ret_1": ret(1),
        "ret_5": ret(5),
        "ret_20": ret(20),
        "mom_60": ret(60),
        "dist_sma20": dist_from_sma(20),
        "dist_sma50": dist_from_sma(50),
        "vol_20": vol_20,
        "volume_ratio": volume_ratio,
    }


def feature_row(features: dict[str, float]) -> list[float]:
    """Flatten a feature mapping into a fixed-order row for the model.

    Args:
        features: A mapping as returned by :func:`features_from_bars`.

    Returns:
        The feature values in :data:`FEATURE_NAMES` order, so every row handed to
        the model lines up column-for-column regardless of dict insertion order.
    """
    return [features[name] for name in FEATURE_NAMES]
