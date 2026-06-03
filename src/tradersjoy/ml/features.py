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

Market-relative features:
    Most of any one stock's daily move is just the whole market moving together,
    which at a few days' horizon is close to unpredictable. So besides plain
    single-name features we add *relative* ones: this stock's recent return minus
    the market's (a benchmark such as SPY). They ask "did it beat the market",
    which is both more learnable and better matched to a strategy that picks the
    strongest few names from a basket. When no benchmark series is supplied these
    features are simply ``0.0`` (neutral), so single-ticker use still works.
"""

from __future__ import annotations

import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradersjoy.core.types import Bar

#: Ordered names of the features produced, the columns of the learning table.
#: The order is fixed so a saved model and live inputs always line up. Appending
#: (rather than reordering) keeps older intuitions about earlier columns intact.
FEATURE_NAMES: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_20",
    "mom_60",
    "dist_sma20",
    "dist_sma50",
    "dist_sma200",
    "vol_20",
    "volume_ratio",
    "rsi_14",
    "drawdown_20",
    "rel_ret_5",
    "rel_ret_20",
)

#: Number of recent daily changes used for the RSI oscillator.
_RSI_WINDOW: int = 14

#: Minimum bars of history required before any feature can be computed. Set by
#: the longest look-back used below (the 200-day trend line needs ``c[-201]``),
#: so a ticker's first ~200 sessions produce no rows. Skipping them is correct:
#: we simply have no opinion until there is enough history to form one.
MIN_BARS: int = 201


def benchmark_returns(closes: list[float]) -> dict[str, float] | None:
    """Compute the benchmark's own 5- and 20-day returns as of its last close.

    Args:
        closes: The benchmark's adjusted closes in ascending day order, ending on
            the day to describe.

    Returns:
        ``{"ret_5": ..., "ret_20": ...}`` for that day, or ``None`` if there is
        not yet enough benchmark history (fewer than 21 closes).
    """
    if len(closes) < 21:
        return None
    last = closes[-1]
    return {
        "ret_5": last / closes[-6] - 1.0 if closes[-6] else 0.0,
        "ret_20": last / closes[-21] - 1.0 if closes[-21] else 0.0,
    }


def features_from_bars(
    bars: list[Bar],
    benchmark: dict[str, float] | None = None,
) -> dict[str, float] | None:
    """Compute the feature vector for the *last* day in ``bars``.

    Args:
        bars: A ticker's bars in ascending day order, ending on the day to
            describe (i.e. "today"). Only this list is read, so the result can
            depend on nothing after that day.
        benchmark: The market benchmark's own returns for the *same* day, as
            returned by :func:`benchmark_returns` (e.g. SPY's ``ret_5``/``ret_20``).
            ``None`` makes the relative features ``0.0``.

    Returns:
        A mapping from each name in :data:`FEATURE_NAMES` to its value, or
        ``None`` if there is not yet enough history (fewer than :data:`MIN_BARS`
        bars) to compute every feature honestly.

    Notes:
        The features, in plain English:

        - ``ret_1`` / ``ret_5`` / ``ret_20``: the simple return over the last 1,
          5, and 20 trading days (recent momentum at three speeds).
        - ``mom_60``: the return over the last 60 days (slower trend).
        - ``dist_sma20`` / ``dist_sma50`` / ``dist_sma200``: how far today's price
          sits above (+) or below (-) its own 20-, 50-, and 200-day average, as a
          fraction. "Stretched" vs "depressed" against short, medium, long norms.
        - ``vol_20``: the standard deviation of the last 20 daily returns, a plain
          gauge of how jumpy the stock has been (its recent risk).
        - ``volume_ratio``: today's volume over the average of the last 20 days'.
        - ``rsi_14``: the classic Relative Strength Index over 14 days, rescaled
          to 0-1. Near 1 means it has risen on most recent days (overbought);
          near 0 the opposite (oversold).
        - ``drawdown_20``: how far below its highest close of the last 20 days the
          stock sits, as a fraction (0 at a fresh high, negative otherwise).
        - ``rel_ret_5`` / ``rel_ret_20``: this stock's 5- and 20-day return minus
          the benchmark's, i.e. how much it beat (+) or lagged (-) the market.
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

    ret_5 = ret(5)
    ret_20 = ret(20)
    bench = benchmark or {}

    return {
        "ret_1": ret(1),
        "ret_5": ret_5,
        "ret_20": ret_20,
        "mom_60": ret(60),
        "dist_sma20": dist_from_sma(20),
        "dist_sma50": dist_from_sma(50),
        "dist_sma200": dist_from_sma(200),
        "vol_20": vol_20,
        "volume_ratio": volume_ratio,
        "rsi_14": _rsi(closes, _RSI_WINDOW),
        "drawdown_20": last / max(closes[-20:]) - 1.0,
        "rel_ret_5": ret_5 - bench.get("ret_5", ret_5),
        "rel_ret_20": ret_20 - bench.get("ret_20", ret_20),
    }


def _rsi(closes: list[float], window: int) -> float:
    """Relative Strength Index over ``window`` days, rescaled to ``[0, 1]``.

    RSI compares the size of recent up moves to recent down moves. The standard
    formula yields 0-100; we divide by 100 so every feature stays on a small,
    comparable scale. With no down moves at all it saturates at 1.0.
    """
    changes = [closes[i] - closes[i - 1] for i in range(len(closes) - window, len(closes))]
    gains = sum(c for c in changes if c > 0)
    losses = -sum(c for c in changes if c < 0)
    if losses == 0:
        return 1.0 if gains > 0 else 0.5
    rs = (gains / window) / (losses / window)
    return (100.0 - 100.0 / (1.0 + rs)) / 100.0


def feature_row(features: dict[str, float]) -> list[float]:
    """Flatten a feature mapping into a fixed-order row for the model.

    Args:
        features: A mapping as returned by :func:`features_from_bars`.

    Returns:
        The feature values in :data:`FEATURE_NAMES` order, so every row handed to
        the model lines up column-for-column regardless of dict insertion order.
    """
    return [features[name] for name in FEATURE_NAMES]
