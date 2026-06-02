"""Name-to-strategy lookup so the CLI can build a strategy from a string.

Keeping construction in one place means ``tradersjoy backtest --strategy sma``
has a single, discoverable source of truth for which names exist and how each is
parameterised, instead of a chain of ``if`` branches inside the CLI.
"""

from __future__ import annotations

from tradersjoy.strategy.base import Strategy
from tradersjoy.strategy.baselines.buy_and_hold import BuyAndHold
from tradersjoy.strategy.baselines.sma_crossover import SMACrossover

#: The strategy names accepted by the CLI, for help text and validation.
STRATEGY_NAMES = ("buyhold", "sma")


def build_strategy(
    name: str,
    tickers: list[str],
    short_window: int = 20,
    long_window: int = 50,
) -> Strategy:
    """Construct a strategy by name for the given universe.

    Args:
        name: One of :data:`STRATEGY_NAMES` (case-insensitive). ``"buyhold"``
            builds :class:`~tradersjoy.strategy.baselines.buy_and_hold.BuyAndHold`;
            ``"sma"`` builds
            :class:`~tradersjoy.strategy.baselines.sma_crossover.SMACrossover`.
        tickers: Universe to trade.
        short_window: Fast SMA length (``sma`` only).
        long_window: Slow SMA length (``sma`` only).

    Returns:
        The constructed :class:`~tradersjoy.strategy.base.Strategy`.

    Raises:
        ValueError: If ``name`` is not a known strategy.
    """
    key = name.strip().lower()
    if key in ("buyhold", "buy_and_hold"):
        return BuyAndHold(tickers)
    if key in ("sma", "sma_crossover"):
        return SMACrossover(tickers, short_window=short_window, long_window=long_window)
    raise ValueError(
        f"Unknown strategy {name!r}. Choose from: {', '.join(STRATEGY_NAMES)}."
    )
