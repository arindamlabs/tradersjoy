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
STRATEGY_NAMES = ("buyhold", "sma", "ml")


def build_strategy(
    name: str,
    tickers: list[str],
    short_window: int = 20,
    long_window: int = 50,
    model_path: str | None = None,
    top_k: int = 5,
    risk: bool = False,
) -> Strategy:
    """Construct a strategy by name for the given universe.

    Args:
        name: One of :data:`STRATEGY_NAMES` (case-insensitive). ``"buyhold"``
            builds :class:`~tradersjoy.strategy.baselines.buy_and_hold.BuyAndHold`;
            ``"sma"`` builds
            :class:`~tradersjoy.strategy.baselines.sma_crossover.SMACrossover`;
            ``"ml"`` loads a trained model and builds
            :class:`~tradersjoy.strategy.ml.strategy.MLStrategy`.
        tickers: Universe to trade.
        short_window: Fast SMA length (``sma`` only).
        long_window: Slow SMA length (``sma`` only).
        model_path: Path to a model saved by ``tradersjoy train`` (``ml`` only,
            required).
        top_k: Number of names the ML strategy holds at once (``ml`` only).
        risk: If true, wrap the built strategy in a
            :class:`~tradersjoy.risk.manager.RiskManagedStrategy` so position
            sizing, the exposure cap, the stop-loss, and the circuit breaker are
            enforced on its orders.

    Returns:
        The constructed :class:`~tradersjoy.strategy.base.Strategy`, optionally
        wrapped in the risk layer.

    Raises:
        ValueError: If ``name`` is not a known strategy, or ``ml`` is requested
            without a ``model_path``.
    """
    key = name.strip().lower()
    if key in ("buyhold", "buy_and_hold"):
        inner: Strategy = BuyAndHold(tickers)
    elif key in ("sma", "sma_crossover"):
        inner = SMACrossover(tickers, short_window=short_window, long_window=long_window)
    elif key == "ml":
        if not model_path:
            raise ValueError(
                "the 'ml' strategy needs a trained model; pass --model PATH "
                "(train one first with `tradersjoy train`)."
            )
        # Imported lazily so non-ML commands never pay for the ML stack.
        from tradersjoy.strategy.ml.strategy import MLStrategy

        inner = MLStrategy.from_path(tickers, model_path, top_k=top_k)
    else:
        raise ValueError(
            f"Unknown strategy {name!r}. Choose from: {', '.join(STRATEGY_NAMES)}."
        )

    if risk:
        from tradersjoy.risk.manager import RiskManagedStrategy

        return RiskManagedStrategy(tickers, inner)
    return inner
