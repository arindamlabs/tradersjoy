"""Walk-forward *backtest*: what the model would have earned, after costs.

:mod:`tradersjoy.ml.walkforward` answers "is the ranking any good?" with AUC and
top-decile lift. That is necessary but not sufficient, because it says nothing
about what acting on the ranking costs. A model can rank correctly and still lose
money if collecting the edge means churning the book 70 times a year.

This module closes that gap. It runs the same year-by-year walk-forward, keeps
each fold's model, and then replays the *whole* out-of-sample span through the
real backtester with each year driven by the model that never saw it. The
resulting equity curve includes slippage, whole-position sizing, and the chosen
rebalance cadence, so it is directly comparable against buy-and-hold.

The distinction from ``backtest --strategy ml`` matters: that command loads one
model trained on all history and replays it over its own training window, which
is in-sample and flatters the model. Nothing here is in-sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import TYPE_CHECKING

from tradersjoy.backtest.engine import run_backtest
from tradersjoy.broker.sim import SimBroker
from tradersjoy.ml.dataset import build_dataset, labelled
from tradersjoy.ml.walkforward import walk_forward
from tradersjoy.strategy.base import BarContext, Strategy
from tradersjoy.strategy.baselines.buy_and_hold import BuyAndHold
from tradersjoy.strategy.ml.strategy import MLStrategy

if TYPE_CHECKING:
    from tradersjoy.backtest.data import BarHistory
    from tradersjoy.backtest.engine import BacktestResult
    from tradersjoy.core.types import Order
    from tradersjoy.ml.model import GBMModel
    from tradersjoy.ml.walkforward import WalkForwardResult


class WalkForwardMLStrategy(Strategy):
    """Drive each calendar year with the fold model that never saw that year.

    Structurally identical to :class:`~tradersjoy.strategy.ml.strategy.MLStrategy`
    from the engine's point of view; it just swaps which model is answering as
    the simulation crosses a year boundary. Days before the first test year
    propose nothing, since no model is yet entitled to an opinion about them.

    Attributes:
        tickers: The universe to rank.
        rebalance: Cadence passed through to each year's inner strategy.
    """

    def __init__(
        self,
        tickers: list[str],
        models_by_year: dict[int, GBMModel],
        top_k: int = 5,
        rebalance: str = "weekly",
    ) -> None:
        """Bind one inner strategy per test year.

        Args:
            tickers: Universe to rank.
            models_by_year: Fold models keyed by the year they are allowed to
                trade, as returned by
                :func:`~tradersjoy.ml.walkforward.walk_forward`.
            top_k: Names held at once.
            rebalance: One of :data:`~tradersjoy.strategy.cadence.CADENCES`.
        """
        self.tickers = tickers
        self.rebalance = rebalance
        self.top_k = top_k
        self._by_year = {
            year: MLStrategy(tickers, model, top_k=top_k, rebalance=rebalance)
            for year, model in models_by_year.items()
        }

    @property
    def name(self) -> str:
        return f"wf-ml(top{self.top_k},{self.rebalance})"

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Delegate to this year's model, or stand aside if there isn't one."""
        inner = self._by_year.get(ctx.day.year)
        return inner.on_bar(ctx) if inner is not None else []


@dataclass(frozen=True, slots=True)
class HorizonResult:
    """One arm of the horizon experiment.

    Attributes:
        horizon: Label look-ahead in trading sessions.
        rebalance: Cadence the arm traded on.
        classification: Out-of-sample ranking metrics.
        strategy: Out-of-sample backtest of the strategy.
        benchmark: Equal-weight buy-and-hold over the identical span.
        n_fills: How many fills the strategy needed.
        first_test_year: First out-of-sample year.
    """

    horizon: int
    rebalance: str
    classification: WalkForwardResult
    strategy: BacktestResult
    benchmark: BacktestResult
    n_fills: int
    first_test_year: int

    @property
    def excess_cagr(self) -> float:
        """Annualised return minus the benchmark's, the number that matters."""
        return self.strategy.metrics.cagr - self.benchmark.metrics.cagr

    @property
    def fills_per_year(self) -> float:
        """Fills per year, the direct driver of the slippage bill."""
        days = max(self.strategy.metrics.num_days, 1)
        return self.n_fills / (days / 365.25)


def evaluate_horizon(
    data: BarHistory,
    tickers: list[str],
    *,
    horizon: int,
    rebalance: str,
    relative: bool = True,
    train_years: int = 5,
    top_k: int = 5,
    slippage_bps: float = 5.0,
    cash: float = 100_000.0,
    threshold: float = 0.0,
) -> HorizonResult:
    """Train, walk-forward, and backtest one (horizon, cadence) pair.

    Args:
        data: Loaded bar panel.
        tickers: Universe to rank and trade.
        horizon: Label look-ahead in trading sessions.
        rebalance: Cadence to trade on; normally matched to ``horizon``.
        relative: Use the cross-sectional (beat-the-median) label.
        train_years: Years of history before the first test year.
        top_k: Names held at once.
        slippage_bps: Adverse slippage per fill.
        cash: Starting balance.
        threshold: Return cut the label must clear.

    Returns:
        A :class:`HorizonResult` for this arm.

    Raises:
        ValueError: If the dataset yields too few labelled rows to evaluate.
    """
    samples = build_dataset(
        data, tickers, horizon=horizon, threshold=threshold, relative=relative
    )
    rows = labelled(samples)
    if len(rows) < 500:
        raise ValueError(
            f"only {len(rows)} labelled rows at horizon {horizon}; too few to score"
        )

    wf = walk_forward(rows, train_years=train_years)
    if not wf.models:
        raise ValueError(f"no walk-forward folds at horizon {horizon}")

    first_test_year = min(wf.models)
    start = date(first_test_year, 1, 1)
    end = data.trading_days[-1]

    strategy = WalkForwardMLStrategy(
        tickers, wf.models, top_k=top_k, rebalance=rebalance
    )
    result = run_backtest(
        strategy, data, SimBroker(slippage_bps=slippage_bps), cash, start, end
    )
    # Same span, same costs, same engine: the only difference is the decision.
    bench = run_backtest(
        BuyAndHold(tickers), data, SimBroker(slippage_bps=slippage_bps), cash, start, end
    )

    return HorizonResult(
        horizon=horizon,
        rebalance=rebalance,
        classification=wf,
        strategy=result,
        benchmark=bench,
        n_fills=len(result.fills),
        first_test_year=first_test_year,
    )
