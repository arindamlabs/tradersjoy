"""The ML strategy: rank the universe by a trained model and hold the best names.

This is where a model earns its keep. Each day it scores every tradable ticker
with the probability that the stock rises over the next few sessions, ranks them,
and aims to hold the top handful equal-weight. Names that fall out of the top
group are sold; new entrants are bought. It is intentionally simple, a clean
"buy what the model likes most" rule, so that any edge in the backtest is the
model's, not a pile of hand-tuned trading heuristics on top of it.

Crucially it computes features with the *same* :func:`features_from_bars` used in
training, reading only history up to the decision day. So the strategy inherits
the no-look-ahead guarantee for free, and there is no train/serve skew: the model
sees live inputs shaped exactly like the ones it learned from.
"""

from __future__ import annotations

from tradersjoy.core.types import Order, Side
from tradersjoy.ml.dataset import DEFAULT_BENCHMARK
from tradersjoy.ml.features import benchmark_returns, features_from_bars
from tradersjoy.ml.model import GBMModel
from tradersjoy.strategy.base import BarContext, Strategy
from tradersjoy.strategy.cadence import CADENCES, is_rebalance_day


class MLStrategy(Strategy):
    """Hold the ``top_k`` names a trained model scores most likely to rise.

    Attributes:
        tickers: The universe to rank each day.
        model: The fitted model used to score names.
        top_k: How many names to hold at once, equal-weight.
        invest_fraction: Fraction of equity to deploy across the held names,
            leaving a buffer against next-open slippage.
        min_score: Only buy a name whose probability clears this floor, so the
            strategy can sit in cash when the model likes nothing.
        rebalance: How often the ranking is acted on. Should match the model's
            label horizon; see :mod:`tradersjoy.strategy.cadence`.
    """

    def __init__(
        self,
        tickers: list[str],
        model: GBMModel,
        top_k: int = 5,
        invest_fraction: float = 0.95,
        min_score: float = 0.5,
        benchmark: str = DEFAULT_BENCHMARK,
        rebalance: str = "weekly",
    ) -> None:
        """Configure the strategy around a pre-fitted model.

        Args:
            tickers: Universe to rank.
            model: A fitted :class:`~tradersjoy.ml.model.GBMModel`.
            top_k: Number of names to hold at once.
            invest_fraction: Fraction of equity to spread across held names.
            min_score: Minimum model probability required to buy a name.
            benchmark: Market symbol for the relative features; must match what
                the model was trained with (default ``SPY``).
            rebalance: One of :data:`~tradersjoy.strategy.cadence.CADENCES`.
                Defaults to ``"weekly"``, the roughly five sessions the default
                label predicts over. Acting on the ranking every session instead
                churns the book about 70x a year for a 6.7%/year slippage drag,
                and sells names before the move they were picked for has had
                time to happen.

        Raises:
            ValueError: If ``rebalance`` is not a recognised cadence.
        """
        if rebalance not in CADENCES:
            raise ValueError(
                f"Unknown rebalance cadence {rebalance!r}. "
                f"Choose from: {', '.join(CADENCES)}."
            )
        self.tickers = tickers
        self.model = model
        self.top_k = top_k
        self.invest_fraction = invest_fraction
        self.min_score = min_score
        self.benchmark = benchmark
        self.rebalance = rebalance

    @classmethod
    def from_path(cls, tickers: list[str], model_path: str, **kwargs: object) -> MLStrategy:
        """Build the strategy by loading a model saved by ``tradersjoy train``.

        Args:
            tickers: Universe to rank.
            model_path: File written by :meth:`~tradersjoy.ml.model.GBMModel.save`.
            **kwargs: Forwarded to :class:`MLStrategy` (``top_k``, etc.).

        Returns:
            A ready-to-run :class:`MLStrategy`.
        """
        return cls(tickers, GBMModel.load(model_path), **kwargs)  # type: ignore[arg-type]

    @property
    def name(self) -> str:
        # The cadence is part of the name so every journal row and backtest
        # scorecard records which one produced it. Two runs of "ml(top5)" that
        # rebalanced differently are not the same strategy.
        return f"ml(top{self.top_k},{self.rebalance})"

    def _scores(self, ctx: BarContext) -> dict[str, float]:
        """Score every ticker that trades today and has enough history."""
        # The market's own recent return today, shared across all names, so the
        # relative features match exactly how the model was trained.
        bench = benchmark_returns(ctx.history.adj_closes(self.benchmark, ctx.day))

        candidates: list[str] = []
        rows: list[list[float]] = []
        for ticker in self.tickers:
            if ticker not in ctx.bars:
                continue  # not trading today; cannot price an order anyway
            feats = features_from_bars(
                ctx.history.history(ticker, ctx.day), benchmark=bench
            )
            if feats is None:
                continue  # not enough history yet
            candidates.append(ticker)
            rows.append([feats[name] for name in self.model.feature_names])
        if not rows:
            return {}
        probs = self.model.predict_proba(rows)
        return dict(zip(candidates, probs, strict=True))

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Rebalance toward the top-``k`` highest-scoring names above ``min_score``.

        On sessions that are not a rebalance day the strategy proposes nothing
        and the book is left alone. Note this suppresses only the *strategy's*
        churn: a risk layer wrapping it still evaluates its rails every session,
        so a stop-loss fires on the day it is breached rather than waiting for
        the next rebalance.
        """
        if not is_rebalance_day(ctx.day, ctx.history.trading_days, self.rebalance):
            return []

        scores = self._scores(ctx)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        target = {t for t, s in ranked[: self.top_k] if s >= self.min_score}

        orders: list[Order] = []
        # Exit names we hold that are no longer in the target set.
        for ticker in self.tickers:
            held = ctx.portfolio.qty(ticker)
            if held > 0 and ticker not in target:
                orders.append(Order(ticker, Side.SELL, held, tag="ml-exit"))

        # Enter target names we do not already hold, equal-weight.
        if target:
            per_name = ctx.portfolio.equity * self.invest_fraction / self.top_k
            for ticker in target:
                if ctx.portfolio.qty(ticker) > 0:
                    continue  # already holding; let it ride
                bar = ctx.bars.get(ticker)
                if bar is None or bar.close <= 0:
                    continue
                qty = per_name / bar.close
                if qty > 0:
                    orders.append(Order(ticker, Side.BUY, qty, tag="ml-entry"))
        return orders
