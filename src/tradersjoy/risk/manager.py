"""The risk layer: wrap any strategy and enforce the rails on its orders.

:class:`RiskManagedStrategy` is itself a :class:`~tradersjoy.strategy.base.Strategy`,
so it slots into the exact same seam the engine and the live trader already use.
Nothing downstream knows or cares that risk management is present: the wrapped
strategy proposes orders, the wrapper rewrites them, and the broker sees only the
rewritten set.

The whole layer is **stateless**: every decision is recomputed each day from
inputs both the backtest and the live account expose identically (current
quantities, the broker-reported cost basis, and price history). It therefore
behaves the same in a backtest and live, with no remembered state that a live
restart could silently lose, which is the trap a trailing stop or a peak-equity
breaker would fall into.

The pipeline each day, in order:

1. **Stop-loss.** Force a full exit of any held name trading at least
   ``stop_loss`` below its cost basis, and drop any proposed buy of that name.
2. **Circuit breaker.** If the benchmark is in a deep drawdown, drop *all* new
   buys (exits, including stops, still go through).
3. **Sizing.** Trim surviving buys so no single name exceeds
   ``max_position_weight`` of equity and total invested never exceeds
   ``max_gross_exposure`` (no margin).
"""

from __future__ import annotations

from tradersjoy.core.types import Order, Side
from tradersjoy.risk.limits import RiskLimits
from tradersjoy.strategy.base import BarContext, Strategy

#: Dollar/share tolerance so floating-point dust never emits a tiny phantom order.
_EPS = 1e-9


class RiskManagedStrategy(Strategy):
    """Wrap a strategy and enforce :class:`RiskLimits` on every order it proposes.

    Attributes:
        tickers: The universe to police. Stops and exposure are evaluated across
            these names, so it should match the inner strategy's universe.
        inner: The wrapped strategy whose orders are rewritten.
        limits: The numeric rails to enforce.
    """

    def __init__(
        self,
        tickers: list[str],
        inner: Strategy,
        limits: RiskLimits | None = None,
    ) -> None:
        """Wire a risk layer around an inner strategy.

        Args:
            tickers: Universe to police (match the inner strategy's universe).
            inner: The strategy whose proposed orders will be risk-managed.
            limits: Limits to enforce; defaults to :class:`RiskLimits` defaults.
        """
        self.tickers = tickers
        self.inner = inner
        self.limits = limits or RiskLimits()

    @property
    def name(self) -> str:
        return f"risk({self.inner.name})"

    def _price(self, ctx: BarContext, ticker: str) -> float | None:
        """Latest known raw price for ``ticker``: today's close, else last close."""
        bar = ctx.bars.get(ticker)
        if bar is not None:
            return bar.close
        hist = ctx.history.history(ticker, ctx.day)
        return hist[-1].close if hist else None

    def _stops(self, ctx: BarContext) -> set[str]:
        """Names trading at least ``stop_loss`` below their cost basis."""
        if self.limits.stop_loss is None:
            return set()
        trigger = 1.0 - self.limits.stop_loss
        stopped: set[str] = set()
        for ticker in self.tickers:
            if ctx.portfolio.qty(ticker) <= 0:
                continue
            cost = ctx.portfolio.avg_cost(ticker)
            price = self._price(ctx, ticker)
            if cost > 0 and price is not None and price <= cost * trigger:
                stopped.add(ticker)
        return stopped

    def _market_crashing(self, ctx: BarContext) -> bool:
        """True if the benchmark sits ``crash_drawdown`` or more below its high."""
        lim = self.limits
        if lim.crash_drawdown is None:
            return False
        closes = ctx.history.adj_closes(lim.benchmark, ctx.day)
        if len(closes) < 2:
            return False
        window = closes[-lim.crash_window :] if lim.crash_window > 0 else closes
        peak = max(window)
        if peak <= 0:
            return False
        drawdown = closes[-1] / peak - 1.0
        return drawdown <= -lim.crash_drawdown

    def on_bar(self, ctx: BarContext) -> list[Order]:
        """Rewrite the inner strategy's orders to respect the risk limits."""
        proposed = self.inner.on_bar(ctx)
        stops = self._stops(ctx)

        # Collect sells: the inner strategy's exits (capped at what is held), plus
        # a forced full exit for every stopped name. Stops win over any proposal.
        sells: dict[str, tuple[float, str]] = {}
        buys: list[Order] = []
        for o in proposed:
            if o.side is Side.SELL:
                held = ctx.portfolio.qty(o.ticker)
                sells[o.ticker] = (min(o.quantity, held), o.tag or "exit")
            elif o.ticker not in stops:  # never buy a name we are stopping out
                buys.append(o)
        for ticker in stops:
            sells[ticker] = (ctx.portfolio.qty(ticker), "risk-stop")

        # Circuit breaker: in a deep market drawdown, stop adding risk entirely.
        if self._market_crashing(ctx):
            buys = []

        buys = self._size(ctx, buys, sells)

        orders: list[Order] = [
            Order(ticker, Side.SELL, qty, tag=tag)
            for ticker, (qty, tag) in sells.items()
            if qty > _EPS
        ]
        orders.extend(buys)
        return orders

    def _size(
        self,
        ctx: BarContext,
        buys: list[Order],
        sells: dict[str, tuple[float, str]],
    ) -> list[Order]:
        """Trim buys to the per-name and gross-exposure caps, in proposed order.

        Names being (fully or partly) sold free up room; everything is valued at
        the latest known price. Buys keep their order's relative priority, so when
        the budget runs out the lowest-priority names are the ones dropped.
        """
        lim = self.limits
        equity = ctx.portfolio.equity
        if equity <= 0:
            return []

        def remaining_value(ticker: str) -> float:
            held = ctx.portfolio.qty(ticker)
            held -= sells.get(ticker, (0.0, ""))[0]
            price = self._price(ctx, ticker)
            return max(0.0, held) * price if price is not None else 0.0

        kept_value = sum(remaining_value(t) for t in self.tickers)
        budget = max(0.0, lim.max_gross_exposure * equity - kept_value)
        per_name_cap = lim.max_position_weight * equity

        sized: list[Order] = []
        for o in buys:
            price = self._price(ctx, o.ticker)
            if price is None or price <= 0:
                continue
            room_for_name = max(0.0, per_name_cap - remaining_value(o.ticker))
            dollars = min(o.quantity * price, room_for_name, budget)
            if dollars <= _EPS:
                continue
            budget -= dollars
            sized.append(Order(o.ticker, Side.BUY, dollars / price, tag=o.tag or "risk-sized"))
        return sized
