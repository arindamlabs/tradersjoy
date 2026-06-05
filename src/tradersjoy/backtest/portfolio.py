"""Portfolio accounting: cash, positions, realised trades, and the equity curve.

The :class:`Portfolio` is the single source of truth for "what do we own and what
is it worth?" during a run. It is mutated only by applying fills (from the
broker) and by marking to market (from the engine, once per day). It deliberately
holds no opinion about strategy or execution; it just keeps the books.

Two notions of profit live here and should not be confused:

- **Realised P&L** is locked in when shares are sold; each sale appends a
  :class:`~tradersjoy.core.types.Trade`. Hit rate is computed from these.
- **Equity** (cash plus the marked value of open positions) also moves with
  *unrealised* gains, and is what the equity curve and return metrics track.
"""

from __future__ import annotations

from datetime import date

from tradersjoy.core.types import Fill, Position, Side, Trade

#: Quantity below which a position is treated as fully closed and dropped, so
#: floating-point residue never leaves a phantom 1e-15-share holding behind.
_DUST = 1e-9


class Portfolio:
    """Mutable book of cash, open positions, closed trades, and equity history.

    Attributes:
        starting_cash: Cash the portfolio began with, the baseline for returns.
        cash: Uninvested cash currently available.
        positions: Open long positions keyed by ticker.
        trades: Closed round trips, appended each time shares are sold.
        fills: Every fill applied, in order, for auditing.
        equity_curve: ``(day, equity)`` points, one per :meth:`mark` call.
    """

    def __init__(self, starting_cash: float) -> None:
        """Open an all-cash portfolio.

        Args:
            starting_cash: Initial cash balance (e.g. ``100_000``).
        """
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: dict[str, Position] = {}
        self.trades: list[Trade] = []
        self.fills: list[Fill] = []
        self.equity_curve: list[tuple[date, float]] = []
        self._last_price: dict[str, float] = {}

    def qty(self, ticker: str) -> float:
        """Return shares currently held in ``ticker`` (``0.0`` if none)."""
        pos = self.positions.get(ticker)
        return pos.quantity if pos else 0.0

    def avg_cost(self, ticker: str) -> float:
        """Return the commission-inclusive average cost of ``ticker`` (``0.0`` if none)."""
        pos = self.positions.get(ticker)
        return pos.avg_cost if pos else 0.0

    @property
    def equity(self) -> float:
        """Total portfolio value: cash plus open positions at last-known prices.

        Uses the most recent price seen for each held ticker (set by the latest
        :meth:`mark` or fill), falling back to a position's cost basis if it has
        somehow never been priced. After the engine marks a day, this reflects
        that day's close.
        """
        held = sum(
            pos.quantity * self._last_price.get(t, pos.avg_cost)
            for t, pos in self.positions.items()
        )
        return self.cash + held

    def apply_fill(self, fill: Fill) -> None:
        """Update cash and positions for one executed fill.

        A buy spends cash and raises the position's commission-inclusive average
        cost; a sell returns cash, realises P&L on the shares sold (appending a
        :class:`~tradersjoy.core.types.Trade`), and reduces or closes the
        position. The fill's price is also recorded as the ticker's latest price.

        Args:
            fill: The executed fill to book. Assumed already validated by the
                broker (a buy it can fund, a sell it can cover).
        """
        self.fills.append(fill)
        self._last_price[fill.ticker] = fill.price

        if fill.side is Side.BUY:
            cost = fill.quantity * fill.price + fill.commission
            self.cash -= cost
            pos = self.positions.get(fill.ticker)
            if pos is None:
                self.positions[fill.ticker] = Position(
                    ticker=fill.ticker,
                    quantity=fill.quantity,
                    avg_cost=cost / fill.quantity,
                )
            else:
                total_qty = pos.quantity + fill.quantity
                pos.avg_cost = (pos.quantity * pos.avg_cost + cost) / total_qty
                pos.quantity = total_qty
            return

        # SELL
        self.cash += fill.quantity * fill.price - fill.commission
        pos = self.positions.get(fill.ticker)
        if pos is None:
            return  # nothing held; broker should have rejected this
        pnl = (fill.price - pos.avg_cost) * fill.quantity - fill.commission
        self.trades.append(
            Trade(
                ticker=fill.ticker,
                quantity=fill.quantity,
                entry_price=pos.avg_cost,
                exit_price=fill.price,
                exit_day=fill.day,
                pnl=pnl,
            )
        )
        pos.quantity -= fill.quantity
        if pos.quantity <= _DUST:
            del self.positions[fill.ticker]

    def mark(self, day: date, prices: dict[str, float]) -> None:
        """Record an equity-curve point for ``day`` using the given prices.

        Updates the last-known price for every ticker present in ``prices``
        (held tickers absent from it keep their prior price, e.g. on a halt),
        then appends ``(day, equity)`` to the curve.

        Args:
            day: The session being marked.
            prices: Latest prices by ticker, typically that day's closes.
        """
        self._last_price.update(prices)
        self.equity_curve.append((day, self.equity))
