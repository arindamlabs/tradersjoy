"""One live decision cycle: read the account, decide, optionally place orders.

A backtest is this same logic run thousands of times in a tight loop over
historical days. Live, it is run once per real day (manually in Phase 3, on a
schedule later), against a real broker and the real paper account. Keeping it to
a single ``run_once`` call makes the live path a thin, auditable wrapper around
the very same :class:`~tradersjoy.strategy.base.Strategy` the backtester drives.

By default a run is a *dry run*: it reads state and computes the orders a
strategy wants, but places nothing. Execution only happens when explicitly
requested, so seeing what the system intends to do is always free and safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import TYPE_CHECKING, Protocol

from tradersjoy.backtest.data import load_history
from tradersjoy.strategy.base import AccountView, BarContext

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tradersjoy.core.types import Order
    from tradersjoy.data.store import Store
    from tradersjoy.strategy.base import Strategy


class ExecutionBroker(Protocol):
    """The broker capabilities a live run needs: read the account, place orders.

    Both :class:`~tradersjoy.broker.alpaca.AlpacaBroker` and test fakes satisfy
    this structurally, so the trader is decoupled from any specific broker.
    """

    def get_account(self) -> AccountView:
        """Return the current account as a read-only snapshot."""
        ...

    def submit(self, orders: Sequence[Order]) -> list[str]:
        """Place the given orders; return a result line per order."""
        ...


@dataclass(frozen=True, slots=True)
class LivePlan:
    """The outcome of one :meth:`LiveTrader.run_once` call, for reporting.

    Attributes:
        day: The latest completed session the decision was based on. Orders act
            on the *next* market open.
        strategy_name: Name of the strategy that decided.
        equity: Account equity at decision time.
        cash: Account cash at decision time.
        starting_equity: The account's original balance, for a P/L reference.
        orders: The orders the strategy produced (before whole-share rounding).
        executed: Whether the orders were actually placed (vs. a dry run).
        results: Per-order outcome lines from the broker, populated only when
            ``executed`` is true.
    """

    day: date
    strategy_name: str
    equity: float
    cash: float
    starting_equity: float
    orders: list[Order]
    executed: bool
    results: list[str] = field(default_factory=list)

    @property
    def pnl(self) -> float:
        """Account profit or loss versus its starting balance."""
        return self.equity - self.starting_equity


class LiveTrader:
    """Runs a single strategy decision against a broker and the local store.

    Attributes:
        broker: The execution broker to read the account from and submit to.
        store: The market-data store recent bars were refreshed into.
        starting_equity: The account's opening balance, used to report P/L.
    """

    def __init__(
        self,
        broker: ExecutionBroker,
        store: Store,
        starting_equity: float = 100_000.0,
    ) -> None:
        """Wire the trader to a broker and a data store.

        Args:
            broker: Execution broker (e.g.
                :class:`~tradersjoy.broker.alpaca.AlpacaBroker`).
            store: Store holding the bars to decide from. Refresh it before
                calling :meth:`run_once` so the decision uses current data.
            starting_equity: Opening account balance for P/L reporting.
        """
        self.broker = broker
        self.store = store
        self.starting_equity = starting_equity

    def run_once(
        self,
        strategy: Strategy,
        tickers: Sequence[str],
        execute: bool = False,
    ) -> LivePlan:
        """Read the account, let ``strategy`` decide, and optionally place orders.

        The decision is made on the most recent session present in the store, so
        the caller must have refreshed bars beforehand. With ``execute=False``
        (the default) nothing is submitted; the returned plan still shows exactly
        what would have been placed.

        Args:
            strategy: The strategy to run for one day.
            tickers: Universe to load and decide over.
            execute: If true, submit the resulting orders to the broker.

        Returns:
            A :class:`LivePlan` describing the decision and any execution.

        Raises:
            ValueError: If the store holds no bars for the requested tickers.
        """
        account = self.broker.get_account()
        history = load_history(self.store, list(tickers))
        if not history.trading_days:
            raise ValueError(
                "No bars in the store for the requested tickers. Run an ingest "
                "or `trade` with --refresh first."
            )
        day = history.trading_days[-1]
        ctx = BarContext(
            day=day,
            bars=history.bars_on(day),
            history=history,
            portfolio=account,
        )
        orders = strategy.on_bar(ctx)

        results: list[str] = []
        if execute and orders:
            results = self.broker.submit(orders)

        return LivePlan(
            day=day,
            strategy_name=strategy.name,
            equity=account.equity,
            cash=account.cash,
            starting_equity=self.starting_equity,
            orders=orders,
            executed=execute and bool(orders),
            results=results,
        )
