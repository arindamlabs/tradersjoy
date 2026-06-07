"""A local journal of live decision runs, for a track record the dashboard reads.

The trading system is otherwise stateless: ``trade`` reads the account, decides,
optionally places orders, prints, and forgets. Alpaca remembers the *account*
(positions, fills, equity), but nothing remembers what the *bot* did, including
dry-run days where it placed nothing but still formed an opinion. This journal
fills that gap: one row per :meth:`~tradersjoy.live.trader.LiveTrader.run_once`,
capturing the decision (the orders it wanted, executed or not) and the equity at
the time. That gives the dashboard an equity curve and a decision log, and it
survives an Alpaca paper-account reset because it lives in our own SQLite file.

Backed by SQLAlchemy on the same database file as the market-data
:class:`~tradersjoy.data.store.Store`, but with its own table, so the two
concerns stay logically separate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from tradersjoy.config import get_settings
from tradersjoy.core.types import Order, Side

if TYPE_CHECKING:
    from tradersjoy.live.trader import LivePlan


class JournalBase(DeclarativeBase):
    """Declarative base for the journal's tables, separate from the data store."""


class RunRow(JournalBase):
    """ORM row for one recorded live run.

    The orders the run produced are stored as a small JSON blob rather than a
    child table: they are only ever read back wholesale for display, never
    queried by ticker, so a separate table would add joins for no benefit.
    """

    __tablename__ = "live_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_at: Mapped[datetime] = mapped_column(DateTime)
    decision_day: Mapped[date] = mapped_column(Date)
    strategy: Mapped[str] = mapped_column(String)
    equity: Mapped[float] = mapped_column(Float)
    cash: Mapped[float] = mapped_column(Float)
    starting_equity: Mapped[float] = mapped_column(Float)
    executed: Mapped[bool] = mapped_column(Boolean)
    orders_json: Mapped[str] = mapped_column(String)


@dataclass(frozen=True, slots=True)
class RunEntry:
    """A recorded run read back from the journal, for reporting.

    Attributes:
        run_at: Wall-clock time the run happened (distinct from ``decision_day``,
            which is the market session decided on).
        decision_day: The latest completed session the decision used.
        strategy: Name of the strategy that decided (e.g. ``"risk(ml(top5))"``).
        equity: Account equity at run time.
        cash: Account cash at run time.
        starting_equity: Opening balance, for the P/L reference.
        executed: Whether the orders were actually placed (vs. a dry run).
        orders: The orders the strategy produced this run.
    """

    run_at: datetime
    decision_day: date
    strategy: str
    equity: float
    cash: float
    starting_equity: float
    executed: bool
    orders: list[Order]

    @property
    def pnl(self) -> float:
        """Account profit or loss versus the starting balance at this run."""
        return self.equity - self.starting_equity


def _encode_orders(orders: list[Order]) -> str:
    """Serialise orders to a compact JSON list for storage."""
    return json.dumps(
        [
            {"ticker": o.ticker, "side": str(o.side), "quantity": o.quantity, "tag": o.tag}
            for o in orders
        ]
    )


def _decode_orders(blob: str) -> list[Order]:
    """Rebuild orders from the JSON written by :func:`_encode_orders`."""
    return [
        Order(d["ticker"], Side(d["side"]), float(d["quantity"]), d.get("tag", ""))
        for d in json.loads(blob)
    ]


class Journal:
    """Append-only log of live runs, mirroring the :class:`Store` access pattern.

    Construct it (optionally with a custom database URL in tests), call
    :meth:`init_db` once, then :meth:`record` per run and :meth:`recent` to read
    the history back.

    Attributes:
        database_url: The SQLAlchemy URL this journal is bound to.
        engine: The SQLAlchemy engine backing all operations.
    """

    def __init__(self, database_url: str | None = None) -> None:
        """Bind the journal to a database, creating the SQLite directory if needed.

        Args:
            database_url: SQLAlchemy URL. Defaults to the configured
                ``DATABASE_URL`` (the same file the market-data store uses).
        """
        from tradersjoy.data.store import Store

        self.database_url = database_url or get_settings().database_url
        Store._ensure_sqlite_dir(self.database_url)
        self.engine = create_engine(self.database_url)

    def init_db(self) -> None:
        """Create the journal table if missing. Idempotent and safe to repeat."""
        JournalBase.metadata.create_all(self.engine)

    def record(
        self,
        *,
        run_at: datetime,
        decision_day: date,
        strategy: str,
        equity: float,
        cash: float,
        starting_equity: float,
        executed: bool,
        orders: list[Order],
    ) -> None:
        """Append one run to the journal.

        Args:
            run_at: Wall-clock time of the run.
            decision_day: The market session the decision was based on.
            strategy: Name of the deciding strategy.
            equity: Account equity at run time.
            cash: Account cash at run time.
            starting_equity: Opening balance for the P/L reference.
            executed: Whether orders were actually placed.
            orders: The orders the strategy produced.
        """
        row = RunRow(
            run_at=run_at,
            decision_day=decision_day,
            strategy=strategy,
            equity=equity,
            cash=cash,
            starting_equity=starting_equity,
            executed=executed,
            orders_json=_encode_orders(orders),
        )
        with Session(self.engine) as session:
            session.add(row)
            session.commit()

    def record_plan(self, plan: LivePlan, run_at: datetime) -> None:
        """Record a :class:`~tradersjoy.live.trader.LivePlan` as one run.

        Convenience wrapper over :meth:`record` so callers that already have a
        plan do not restate every field.

        Args:
            plan: The plan returned by ``run_once``.
            run_at: Wall-clock time of the run.
        """
        self.record(
            run_at=run_at,
            decision_day=plan.day,
            strategy=plan.strategy_name,
            equity=plan.equity,
            cash=plan.cash,
            starting_equity=plan.starting_equity,
            executed=plan.executed,
            orders=plan.orders,
        )

    def recent(self, limit: int = 200) -> list[RunEntry]:
        """Return the most recent runs, newest first.

        Args:
            limit: Maximum number of runs to return.

        Returns:
            Up to ``limit`` :class:`RunEntry` records, ordered newest first.
        """
        stmt = select(RunRow).order_by(RunRow.run_at.desc()).limit(limit)
        with Session(self.engine) as session:
            rows = list(session.scalars(stmt))
        return [
            RunEntry(
                run_at=r.run_at,
                decision_day=r.decision_day,
                strategy=r.strategy,
                equity=r.equity,
                cash=r.cash,
                starting_equity=r.starting_equity,
                executed=r.executed,
                orders=_decode_orders(r.orders_json),
            )
            for r in rows
        ]
