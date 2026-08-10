"""Unattended daily operation: the guarded wrapper that lets the bot run itself.

``trade`` is the *manual* live path: a human decides to run it, reads the
printout, and judges whether the result looks sane. Automation removes that
human, and with them every implicit check they were performing. This module puts
those checks back explicitly, because the failure modes of an unwatched trader
are quiet ones:

- **It stops running and nobody notices.** Every firing writes a heartbeat row,
  so "the scheduler died in June" is a visible gap rather than an assumption.
- **It runs on data that never arrived.** If yfinance is down or lagging, the
  store still holds *yesterday's* bars; deciding on them would silently re-trade
  a stale opinion. The freshness guard refuses instead.
- **It decides mid-session.** Today's bar is not final until the close, so a run
  that fires while the market is open would rank stocks on a half-formed day.
- **It fires twice for the same session.** A systemd retry or a manual re-run
  would otherwise re-decide an already-traded day and churn the book.
- **It cannot be stopped without a keyboard.** A halt file pauses trading
  without editing timers or unsetting credentials.

Each guard is a reason to *not* trade. That asymmetry is deliberate: the cost of
skipping a day is one missed rotation, and the cost of trading on bad state is
real money in the real version of this system.
"""

from __future__ import annotations

import fcntl
import logging
import logging.handlers
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from tradersjoy.config import PROJECT_ROOT

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from tradersjoy.live.trader import LivePlan

#: Touch this file to pause unattended trading; delete it to resume. Lives under
#: the gitignored data directory so it is a purely local operational switch.
DEFAULT_HALT_FILE = PROJECT_ROOT / "data" / "HALT"

#: Held for the duration of a run so two schedulers cannot trade at once.
DEFAULT_LOCK_FILE = PROJECT_ROOT / "data" / "autorun.lock"

#: Rotating log directory, so an unattended run leaves a trail even when nobody
#: is watching stdout.
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

log = logging.getLogger("tradersjoy.autorun")


@dataclass(frozen=True, slots=True)
class AutoRunResult:
    """What one unattended firing did, and what the process should exit with.

    Attributes:
        status: ``traded``, ``no_orders``, ``skipped``, or ``failed``.
        reason: Human-readable explanation, most useful for skips and failures.
        plan: The decision, when the run got far enough to make one.
        session: The session the run targeted, when it resolved one.
    """

    status: str
    reason: str
    plan: LivePlan | None = None
    session: date | None = None

    @property
    def exit_code(self) -> int:
        """Process exit status: non-zero only when something actually went wrong.

        A skip is a success. Weekends, holidays, and a deliberate halt are the
        system working as designed, and making systemd shout about them would
        train the operator to ignore it, which is exactly how a real failure
        gets missed.
        """
        return 1 if self.status == "failed" else 0


def configure_logging(
    log_dir: Path | None = None, level: int = logging.INFO
) -> None:
    """Send autorun logs to both a rotating file and stdout.

    Two sinks because they serve different readers: stdout is captured by
    systemd into the journal (``journalctl --user -u tradersjoy``), while the
    file survives independently and can be tailed directly.

    Args:
        log_dir: Directory for ``autorun.log``. Defaults to ``<repo>/logs``.
        level: Logging threshold.
    """
    directory = log_dir or DEFAULT_LOG_DIR
    directory.mkdir(parents=True, exist_ok=True)

    log.setLevel(level)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")

    rotating = logging.handlers.RotatingFileHandler(
        directory / "autorun.log", maxBytes=1_000_000, backupCount=5
    )
    rotating.setFormatter(fmt)
    log.addHandler(rotating)

    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    log.addHandler(stream)


@contextmanager
def _exclusive_lock(lock_file: Path) -> Iterator[bool]:
    """Yield whether an exclusive, non-blocking lock on ``lock_file`` was taken.

    Non-blocking on purpose: if a previous run is somehow still going, the right
    move is to report and leave, not to queue up a second trader behind it.
    """
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_file.open("w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        yield True
    finally:
        handle.close()


def _already_ran_for(journal, session: date) -> bool:  # noqa: ANN001 - Journal
    """Whether an unattended run already traded the given session.

    Guards against double-firing (a systemd restart, an operator re-run). Only
    *executed* runs count: a dry run made no commitment, so repeating it is
    harmless and still useful.
    """
    return any(
        r.decision_day == session and r.executed and r.status == "traded"
        for r in journal.recent_auto_runs(limit=25)
    )


def run_daily(
    *,
    strategy: str = "ml",
    tickers: Sequence[str] | None = None,
    execute: bool = False,
    model: str | None = None,
    top_k: int = 5,
    risk: bool = True,
    short_window: int = 20,
    long_window: int = 50,
    lookback_days: int = 400,
    force: bool = False,
    halt_file: Path | None = None,
    lock_file: Path | None = None,
    broker=None,  # noqa: ANN001 - duck-typed for tests
    store=None,  # noqa: ANN001
    journal=None,  # noqa: ANN001
    now: datetime | None = None,
) -> AutoRunResult:
    """Run one guarded, unattended decision cycle and record what happened.

    Walks the guards in order (lock, halt switch, session resolution, duplicate
    check, data freshness) and only then lets the strategy decide. Every exit
    path, including every refusal, writes a heartbeat row.

    Args:
        strategy: Strategy name to build (e.g. ``"ml"``, ``"buyhold"``).
        tickers: Universe override; defaults to the configured watchlist.
        execute: Place orders for real. False (the default) still runs every
            guard and records the decision, placing nothing.
        model: Path to a trained model, for the ``ml`` strategy.
        top_k: Names the ``ml`` strategy holds at once.
        risk: Wrap the strategy in the risk layer.
        short_window: Fast SMA window (``sma`` only).
        long_window: Slow SMA window (``sma`` only).
        lookback_days: Days of recent bars to refresh before deciding.
        force: Bypass the duplicate-session guard.
        halt_file: Override the halt switch path.
        lock_file: Override the lock file path.
        broker: Injected broker (tests); defaults to a real ``AlpacaBroker``.
        store: Injected store (tests); defaults to the configured ``Store``.
        journal: Injected journal (tests); defaults to the configured one.
        now: Instant to evaluate the session against (tests).

    Returns:
        An :class:`AutoRunResult` describing the outcome.
    """
    from tradersjoy.data.ingest import load_universe
    from tradersjoy.data.store import Store
    from tradersjoy.live.journal import Journal

    ran_at = now or datetime.now()
    journal = journal or Journal()
    journal.init_db()

    def finish(
        status: str,
        reason: str,
        plan: LivePlan | None = None,
        session: date | None = None,
    ) -> AutoRunResult:
        """Record the heartbeat and build the result for any exit path."""
        journal.record_auto_run(
            ran_at=ran_at,
            status=status,
            reason=reason,
            decision_day=session,
            executed=bool(plan and plan.executed),
            n_orders=len(plan.orders) if plan else 0,
        )
        level = logging.ERROR if status == "failed" else logging.INFO
        log.log(level, "%s: %s", status, reason)
        return AutoRunResult(status=status, reason=reason, plan=plan, session=session)

    lock_path = lock_file or DEFAULT_LOCK_FILE
    with _exclusive_lock(lock_path) as acquired:
        if not acquired:
            return finish("failed", f"another run holds {lock_path.name}")

        halt = halt_file or DEFAULT_HALT_FILE
        if halt.exists():
            return finish("skipped", f"halt file present at {halt}")

        if broker is None:
            from tradersjoy.broker.alpaca import AlpacaBroker

            broker = AlpacaBroker()
        store = store or Store()

        if broker.is_market_open():
            return finish("skipped", "market is still open; today's bar is not final")

        session = broker.last_completed_session(now)
        if session is None:
            return finish("failed", "could not resolve the last completed session")
        log.info("target session: %s", session.isoformat())

        if not force and _already_ran_for(journal, session):
            return finish(
                "skipped", f"already traded session {session.isoformat()}", session=session
            )

        tick_list = list(tickers) if tickers else load_universe()[0]
        if not tick_list:
            return finish("failed", "empty universe", session=session)

        try:
            from tradersjoy.strategy.registry import build_strategy

            strat = build_strategy(
                strategy,
                tick_list,
                short_window=short_window,
                long_window=long_window,
                model_path=model,
                top_k=top_k,
                risk=risk,
            )
        except ValueError as exc:
            return finish("failed", f"could not build strategy: {exc}", session=session)

        # Refresh first, then insist the refresh actually reached the session we
        # intend to trade. A partial or failed refresh must not silently fall
        # through to yesterday's data.
        try:
            _refresh(store, tick_list, lookback_days)
        except Exception as exc:  # noqa: BLE001 - a broken feed must not trade
            return finish("failed", f"data refresh raised: {exc}", session=session)

        latest = _latest_stored_day(store, tick_list)
        if latest != session:
            return finish(
                "failed",
                f"stale data: store reaches {latest}, need {session.isoformat()}",
                session=session,
            )

        try:
            from tradersjoy.live.trader import LiveTrader

            plan = LiveTrader(broker, store).run_once(strat, tick_list, execute=execute)
        except Exception as exc:  # noqa: BLE001 - report, never crash the timer silently
            return finish("failed", f"decision raised: {exc}", session=session)

        journal.record_plan(plan, run_at=ran_at)
        for line in plan.results:
            log.info("  %s", line)

        if not plan.orders:
            return finish("no_orders", "strategy wanted no trades", plan, session)

        if plan.executed:
            return finish("traded", f"placed {len(plan.orders)} order(s)", plan, session)
        return finish(
            "dry_run",
            f"would have placed {len(plan.orders)} order(s); --execute was off",
            plan,
            session,
        )


def _refresh(store, tickers: Sequence[str], lookback_days: int) -> None:  # noqa: ANN001
    """Pull recent bars into the store so the decision uses current data."""
    from tradersjoy.data.ingest import ingest as run_ingest
    from tradersjoy.data.sources.yfinance_source import YFinanceSource

    start = date.today() - timedelta(days=lookback_days)
    results = run_ingest(YFinanceSource(), store, list(tickers), start, None)
    failed = [r.ticker for r in results if r.error is not None]
    log.info("refreshed %d/%d tickers", len(results) - len(failed), len(results))
    if failed:
        log.warning("refresh failed for: %s", ", ".join(failed))


def _latest_stored_day(store, tickers: Sequence[str]) -> date | None:  # noqa: ANN001
    """Return the most recent session present in the store, or ``None``."""
    from tradersjoy.backtest.data import load_history

    history = load_history(store, list(tickers))
    return history.trading_days[-1] if history.trading_days else None
