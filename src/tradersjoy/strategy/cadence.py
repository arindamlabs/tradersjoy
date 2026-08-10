"""When a strategy is allowed to rebalance, derived purely from the calendar.

Measured over 2016-2026, re-ranking the universe every session churns 1.4 of 5
positions a day: about 70x annual turnover, or a 6.7%/year drag at 5 bps a side.
Half of all positions last exactly one session and two thirds of sells are
re-bought within five days, while the model's label predicts *five* sessions
ahead. The strategy was consistently selling before the thing it predicted had
time to happen. Rebalancing on the label's own horizon instead cuts the drag to
about 1.9%/year.

**Why this is calendar-derived rather than counted.** The obvious implementation
of "every 5 days" remembers when it last rebalanced. That works perfectly in a
backtest and breaks live, because
:meth:`~tradersjoy.live.trader.LiveTrader.run_once` starts fresh each day with no
memory of previous runs, so the counter would reset to zero every session and
rebalance every day regardless. It is the same trap the risk layer avoided by
staying stateless (see :mod:`tradersjoy.risk.manager`), and the same failure
mode: silently correct in backtest, silently wrong live.

So a rebalance day is defined as **the first trading session of a calendar
period**. That is a pure function of the date plus the sessions around it, needs
no memory, and gives backtest and live structurally identical answers. It also
handles holidays for free: if the market is shut on Monday, the week's first
session is simply Tuesday.
"""

from __future__ import annotations

import bisect
from datetime import date

#: Cadences accepted by the CLI and the strategy registry.
CADENCES = ("daily", "weekly", "fortnightly", "monthly")

#: Roughly how many trading sessions each cadence spans. Used for help text and
#: for lining a cadence up with a model's label horizon.
SESSIONS_PER_PERIOD = {"daily": 1, "weekly": 5, "fortnightly": 10, "monthly": 21}


def period_key(day: date, cadence: str) -> object:
    """Return an opaque key identifying which period ``day`` falls in.

    Two dates share a key exactly when they belong to the same rebalance period.
    The key's type is deliberately unspecified; only equality is meaningful.

    Args:
        day: The date to bucket.
        cadence: One of :data:`CADENCES`.

    Returns:
        A hashable key that compares equal for dates in the same period.

    Raises:
        ValueError: If ``cadence`` is not recognised.
    """
    if cadence == "daily":
        return day.toordinal()
    if cadence == "weekly":
        iso = day.isocalendar()
        return (iso[0], iso[1])
    if cadence == "fortnightly":
        # Bucketed on the day ordinal rather than ISO-week parity. Week numbers
        # restart each year, so parity would make weeks 52 and 1 adjacent and
        # occasionally emit a one-week "fortnight" at the year boundary.
        return (day.toordinal() - 1) // 14
    if cadence == "monthly":
        return (day.year, day.month)
    raise ValueError(
        f"Unknown rebalance cadence {cadence!r}. Choose from: {', '.join(CADENCES)}."
    )


def is_rebalance_day(day: date, trading_days: list[date], cadence: str) -> bool:
    """Whether ``day`` is the first trading session of its period.

    Args:
        day: The session being decided on.
        trading_days: All known sessions, ascending. Only the one immediately
            before ``day`` is consulted, so the caller may pass a window rather
            than all of history; it must merely reach back one session.
        cadence: One of :data:`CADENCES`.

    Returns:
        True if a rebalance is due. Always true for ``"daily"``, and true for
        the earliest known session, which has no predecessor to compare against
        and must be allowed to establish the initial positions.
    """
    if cadence == "daily":
        return True
    index = bisect.bisect_left(trading_days, day)
    if index == 0:
        return True
    return period_key(trading_days[index - 1], cadence) != period_key(day, cadence)
