"""Pure shaping helpers for the dashboard, kept here so they can be unit-tested.

Streamlit rendering lives in ``app.py`` and needs a browser to exercise; the
logic that turns journal entries into something plottable is ordinary data
massaging and belongs in functions a test can call directly.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tradersjoy.live.journal import RunEntry


def equity_curve(entries: list[RunEntry]) -> list[tuple[date, float]]:
    """Collapse runs into one equity point per session, ascending by day.

    A day can hold several runs (a few dry-run tinkers, then the real one); the
    curve should show one value per session, so the latest run for each
    ``decision_day`` wins. Sorting the input by wall-clock time first means that
    last write is the most recent decision for that day.

    Args:
        entries: Recorded runs in any order.

    Returns:
        ``(day, equity)`` pairs sorted ascending by day, one per distinct
        ``decision_day``.
    """
    by_day: dict[date, float] = {}
    for e in sorted(entries, key=lambda e: e.run_at):
        by_day[e.decision_day] = e.equity
    return sorted(by_day.items())
