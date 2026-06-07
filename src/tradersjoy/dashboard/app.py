"""Read-only Streamlit dashboard for the tradersjoy paper account.

Launch with ``tradersjoy dashboard``. It pulls the live snapshot (equity, cash,
positions, pending orders) straight from Alpaca and the decision history from the
local run journal, then shows an equity curve and a decision log. It is
deliberately read-only: it never places or cancels an order, so opening it to
watch is always safe.

This module is a thin rendering layer. The shaping logic it depends on lives in
:mod:`tradersjoy.dashboard.data` so it can be tested without a browser.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from tradersjoy.dashboard.data import equity_curve
from tradersjoy.live.journal import Journal

# Matches LiveTrader's default opening balance; the P/L reference for the header.
STARTING_EQUITY = 100_000.0

st.set_page_config(page_title="tradersjoy", layout="wide")
st.title("tradersjoy - paper account")
st.caption("Read-only view. Nothing here places or cancels an order.")


@st.cache_data(ttl=60)
def _load_account() -> dict:
    """Fetch the live account snapshot from Alpaca (cached for 60s)."""
    from tradersjoy.broker.alpaca import AlpacaBroker

    broker = AlpacaBroker()
    acct = broker.get_account()
    return {
        "equity": acct.equity,
        "cash": acct.cash,
        "positions": broker.positions_detail(),
        "open_orders": broker.open_orders(),
    }


def _render_account() -> None:
    """Render the live account header, positions, and pending orders."""
    try:
        acct = _load_account()
    except Exception as exc:  # noqa: BLE001 - surface any broker/network error in the UI
        st.error(f"Could not read the Alpaca paper account: {exc}")
        st.caption("Check your keys in .env, then use the menu (top-right) to rerun.")
        return

    pnl = acct["equity"] - STARTING_EQUITY
    invested = acct["equity"] - acct["cash"]
    c1, c2, c3 = st.columns(3)
    c1.metric("Equity", f"${acct['equity']:,.2f}", f"{pnl:+,.2f} vs start")
    c2.metric("Cash", f"${acct['cash']:,.2f}")
    c3.metric("Invested", f"${invested:,.2f}")

    st.subheader("Positions")
    positions = acct["positions"]
    if positions:
        df = pd.DataFrame(positions)
        df["unrealized_plpc"] = df["unrealized_plpc"] * 100.0
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "qty": st.column_config.NumberColumn("Shares", format="%.0f"),
                "avg_cost": st.column_config.NumberColumn("Avg cost", format="$%.2f"),
                "price": st.column_config.NumberColumn("Price", format="$%.2f"),
                "market_value": st.column_config.NumberColumn("Value", format="$%.2f"),
                "unrealized_pl": st.column_config.NumberColumn("Unreal P/L", format="$%.2f"),
                "unrealized_plpc": st.column_config.NumberColumn("Unreal %", format="%.2f%%"),
            },
        )
    else:
        st.caption("No open positions yet.")

    st.subheader("Pending orders")
    open_orders = acct["open_orders"]
    if open_orders:
        st.dataframe(pd.DataFrame(open_orders), hide_index=True, use_container_width=True)
        st.caption("These queued orders fill at the next market open.")
    else:
        st.caption("No pending orders.")


def _render_journal() -> None:
    """Render the equity curve and decision log from the local run journal."""
    journal = Journal()
    journal.init_db()
    entries = journal.recent(limit=500)

    st.subheader("Equity over time")
    curve = equity_curve(entries)
    if curve:
        cdf = pd.DataFrame(curve, columns=["day", "equity"]).set_index("day")
        st.line_chart(cdf)
        if len(curve) < 2:
            st.caption(
                "Only one session logged so far; the line fills in as the bot "
                "runs each day."
            )
    else:
        st.caption("No runs logged yet. Run `tradersjoy trade` to start the journal.")

    st.subheader("Decision log")
    if entries:
        rows = [
            {
                "run_at": e.run_at,
                "day": e.decision_day,
                "strategy": e.strategy,
                "equity": e.equity,
                "pnl": e.pnl,
                "executed": e.executed,
                "orders": ", ".join(
                    f"{o.side} {o.quantity:.0f} {o.ticker}" for o in e.orders
                )
                or "(none)",
            }
            for e in entries
        ]
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            use_container_width=True,
            column_config={
                "equity": st.column_config.NumberColumn("Equity", format="$%.2f"),
                "pnl": st.column_config.NumberColumn("P/L", format="$%.2f"),
            },
        )
    else:
        st.caption("No runs logged yet.")


_render_account()
st.divider()
_render_journal()
