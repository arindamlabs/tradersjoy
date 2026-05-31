"""Command-line entry point for tradersjoy.

Exposes one Typer sub-command per phase of the system (``ingest``, ``backtest``,
``trade``, ``train``, ``dashboard``). Commands import their heavy dependencies
lazily inside the function body so that ``tradersjoy --help`` and unrelated
commands stay fast and do not pay for, say, importing yfinance.

The installed console script ``tradersjoy`` maps to :data:`app` (see the
``[project.scripts]`` table in ``pyproject.toml``).
"""

from datetime import date

import typer

#: The root Typer application all sub-commands attach to.
app = typer.Typer(
    name="tradersjoy",
    help="Automated paper-trading system.",
    no_args_is_help=True,
)


@app.command()
def ingest(
    start: str | None = typer.Option(
        None, help="Backfill start date YYYY-MM-DD. Defaults to universe.yaml."
    ),
    end: str | None = typer.Option(
        None, help="Backfill end date YYYY-MM-DD. Defaults to today."
    ),
    tickers: str | None = typer.Option(
        None, help="Comma-separated tickers to override the watchlist."
    ),
    source: str = typer.Option("yfinance", help="Data source: yfinance."),
) -> None:
    """Pull historical daily bars into the local store.

    Resolves the ticker list and start date (CLI options override
    ``config/universe.yaml``), fetches each ticker from the chosen data source,
    upserts the bars, and prints a per-ticker summary. Because ingest is
    idempotent, re-running it to extend or refresh data is always safe.

    Exits with code 2 for an unknown ``--source``, code 1 if any ticker failed
    (others still persist), and code 0 when all tickers succeed.
    """
    from tradersjoy.data.ingest import ingest as run_ingest
    from tradersjoy.data.ingest import load_universe
    from tradersjoy.data.store import Store

    if source == "yfinance":
        from tradersjoy.data.sources.yfinance_source import YFinanceSource

        ds = YFinanceSource()
    else:
        typer.echo(f"Unknown source: {source!r}. Supported: yfinance.")
        raise typer.Exit(code=2)

    universe_tickers, universe_start = load_universe()
    tick_list = (
        [t.strip().upper() for t in tickers.split(",") if t.strip()]
        if tickers
        else universe_tickers
    )
    start_date = date.fromisoformat(start) if start else universe_start
    end_date = date.fromisoformat(end) if end else None

    typer.echo(
        f"Ingesting {len(tick_list)} tickers from {source} "
        f"since {start_date.isoformat()}..."
    )
    store = Store()
    results = run_ingest(ds, store, tick_list, start_date, end_date)

    ok = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    for r in ok:
        span = (
            f"{r.start.isoformat()} -> {r.end.isoformat()}"
            if r.start and r.end
            else "no data"
        )
        typer.echo(f"  {r.ticker:6s} {r.rows:6d} bars  [{span}]")
    for r in failed:
        typer.echo(f"  {r.ticker:6s} FAILED: {r.error}")

    total = store.count()
    typer.echo(
        f"Done. {len(ok)} ok, {len(failed)} failed. Total rows in store: {total}."
    )
    if failed:
        raise typer.Exit(code=1)


@app.command()
def backtest() -> None:
    """Run a strategy against historical bars."""
    typer.echo("backtest: not implemented yet (Phase 2)")
    raise typer.Exit(code=1)


@app.command()
def trade() -> None:
    """Run the live paper-trading loop against the Alpaca paper broker."""
    typer.echo("trade: not implemented yet (Phase 3)")
    raise typer.Exit(code=1)


@app.command()
def train() -> None:
    """Train an ML strategy using walk-forward validation."""
    typer.echo("train: not implemented yet (Phase 4)")
    raise typer.Exit(code=1)


@app.command()
def dashboard() -> None:
    """Launch the Streamlit dashboard."""
    typer.echo("dashboard: not implemented yet (Phase 6)")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
