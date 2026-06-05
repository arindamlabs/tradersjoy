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
def backtest(
    strategy: str = typer.Option(
        "buyhold", help="Strategy to run: buyhold or sma."
    ),
    tickers: str | None = typer.Option(
        None, help="Comma-separated tickers. Defaults to the universe watchlist."
    ),
    start: str | None = typer.Option(
        None, "--from", help="Backtest start date YYYY-MM-DD. Defaults to earliest."
    ),
    end: str | None = typer.Option(
        None, "--to", help="Backtest end date YYYY-MM-DD. Defaults to latest."
    ),
    cash: float = typer.Option(100_000.0, help="Starting cash balance."),
    slippage_bps: float = typer.Option(
        5.0, help="Adverse slippage per fill, in basis points."
    ),
    short_window: int = typer.Option(20, help="Fast SMA window (sma only)."),
    long_window: int = typer.Option(50, help="Slow SMA window (sma only)."),
    model: str | None = typer.Option(
        None, "--model", help="Path to a trained model (ml strategy only)."
    ),
    top_k: int = typer.Option(5, help="Names the ml strategy holds at once (ml only)."),
    risk: bool = typer.Option(
        False,
        "--risk/--no-risk",
        help="Wrap the strategy in the risk layer (position sizing, exposure cap, "
        "stop-loss, circuit breaker).",
    ),
) -> None:
    """Run a strategy against stored historical bars and print its scorecard.

    Loads the requested tickers and window from the local store, replays them
    through the chosen strategy with a simulated broker (orders fill at the next
    session's open, with slippage), and reports total return, CAGR, Sharpe, max
    drawdown, and hit rate.

    Exits with code 2 for an unknown strategy or empty universe, and code 0 on a
    completed run.
    """
    from tradersjoy.backtest.data import load_history
    from tradersjoy.backtest.engine import run_backtest
    from tradersjoy.broker.sim import SimBroker
    from tradersjoy.data.ingest import load_universe
    from tradersjoy.data.store import Store
    from tradersjoy.strategy.registry import build_strategy

    if tickers:
        tick_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        tick_list, _ = load_universe()
    if not tick_list:
        typer.echo("No tickers to backtest.")
        raise typer.Exit(code=2)

    try:
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
        typer.echo(str(exc))
        raise typer.Exit(code=2) from exc

    if strategy.strip().lower() == "ml":
        typer.echo(
            "Note: backtesting 'ml' over its own training window is IN-SAMPLE and "
            "flatters the model. The honest score is the walk-forward from `train`."
        )

    start_date = date.fromisoformat(start) if start else None
    end_date = date.fromisoformat(end) if end else None

    store = Store()
    data = load_history(store, tick_list, start_date, end_date)
    if not data.trading_days:
        typer.echo(
            "No bars found for the requested tickers/window. Run `ingest` first?"
        )
        raise typer.Exit(code=2)

    broker = SimBroker(slippage_bps=slippage_bps)
    result = run_backtest(strat, data, broker, cash, start_date, end_date)
    typer.echo(result.summary())
    if broker.rejections:
        typer.echo(f"\nNote: {len(broker.rejections)} order(s) rejected (unfunded/uncovered).")


@app.command()
def trade(
    strategy: str = typer.Option("buyhold", help="Strategy to run: buyhold or sma."),
    tickers: str | None = typer.Option(
        None, help="Comma-separated tickers. Defaults to the universe watchlist."
    ),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Actually place orders on the paper account. Omit for a safe dry run.",
    ),
    refresh: bool = typer.Option(
        True, help="Refresh recent bars from yfinance before deciding."
    ),
    lookback_days: int = typer.Option(
        400, help="Days of recent data to refresh before deciding."
    ),
    short_window: int = typer.Option(20, help="Fast SMA window (sma only)."),
    long_window: int = typer.Option(50, help="Slow SMA window (sma only)."),
    model: str | None = typer.Option(
        None, "--model", help="Path to a trained model (ml strategy only)."
    ),
    top_k: int = typer.Option(5, help="Names the ml strategy holds at once (ml only)."),
    risk: bool = typer.Option(
        False,
        "--risk/--no-risk",
        help="Wrap the strategy in the risk layer (position sizing, exposure cap, "
        "stop-loss, circuit breaker).",
    ),
) -> None:
    """Run one live decision against the Alpaca paper account.

    Refreshes recent bars, reads the paper account, lets the strategy decide on
    the latest close, and prints the orders. By default this is a DRY RUN that
    places nothing; pass ``--execute`` to actually submit the orders (they queue
    for the next market open). Run it once per day, ideally after the close.

    Exits with code 2 for an unknown strategy, empty universe, or missing data.
    """
    from datetime import timedelta

    from tradersjoy.broker.alpaca import AlpacaBroker, plan_whole_share_orders
    from tradersjoy.data.ingest import load_universe
    from tradersjoy.data.store import Store
    from tradersjoy.live.trader import LiveTrader
    from tradersjoy.strategy.registry import build_strategy

    if tickers:
        tick_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        tick_list, _ = load_universe()
    if not tick_list:
        typer.echo("No tickers to trade.")
        raise typer.Exit(code=2)

    try:
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
        typer.echo(str(exc))
        raise typer.Exit(code=2) from exc

    if refresh:
        from tradersjoy.data.ingest import ingest as run_ingest
        from tradersjoy.data.sources.yfinance_source import YFinanceSource

        refresh_start = date.today() - timedelta(days=lookback_days)
        typer.echo(f"Refreshing bars since {refresh_start.isoformat()}...")
        results = run_ingest(YFinanceSource(), Store(), tick_list, refresh_start, None)
        n_ok = sum(1 for r in results if r.error is None)
        n_failed = len(results) - n_ok
        msg = f"  refreshed {n_ok}/{len(tick_list)} tickers"
        if n_failed:
            msg += f", {n_failed} failed (using whatever is already stored)"
        typer.echo(msg)

    broker = AlpacaBroker()
    trader = LiveTrader(broker, Store())

    mode = "LIVE (paper) EXECUTION" if execute else "DRY RUN (no orders placed)"
    typer.echo(f"\n=== {mode} ===")
    try:
        plan = trader.run_once(strat, tick_list, execute=execute)
    except ValueError as exc:
        typer.echo(str(exc))
        raise typer.Exit(code=2) from exc

    typer.echo(f"Decision date:  {plan.day.isoformat()}  (orders act on next open)")
    typer.echo(f"Account equity: ${plan.equity:,.2f}")
    typer.echo(f"Cash:           ${plan.cash:,.2f}")
    typer.echo(f"P/L vs ${plan.starting_equity:,.0f}: ${plan.pnl:+,.2f}")
    typer.echo(f"Strategy:       {plan.strategy_name}")

    if not plan.orders:
        typer.echo("\nNo orders today; nothing to do.")
        return

    if plan.executed:
        typer.echo(f"\nSubmitted {len(plan.orders)} order intent(s):")
        for line in plan.results:
            typer.echo(f"  {line}")
    else:
        typer.echo(f"\nWould place {len(plan.orders)} order(s) (whole shares):")
        for op in plan_whole_share_orders(plan.orders):
            note = f"  ({op.note})" if op.note else ""
            typer.echo(
                f"  {op.side:4s} {op.shares:>6d}  {op.ticker:6s}"
                f"  [requested {op.requested_qty:.4f}]{note}"
            )
        typer.echo(
            "\nThis was a DRY RUN. Re-run with --execute to place these orders."
        )


@app.command()
def train(
    tickers: str | None = typer.Option(
        None, help="Comma-separated tickers. Defaults to the universe watchlist."
    ),
    horizon: int = typer.Option(
        5, help="Forward look in trading days the label predicts over."
    ),
    threshold: float = typer.Option(
        0.0,
        help="Return cut: absolute mode, the bar to clear; relative mode, the "
        "excess over the day's median to clear.",
    ),
    relative: bool = typer.Option(
        True,
        "--relative/--absolute",
        help="Label by beating the universe median (cross-sectional) vs plain "
        "up/down. Relative aligns the target with what top-K actually needs.",
    ),
    train_years: int = typer.Option(
        5, help="Initial years of history before the first walk-forward test year."
    ),
    start: str | None = typer.Option(
        None, "--from", help="Earliest day to use YYYY-MM-DD. Defaults to earliest."
    ),
    end: str | None = typer.Option(
        None, "--to", help="Latest day to use YYYY-MM-DD. Defaults to latest."
    ),
    model_out: str = typer.Option(
        "data/models/ml.joblib", help="Where to save the final deployable model."
    ),
) -> None:
    """Train an ML strategy and score it honestly with walk-forward validation.

    Builds a learning table (features + a forward up/down label) from the stored
    bars, runs a year-by-year walk-forward (train on the past, test on the next
    unseen year) to produce an out-of-sample scorecard, then retrains one final
    model on *all* labelled history and saves it for live use.

    The walk-forward numbers are the honest track record; the saved model is the
    one you would actually deploy. Exits with code 2 on an empty universe or too
    little data to evaluate.
    """
    from tradersjoy.backtest.data import load_history
    from tradersjoy.data.ingest import load_universe
    from tradersjoy.data.store import Store
    from tradersjoy.ml.dataset import build_dataset, labelled, matrix
    from tradersjoy.ml.model import GBMModel
    from tradersjoy.ml.walkforward import walk_forward

    if tickers:
        tick_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    else:
        tick_list, _ = load_universe()
    if not tick_list:
        typer.echo("No tickers to train on.")
        raise typer.Exit(code=2)

    start_date = date.fromisoformat(start) if start else None
    end_date = date.fromisoformat(end) if end else None

    store = Store()
    data = load_history(store, tick_list, start_date, end_date)
    if not data.trading_days:
        typer.echo("No bars found for the requested tickers/window. Run `ingest` first?")
        raise typer.Exit(code=2)

    target = (
        f"beat universe median over next {horizon} day(s)"
        if relative
        else f"up/down over next {horizon} day(s)"
    )
    typer.echo(
        f"Building dataset: {len(tick_list)} tickers, "
        f"label = {target} (threshold {threshold:+.2%})..."
    )
    samples = build_dataset(
        data, tick_list, horizon=horizon, threshold=threshold, relative=relative
    )
    labelled_samples = labelled(samples)
    if len(labelled_samples) < 500:
        typer.echo(
            f"Only {len(labelled_samples)} labelled rows; too few to evaluate "
            "honestly. Ingest more history or widen the ticker list."
        )
        raise typer.Exit(code=2)
    typer.echo(
        f"  {len(labelled_samples):,} labelled rows, "
        f"{len(samples) - len(labelled_samples):,} most-recent rows held back "
        "(no known future yet)."
    )

    typer.echo("\nRunning walk-forward validation (train past -> test next year)...")
    result = walk_forward(labelled_samples, train_years=train_years)
    typer.echo("")
    typer.echo(result.summary())

    if result.overall is not None and result.overall.auc < 0.52:
        typer.echo(
            "\nReading this honestly: an AUC this close to 0.50 means the model "
            "has little or no real ranking edge yet. That is the expected, sober "
            "first result, not a failure of the plumbing."
        )

    typer.echo("\nRetraining a final model on all labelled history...")
    X, y = matrix(labelled_samples)
    final = GBMModel().fit(X, y)
    saved = final.save(model_out)
    typer.echo(f"Saved deployable model -> {saved}")
    typer.echo(
        "\nUse it:\n"
        f"  tradersjoy backtest --strategy ml --model {saved}   # in-sample; see caveat\n"
        f"  tradersjoy trade    --strategy ml --model {saved}   # live dry run"
    )


@app.command()
def dashboard() -> None:
    """Launch the Streamlit dashboard."""
    typer.echo("dashboard: not implemented yet (Phase 6)")
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
