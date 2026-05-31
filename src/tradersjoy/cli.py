import typer

app = typer.Typer(
    name="tradersjoy",
    help="Automated paper-trading system. Phase 0 scaffold: subcommands are stubs.",
    no_args_is_help=True,
)


@app.command()
def ingest() -> None:
    """Pull historical + recent bars into the local store."""
    typer.echo("ingest: not implemented yet (Phase 1)")
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
