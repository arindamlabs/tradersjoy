from typer.testing import CliRunner

import tradersjoy
from tradersjoy.cli import app


def test_version_string() -> None:
    assert tradersjoy.__version__ == "0.1.0"


def test_cli_help_runs() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "tradersjoy" in result.stdout
    for cmd in ("ingest", "backtest", "trade", "train", "dashboard"):
        assert cmd in result.stdout


def test_settings_loads_with_defaults(monkeypatch) -> None:
    for var in ("ALPACA_API_KEY", "ALPACA_API_SECRET", "ALPACA_BASE_URL", "DATABASE_URL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(
        "tradersjoy.config.Settings.model_config",
        {"env_file": None, "extra": "ignore"},
    )
    from tradersjoy.config import Settings

    s = Settings()
    assert s.alpaca_base_url == "https://paper-api.alpaca.markets"
