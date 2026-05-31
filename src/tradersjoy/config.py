"""Runtime configuration loaded from the environment.

All secrets and environment-specific values (Alpaca keys, the database URL) are
read from a local ``.env`` file via ``pydantic-settings``, never hard-coded.
This keeps credentials out of the repository and lets the same code run against
a paper account today and (eventually, deliberately) a live account later by
changing only the environment.

Typical use::

    from tradersjoy.config import get_settings

    settings = get_settings()
    client = TradingClient(settings.alpaca_api_key, settings.alpaca_api_secret)
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Absolute path to the repository root (three levels up from this file:
#: ``src/tradersjoy/config.py`` -> ``src/tradersjoy`` -> ``src`` -> root).
#: Used to locate ``.env`` and the default on-disk SQLite database regardless of
#: the current working directory the CLI is invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed application settings sourced from ``.env`` and the environment.

    Each field maps to an uppercase environment variable via its ``alias``.
    Values resolve in this order: real environment variables first, then the
    ``.env`` file, then the defaults declared here. Unknown environment
    variables are ignored (``extra="ignore"``) so unrelated shell variables do
    not cause validation errors.

    Attributes:
        alpaca_api_key: Alpaca API key ID (``ALPACA_API_KEY``). Empty by default
            so the app can import without credentials; commands that actually
            talk to Alpaca validate it is set.
        alpaca_api_secret: Alpaca secret key (``ALPACA_API_SECRET``).
        alpaca_base_url: Alpaca REST endpoint (``ALPACA_BASE_URL``). Defaults to
            the paper-trading host. This default is a safety guarantee: unless
            explicitly overridden, the system cannot place real-money orders.
        database_url: SQLAlchemy database URL (``DATABASE_URL``). Defaults to a
            SQLite file under ``<repo>/data/``.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_api_secret: str = Field(default="", alias="ALPACA_API_SECRET")
    alpaca_base_url: str = Field(
        default="https://paper-api.alpaca.markets", alias="ALPACA_BASE_URL"
    )
    database_url: str = Field(
        default=f"sqlite:///{PROJECT_ROOT / 'data' / 'tradersjoy.sqlite'}",
        alias="DATABASE_URL",
    )


def get_settings() -> Settings:
    """Build a fresh :class:`Settings` instance from the current environment.

    A function (rather than a module-level singleton) so tests can monkeypatch
    the environment or ``Settings.model_config`` and get a clean read, and so
    importing this module never triggers file/environment access as a side
    effect.

    Returns:
        A validated :class:`Settings` populated from ``.env`` and the
        environment.
    """
    return Settings()
