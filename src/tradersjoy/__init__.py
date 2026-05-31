"""tradersjoy: an automated paper-trading system.

A learning-grade quant stack for daily-swing strategies on US equities, executed
against the Alpaca paper-trading API. Sub-packages:

- :mod:`tradersjoy.core`     - storage-agnostic domain types (e.g. ``Bar``).
- :mod:`tradersjoy.data`     - market-data sources, ingest, and the local store.
- :mod:`tradersjoy.broker`   - order execution (simulated and live paper).
- :mod:`tradersjoy.strategy` - signal-generating strategies.
- :mod:`tradersjoy.ml`       - feature engineering, training, walk-forward eval.
- :mod:`tradersjoy.risk`     - position sizing and risk limits.

The top-level :mod:`tradersjoy.config` and :mod:`tradersjoy.cli` provide settings
and the command-line entry point.
"""

#: Declares the docstring convention used throughout the package (PEP 258).
#: A documentation marker for humans and tooling. Note: pdoc does not read this,
#: so the docs commands pass ``-d google`` explicitly (see README).
__docformat__ = "google"

__version__ = "0.1.0"
