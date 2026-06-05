"""Risk management: the rails that sit between a strategy and the broker.

A strategy decides *what* it wants to own; this package decides *how much* and
*when to refuse*. It is deliberately a thin, stateless layer (see
:class:`~tradersjoy.risk.manager.RiskManagedStrategy`) so the very same rails
apply identically in a backtest and in live paper trading.
"""

from __future__ import annotations

from tradersjoy.risk.limits import RiskLimits
from tradersjoy.risk.manager import RiskManagedStrategy

__all__ = ["RiskLimits", "RiskManagedStrategy"]
