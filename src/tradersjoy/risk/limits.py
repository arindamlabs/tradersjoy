"""The risk limits: a small, explicit bundle of the rails' numeric knobs.

Every rail is a single, named number with a sober default, so what the system
will and will not do is readable at a glance and tunable in one place. ``None``
on an optional rail switches that rail off cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Default market symbol whose drawdown drives the circuit breaker.
DEFAULT_BENCHMARK: str = "SPY"


@dataclass(frozen=True, slots=True)
class RiskLimits:
    """The numeric limits the risk layer enforces.

    Attributes:
        max_position_weight: Largest share of equity any one name may occupy.
            Buys are trimmed so a single position never exceeds this, capping the
            damage one blow-up can do. ``0.20`` means 20%.
        max_gross_exposure: Largest share of equity that may be invested at once,
            across all names. ``1.00`` means never exceed 100%, i.e. never use
            margin (the project's hard rule). Buys are trimmed to fit.
        stop_loss: Exit a position once its price falls this far below its cost
            basis. ``0.10`` means "down 10% from what we paid, get out". ``None``
            disables the stop.
        crash_drawdown: Block *new* buys (exits still allowed) while the benchmark
            sits at least this far below its recent high. ``0.15`` means "if the
            market is 15%+ off its high, stop adding risk". ``None`` disables it.
        crash_window: How many trading days back to look for the benchmark's high
            when measuring that drawdown.
        benchmark: Symbol whose drawdown the circuit breaker watches.
    """

    max_position_weight: float = 0.20
    max_gross_exposure: float = 1.00
    stop_loss: float | None = 0.10
    crash_drawdown: float | None = 0.15
    crash_window: int = 60
    benchmark: str = DEFAULT_BENCHMARK
