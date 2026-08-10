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


#: Named bundles of limits, so a deployment can name its risk posture rather
#: than restate five numbers.
#:
#: The distinction is not cosmetic. Measured out-of-sample (2010-2026) on the
#: ml/weekly strategy, the two *reactive* rails cost 6.4 points of CAGR and 0.14
#: of Sharpe while making max drawdown 4.8 points **worse**, and they hurt in 12
#: of 17 years including 2022, the bear market they exist for. The mechanism is
#: understood: a strategy that already re-ranks and exits weak names every week
#: does not need a second exit rule, and a cost-basis stop mostly sells transient
#: weakness just before the rebound, while the circuit breaker blocks re-entry
#: during exactly the stretches with the best forward returns.
#:
#: The *structural* caps, by contrast, are close to free (29.77% vs 29.81% CAGR,
#: identical drawdown) because a top-5 equal-weight book sits just under them
#: anyway. They cost nothing in the normal case and still bound the damage from
#: a bug, a runaway position, or accidental leverage. That asymmetry is why they
#: are separable.
RISK_PROFILES: dict[str, RiskLimits] = {
    # Everything on: both structural caps and both reactive rails.
    "full": RiskLimits(),
    # Structural caps only. Keeps the cheap insurance, drops the two rails that
    # measurably hurt this strategy.
    "caps": RiskLimits(stop_loss=None, crash_drawdown=None),
}


def limits_for(profile: str) -> RiskLimits:
    """Return the named risk profile.

    Args:
        profile: A key of :data:`RISK_PROFILES`.

    Returns:
        The corresponding :class:`RiskLimits`.

    Raises:
        ValueError: If ``profile`` is not a known name.
    """
    try:
        return RISK_PROFILES[profile]
    except KeyError:
        raise ValueError(
            f"Unknown risk profile {profile!r}. "
            f"Choose from: {', '.join(RISK_PROFILES)}."
        ) from None
