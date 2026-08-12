"""A hard budget on free parameters, denominated in observed trades.

Every free parameter is another axis along which noise can be fitted. A lookback, a
threshold, a filter, a regime switch, an entry of a lookup table: each one is a degree of
freedom that lets the configuration bend towards the particular sequence of prices it was
shown. The bend is invisible in-sample, because in-sample is precisely where it was
manufactured.

The defence is arithmetic, not statistics. A configuration must earn its parameters with
observations, and the observation that matters for a trading rule is a trade, not a bar. Ten
thousand bars during which the rule held one position is one decision, not ten thousand. The
default requirement of 50 trades per parameter is a convention, not a theorem; it is
deliberately blunt so that it is hard to argue with and impossible to tune.

This module contains no multiple-testing correction and never will. The remedy for a large
search is a large held-out set, evaluated across instruments and periods, not a p-value
adjusted after the fact.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["BudgetVerdict", "parameter_budget"]


@dataclass(frozen=True, slots=True)
class BudgetVerdict:
    """The outcome of a parameter-budget check, with the numbers behind it.

    Attributes:
        n_trades: Observed trades credited to the configuration.
        n_parameters: Free parameters the configuration declared.
        min_trades_per_parameter: Required ratio.
        trades_per_parameter: ``n_trades / n_parameters``, or infinity when the
            configuration declares no parameters.
        required_trades: Trades needed to satisfy the budget at the declared parameter
            count.
        passed: Whether ``trades_per_parameter >= min_trades_per_parameter``.
    """

    n_trades: int
    n_parameters: int
    min_trades_per_parameter: float
    trades_per_parameter: float
    required_trades: float
    passed: bool

    def describe(self) -> str:
        """Return a one-line human-readable summary of the verdict."""
        ratio = (
            "inf" if math.isinf(self.trades_per_parameter) else f"{self.trades_per_parameter:.2f}"
        )
        outcome = "pass" if self.passed else "fail"
        return (
            f"{outcome}: {self.n_trades} trades / {self.n_parameters} parameters "
            f"= {ratio} per parameter (need {self.min_trades_per_parameter:g}, "
            f"i.e. {self.required_trades:.0f} trades)"
        )


def parameter_budget(
    n_trades: int,
    n_parameters: int,
    min_trades_per_parameter: float = 50.0,
) -> BudgetVerdict:
    """Check whether a configuration has enough observed trades to justify its parameters.

    The rule is a single inequality, ``n_trades / n_parameters >= min_trades_per_parameter``,
    evaluated with ``>=`` so that a configuration sitting exactly on the boundary passes.

    A configuration declaring zero parameters is a fixed rule with nothing to fit, so the
    ratio is infinite and the check passes for any trade count, including zero. Declaring
    zero parameters for a rule that was in fact selected by looking at data is the one way
    to render this check useless, and the responsibility for that sits with the caller.

    Args:
        n_trades: Number of trades observed. Must be non-negative.
        n_parameters: Number of free parameters declared by the configuration. Must be
            non-negative.
        min_trades_per_parameter: Required trades per parameter. Must be positive.

    Returns:
        A :class:`BudgetVerdict` carrying the ratio, the requirement, and the pass/fail flag.

    Raises:
        TypeError: If ``n_trades`` or ``n_parameters`` is not an ``int``.
        ValueError: If either count is negative, or ``min_trades_per_parameter`` is not
            finite and positive.
    """
    if not isinstance(n_trades, int) or isinstance(n_trades, bool):
        raise TypeError(f"n_trades must be an int, got {type(n_trades).__name__}")
    if not isinstance(n_parameters, int) or isinstance(n_parameters, bool):
        raise TypeError(f"n_parameters must be an int, got {type(n_parameters).__name__}")
    if n_trades < 0:
        raise ValueError(f"n_trades must be non-negative, got {n_trades}")
    if n_parameters < 0:
        raise ValueError(
            f"n_parameters must be non-negative, got {n_parameters}. Count every value that "
            "was chosen by looking at data."
        )
    minimum = float(min_trades_per_parameter)
    if not math.isfinite(minimum) or minimum <= 0.0:
        raise ValueError(
            f"min_trades_per_parameter must be finite and positive, got "
            f"{min_trades_per_parameter!r}"
        )

    if n_parameters == 0:
        return BudgetVerdict(
            n_trades=n_trades,
            n_parameters=0,
            min_trades_per_parameter=minimum,
            trades_per_parameter=math.inf,
            required_trades=0.0,
            passed=True,
        )
    ratio = n_trades / n_parameters
    return BudgetVerdict(
        n_trades=n_trades,
        n_parameters=n_parameters,
        min_trades_per_parameter=minimum,
        trades_per_parameter=ratio,
        required_trades=minimum * n_parameters,
        passed=ratio >= minimum,
    )
