"""Pure performance statistics over a return series, with the conventions written down.

Most public Sharpe implementations are silently wrong in one of four ways: they mix
logarithmic and simple returns, they use a population standard deviation where the sample
one is meant, they annualise a ratio that was computed on already-annualised inputs, or they
quietly assume 252 periods per year for data that is not daily. Every convention used here
is stated in the docstring of the function that depends on it, and ``periods_per_year`` is
always explicit and never has a default.

Conventions used throughout this module:

* Returns are **simple** per-period returns, ``p_t / p_{t-1} - 1``, not log returns.
* Compounding is geometric: an equity curve is ``cumprod(1 + r)`` starting from ``1.0``.
* Dispersion uses the **sample** standard deviation, ``ddof=1``.
* Annualisation of a volatility or a Sharpe ratio multiplies by ``sqrt(periods_per_year)``.
  This is the square-root-of-time rule and it assumes serially independent returns. No
  autocorrelation, Newey-West, or Lo adjustment is applied. If your returns are strongly
  autocorrelated the number is optimistic, and no amount of arithmetic here fixes that.
* Transaction costs default to ``0.0``. The zero-cost figure is the baseline; costs are an
  overlay you add deliberately, not a hidden constant.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "annualized_return",
    "annualized_volatility",
    "equity_curve",
    "hit_rate",
    "max_drawdown",
    "sharpe",
    "strategy_returns",
    "total_return",
    "trade_count",
    "turnover",
]

_POSITION_TOLERANCE = 1e-12


def _as_returns(returns: npt.ArrayLike, *, minimum: int = 1) -> npt.NDArray[np.float64]:
    array = np.asarray(returns, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"returns must be one-dimensional, got shape {array.shape}")
    if array.size < minimum:
        raise ValueError(f"returns must contain at least {minimum} element(s), got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError("returns must be finite; found nan or inf")
    if np.any(array <= -1.0):
        raise ValueError(
            "returns must exceed -1.0 (a return of -100% wipes the equity curve to zero and "
            "makes every compounded statistic meaningless)"
        )
    return array


def _as_positions(positions: npt.ArrayLike, *, minimum: int = 1) -> npt.NDArray[np.float64]:
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"positions must be one-dimensional, got shape {array.shape}")
    if array.size < minimum:
        raise ValueError(f"positions must contain at least {minimum} element(s), got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError("positions must be finite; found nan or inf")
    if np.any(np.abs(array) > 1.0 + _POSITION_TOLERANCE):
        worst = int(np.argmax(np.abs(array)))
        raise ValueError(
            f"positions must lie in [-1, 1]; index {worst} is {array[worst]!r}. Leverage is "
            "not modelled here, so scale your sizing before handing positions to the harness."
        )
    return array


def _check_periods_per_year(periods_per_year: float) -> float:
    value = float(periods_per_year)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"periods_per_year must be finite and positive, got {periods_per_year!r}")
    return value


def strategy_returns(
    positions: npt.ArrayLike,
    prices: npt.ArrayLike,
    fees: float = 0.0,
) -> npt.NDArray[np.float64]:
    """Convert a position path and a price path into a per-bar return series.

    This is the only place in the library that touches prices, and it does the minimum a
    validation harness needs. There is no order book, no slippage model, no borrow cost, and
    no financing.

    Alignment is causal by construction: ``positions[t]`` is held across the price move from
    bar ``t`` to bar ``t + 1``, so the output has one fewer element than the inputs. The
    position before the first bar is zero, so entering the first position is charged as
    turnover.

    Args:
        positions: Target positions in ``[-1, 1]``, one per bar, length ``n``.
        prices: Strictly positive prices, length ``n``.
        fees: Cost charged per unit of turnover, expressed as a fraction of capital. The
            default of ``0.0`` is deliberate: establish the zero-cost baseline first, then
            add costs as a separate, visible experiment. A value of ``0.0005`` charges five
            basis points to move the whole book from flat to fully long.

    Returns:
        An array of length ``n - 1``. Element ``t`` is
        ``positions[t] * (prices[t + 1] / prices[t] - 1)`` less
        ``fees * abs(positions[t] - positions[t - 1])``, with ``positions[-1]`` read as
        ``0.0``.

    Raises:
        ValueError: If the inputs are not one-dimensional and equal length, if fewer than
            two bars are supplied, if any price is non-positive or non-finite, if positions
            leave ``[-1, 1]``, or if ``fees`` is negative or non-finite.
    """
    position_array = _as_positions(positions, minimum=2)
    price_array = np.asarray(prices, dtype=np.float64)
    if price_array.ndim != 1:
        raise ValueError(f"prices must be one-dimensional, got shape {price_array.shape}")
    if price_array.size != position_array.size:
        raise ValueError(
            f"positions and prices must have equal length, got {position_array.size} and "
            f"{price_array.size}"
        )
    if not np.all(np.isfinite(price_array)):
        raise ValueError("prices must be finite; found nan or inf")
    if np.any(price_array <= 0.0):
        worst = int(np.argmin(price_array))
        raise ValueError(
            f"prices must be strictly positive; index {worst} is {price_array[worst]!r}"
        )
    fee_rate = float(fees)
    if not np.isfinite(fee_rate) or fee_rate < 0.0:
        raise ValueError(f"fees must be finite and non-negative, got {fees!r}")

    price_returns = price_array[1:] / price_array[:-1] - 1.0
    held = position_array[:-1]
    gross = held * price_returns
    if fee_rate == 0.0:
        return gross
    previous = np.concatenate(([0.0], held[:-1]))
    return gross - fee_rate * np.abs(held - previous)


def equity_curve(returns: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Compound a return series into a wealth index starting at ``1.0``.

    Args:
        returns: Simple per-period returns.

    Returns:
        An array of the same length as ``returns`` where element ``t`` is
        ``prod(1 + returns[: t + 1])``. The implied value before the first period is
        ``1.0`` and is not included.

    Raises:
        ValueError: If ``returns`` is empty, not one-dimensional, non-finite, or contains a
            value at or below ``-1.0``.
    """
    return np.asarray(np.cumprod(1.0 + _as_returns(returns)), dtype=np.float64)


def total_return(returns: npt.ArrayLike) -> float:
    """Compounded return over the whole series.

    Args:
        returns: Simple per-period returns.

    Returns:
        ``prod(1 + returns) - 1``. A value of ``0.25`` means the equity curve ended 25%
        above where it started.

    Raises:
        ValueError: If ``returns`` is empty, not one-dimensional, non-finite, or contains a
            value at or below ``-1.0``.
    """
    return float(np.prod(1.0 + _as_returns(returns)) - 1.0)


def annualized_return(returns: npt.ArrayLike, periods_per_year: float) -> float:
    """Geometric annualised return, also called the compound annual growth rate.

    The convention is ``(1 + total_return) ** (periods_per_year / n) - 1``, where ``n`` is
    the number of observations. This is the constant annual rate that would have produced
    the same terminal wealth. It is not the arithmetic mean return scaled by
    ``periods_per_year``, which is a different and larger number whenever returns vary.

    Args:
        returns: Simple per-period returns.
        periods_per_year: Observations per year for this sampling frequency. Common values
            are 252 for trading days, 52 for weeks, 12 for months.

    Returns:
        The compound annual growth rate as a fraction.

    Raises:
        ValueError: If ``returns`` is invalid or ``periods_per_year`` is not positive.
    """
    array = _as_returns(returns)
    scale = _check_periods_per_year(periods_per_year)
    growth = float(np.prod(1.0 + array))
    return float(growth ** (scale / array.size) - 1.0)


def annualized_volatility(returns: npt.ArrayLike, periods_per_year: float) -> float:
    """Annualised standard deviation of simple returns.

    The convention is ``std(returns, ddof=1) * sqrt(periods_per_year)``. The sample standard
    deviation is used because a return series is a sample, not a population. The
    square-root-of-time scaling assumes serial independence; it is not adjusted for
    autocorrelation.

    Args:
        returns: Simple per-period returns. At least two observations are required for a
            sample standard deviation to exist.
        periods_per_year: Observations per year for this sampling frequency.

    Returns:
        Annualised volatility as a fraction. Returns ``0.0`` for a constant series.

    Raises:
        ValueError: If fewer than two returns are supplied, the series is invalid, or
            ``periods_per_year`` is not positive.
    """
    array = _as_returns(returns, minimum=2)
    scale = _check_periods_per_year(periods_per_year)
    return float(np.std(array, ddof=1) * np.sqrt(scale))


def sharpe(
    returns: npt.ArrayLike,
    periods_per_year: float,
    risk_free_rate: float = 0.0,
) -> float:
    """Annualised Sharpe ratio of a simple return series.

    The convention is::

        excess = returns - risk_free_rate / periods_per_year
        sharpe = mean(excess) / std(excess, ddof=1) * sqrt(periods_per_year)

    Three choices are worth stating because implementations differ. The numerator is the
    **arithmetic** mean of per-period excess returns, not the geometric mean, because the
    ratio is a signal-to-noise measure rather than a growth rate. The denominator uses
    ``ddof=1``. The whole ratio is scaled by ``sqrt(periods_per_year)`` exactly once, so
    inputs must be per-period returns and never pre-annualised ones.

    No autocorrelation adjustment is applied, and none is planned. If the returns are
    serially dependent the true sampling error is larger than this figure implies, which is
    a reason to demand held-out survival rather than to patch the estimator.

    Args:
        returns: Simple per-period returns, at least two of them.
        periods_per_year: Observations per year for this sampling frequency.
        risk_free_rate: Annual risk-free rate as a fraction, converted to a per-period rate
            by dividing by ``periods_per_year``. Defaults to ``0.0``.

    Returns:
        The annualised Sharpe ratio. Returns ``0.0`` when the excess return series has zero
        dispersion, since a flat series carries no information either way.

    Raises:
        ValueError: If fewer than two returns are supplied, the series is invalid,
            ``periods_per_year`` is not positive, or ``risk_free_rate`` is not finite.
    """
    array = _as_returns(returns, minimum=2)
    scale = _check_periods_per_year(periods_per_year)
    rate = float(risk_free_rate)
    if not np.isfinite(rate):
        raise ValueError(f"risk_free_rate must be finite, got {risk_free_rate!r}")
    excess = array - rate / scale
    dispersion = float(np.std(excess, ddof=1))
    if dispersion == 0.0:
        return 0.0
    return float(np.mean(excess) / dispersion * np.sqrt(scale))


def max_drawdown(returns: npt.ArrayLike) -> float:
    """Largest peak-to-trough decline of the compounded equity curve.

    The equity curve starts at ``1.0`` before the first return, so a decline in the very
    first period counts. The result is reported as a **non-negative magnitude**: ``0.25``
    means the curve lost a quarter of its value from its running high. Sign conventions
    differ between libraries, which is exactly why this one is spelled out.

    Args:
        returns: Simple per-period returns.

    Returns:
        A value in ``[0, 1)``. Zero means the curve never traded below a prior high.

    Raises:
        ValueError: If ``returns`` is empty, not one-dimensional, non-finite, or contains a
            value at or below ``-1.0``.
    """
    curve = equity_curve(returns)
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], curve)))[1:]
    return float(np.max(1.0 - curve / running_peak))


def hit_rate(returns: npt.ArrayLike) -> float:
    """Fraction of non-zero periods that were positive.

    Flat periods are excluded from both the numerator and the denominator. A strategy that
    is out of the market for most of the sample should not be credited with, or penalised
    for, the bars it did not hold.

    Args:
        returns: Simple per-period returns.

    Returns:
        A value in ``[0, 1]``, or ``nan`` when every period is exactly zero and the
        quantity is undefined.

    Raises:
        ValueError: If ``returns`` is empty, not one-dimensional, non-finite, or contains a
            value at or below ``-1.0``.
    """
    array = _as_returns(returns)
    active = array[array != 0.0]
    if active.size == 0:
        return float("nan")
    return float(np.count_nonzero(active > 0.0) / active.size)


def turnover(positions: npt.ArrayLike) -> float:
    """Total absolute position change over the path.

    Turnover is ``sum(abs(diff(positions)))`` with an implied starting position of zero, so
    the initial entry is charged. The final position is **not** forced to zero, because
    liquidating at the end is a reporting artefact rather than a decision the strategy made.

    A value of ``2.0`` corresponds to one full round trip: flat to fully long and back to
    flat, or fully long to fully short.

    Args:
        positions: Target positions in ``[-1, 1]``, one per bar.

    Returns:
        Total turnover in units of capital.

    Raises:
        ValueError: If ``positions`` is empty, not one-dimensional, non-finite, or leaves
            ``[-1, 1]``.
    """
    array = _as_positions(positions)
    return float(np.abs(array[0]) + np.sum(np.abs(np.diff(array))))


def trade_count(positions: npt.ArrayLike, tolerance: float = _POSITION_TOLERANCE) -> int:
    """Number of bars at which the target position changed.

    The implied position before the first bar is zero, so a non-zero opening position counts
    as one trade. Adjusting an existing position counts as a trade as well, because a
    position adjustment is a decision the configuration made and therefore an observation
    the parameter budget is entitled to count.

    Args:
        positions: Target positions in ``[-1, 1]``, one per bar.
        tolerance: Absolute change below which two consecutive positions are treated as
            identical. Guards against floating-point dust being counted as activity.

    Returns:
        The number of position changes.

    Raises:
        ValueError: If ``positions`` is invalid or ``tolerance`` is negative or non-finite.
    """
    array = _as_positions(positions)
    limit = float(tolerance)
    if not np.isfinite(limit) or limit < 0.0:
        raise ValueError(f"tolerance must be finite and non-negative, got {tolerance!r}")
    changes = np.abs(np.diff(np.concatenate(([0.0], array))))
    return int(np.count_nonzero(changes > limit))
