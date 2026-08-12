"""Prefix-invariance test for look-ahead bias. This is the headline check of the library.

The idea is small enough to state in one sentence: if a strategy is causal, then running it
on the first ``k`` bars must produce exactly the positions it produces for those same ``k``
bars when the whole series is supplied.

Every common form of look-ahead breaks that invariance. A centred moving average at bar
``t`` reads bar ``t + w``, so appending data changes it. A full-sample z-score uses a mean
and a standard deviation that shift when the sample grows. A shift in the wrong direction,
a backward fill, a resample with a right-closed label, a target computed with
``prices[t + 1]``: all of them move earlier positions when later bars arrive.

The test cannot prove causality. A strategy that peeks only at the final bar of the series
will pass, because the truncated run has no final bar to peek past. It is a fast, cheap
falsifier that catches the mistakes people actually make, and the harness runs it before it
reports a single performance number.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import numpy.typing as npt

from .protocol import Strategy

__all__ = ["LookaheadError", "assert_causal", "coerce_positions"]

_DEFAULT_PREFIX_FRACTIONS: tuple[float, ...] = (0.35, 0.60, 0.85)
_POSITION_TOLERANCE = 1e-12


class LookaheadError(Exception):
    """Raised when a strategy's positions change once future bars are appended.

    Attributes:
        index: The first bar whose position differed between the truncated run and the full
            run. This is the earliest point at which the future leaked in, and is usually
            within one window length of the offending computation.
        prefix_length: Number of bars in the truncated run that exposed the difference.
        prefix_value: The position the strategy produced at ``index`` when it could only see
            ``prefix_length`` bars.
        full_value: The position it produced at the same bar when it could see everything.
    """

    def __init__(
        self,
        index: int,
        prefix_length: int,
        prefix_value: float,
        full_value: float,
    ) -> None:
        self.index = index
        self.prefix_length = prefix_length
        self.prefix_value = prefix_value
        self.full_value = full_value
        super().__init__(
            f"look-ahead detected at bar {index}: the strategy returned "
            f"{prefix_value!r} when shown the first {prefix_length} bars but {full_value!r} "
            "when shown the full series. A causal strategy cannot revise a past position "
            "after seeing future data."
        )


def coerce_positions(
    raw: object,
    expected_length: int,
    *,
    label: str = "positions",
) -> npt.NDArray[np.float64]:
    """Validate a strategy's output and convert it to a float array.

    Args:
        raw: Whatever ``Strategy.positions`` returned.
        expected_length: The number of bars the strategy was given.
        label: Name used in error messages, so failures identify which run misbehaved.

    Returns:
        A one-dimensional ``float64`` array of length ``expected_length``.

    Raises:
        TypeError: If the result cannot be interpreted as a numeric sequence.
        ValueError: If the result is not one-dimensional, has the wrong length, is
            non-finite, or leaves the ``[-1, 1]`` range required by the protocol.
    """
    try:
        array = np.asarray(cast(npt.ArrayLike, raw), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{label} must be a sequence of floats, got {type(raw).__name__}: {exc}"
        ) from exc
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional, got shape {array.shape}")
    if array.size != expected_length:
        raise ValueError(
            f"{label} must return one position per bar: expected {expected_length}, got "
            f"{array.size}"
        )
    if not np.all(np.isfinite(array)):
        bad = int(np.argmax(~np.isfinite(array)))
        raise ValueError(f"{label} must be finite; index {bad} is {array[bad]!r}")
    if np.any(np.abs(array) > 1.0 + _POSITION_TOLERANCE):
        worst = int(np.argmax(np.abs(array)))
        raise ValueError(
            f"{label} must lie in [-1, 1]; index {worst} is {array[worst]!r}. The protocol "
            "expresses a position as a signed fraction of capital, so leverage must be "
            "applied outside the harness."
        )
    return array


def assert_causal(
    strategy: Strategy,
    prices: npt.ArrayLike,
    *,
    prefix_fractions: tuple[float, ...] = _DEFAULT_PREFIX_FRACTIONS,
    tolerance: float = 1e-10,
) -> npt.NDArray[np.float64]:
    """Verify that a strategy's early positions do not change when later bars are appended.

    The strategy is run once on the full series and once per entry in ``prefix_fractions``
    on a truncated copy. Each truncated result must match the corresponding leading segment
    of the full result to within ``tolerance``.

    Args:
        strategy: Any object satisfying :class:`holdout_first.protocol.Strategy`.
        prices: A one-dimensional array of strictly positive prices.
        prefix_fractions: Fractions of the series length at which to truncate. Each must lie
            strictly inside ``(0, 1)``. Several are used because a leak with a long window
            may only surface at a particular truncation point.
        tolerance: Absolute difference below which two positions are considered equal.
            Accommodates the reordering of floating-point operations that numpy may perform
            on arrays of different lengths, which is not a causality violation.

    Returns:
        The positions produced on the full series, validated and converted to ``float64``.
        Callers that need the positions anyway can reuse this instead of re-running.

    Raises:
        LookaheadError: If any truncated run disagrees with the full run.
        TypeError: If ``strategy`` has no ``positions`` method or returns a non-numeric
            result.
        ValueError: If ``prices`` is invalid, the series is too short to truncate, any
            entry of ``prefix_fractions`` is outside ``(0, 1)``, or ``tolerance`` is
            negative.
    """
    if not hasattr(strategy, "positions"):
        raise TypeError(
            f"{type(strategy).__name__} does not implement Strategy: no 'positions' method"
        )
    price_array = np.asarray(prices, dtype=np.float64)
    if price_array.ndim != 1:
        raise ValueError(f"prices must be one-dimensional, got shape {price_array.shape}")
    if price_array.size < 4:
        raise ValueError(
            f"prices must contain at least 4 bars to form a meaningful prefix, got "
            f"{price_array.size}"
        )
    if not np.all(np.isfinite(price_array)):
        raise ValueError("prices must be finite; found nan or inf")
    if np.any(price_array <= 0.0):
        worst = int(np.argmin(price_array))
        raise ValueError(
            f"prices must be strictly positive; index {worst} is {price_array[worst]!r}"
        )
    if not prefix_fractions:
        raise ValueError("prefix_fractions must contain at least one fraction")
    limit = float(tolerance)
    if not np.isfinite(limit) or limit < 0.0:
        raise ValueError(f"tolerance must be finite and non-negative, got {tolerance!r}")

    n = price_array.size
    full = coerce_positions(strategy.positions(price_array), n, label="positions(full series)")

    seen: set[int] = set()
    for fraction in prefix_fractions:
        value = float(fraction)
        if not 0.0 < value < 1.0:
            raise ValueError(
                f"prefix_fractions entries must lie strictly inside (0, 1), got {fraction!r}"
            )
        prefix_length = max(2, min(n - 1, int(n * value)))
        if prefix_length in seen:
            continue
        seen.add(prefix_length)
        truncated = coerce_positions(
            strategy.positions(price_array[:prefix_length].copy()),
            prefix_length,
            label=f"positions(first {prefix_length} bars)",
        )
        difference = np.abs(truncated - full[:prefix_length])
        if np.any(difference > limit):
            index = int(np.argmax(difference > limit))
            raise LookaheadError(
                index=index,
                prefix_length=prefix_length,
                prefix_value=float(truncated[index]),
                full_value=float(full[index]),
            )
    return full
