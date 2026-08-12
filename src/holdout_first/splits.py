"""Contiguous train/test splits that allocate most of the data to the held-out side.

The conventional split fits on roughly 70% of history and validates on the remaining 30%.
That gives the majority of the evidence to the step most prone to self-deception, and
leaves a test set small enough that a single benign stretch can carry it. This module
inverts the ratio: the default train fraction is 0.30, and asking for more than half the
data to fit on raises rather than warns.

Splits are always contiguous and always in time order. Nothing here shuffles, resamples, or
draws with replacement, because doing so on a price series manufactures an out-of-sample set
that is interleaved with the training data and therefore is not out of sample at all.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Split", "fraction_split", "walk_forward_periods"]

_DEFAULT_TRAIN_FRACTION = 0.30


@dataclass(frozen=True, slots=True)
class Split:
    """One contiguous train range and the held-out range that follows it.

    All bounds are half-open, ``[start, stop)``, and index into the original series.

    Attributes:
        train_start: First index of the training range.
        train_stop: One past the last index of the training range.
        test_start: First index of the held-out range. Equals ``train_stop + gap``.
        test_stop: One past the last index of the held-out range.
        gap: Number of bars discarded between train and test (the embargo).
    """

    train_start: int
    train_stop: int
    test_start: int
    test_stop: int
    gap: int

    @property
    def train_length(self) -> int:
        """Number of bars in the training range."""
        return self.train_stop - self.train_start

    @property
    def test_length(self) -> int:
        """Number of bars in the held-out range."""
        return self.test_stop - self.test_start

    @property
    def train_slice(self) -> slice:
        """The training range as a :class:`slice` for indexing the original series."""
        return slice(self.train_start, self.train_stop)

    @property
    def test_slice(self) -> slice:
        """The held-out range as a :class:`slice` for indexing the original series."""
        return slice(self.test_start, self.test_stop)


def _validate_common(n: int, train_fraction: float, gap: int, allow_large_train: bool) -> None:
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"n must be an int, got {type(n).__name__}")
    if not isinstance(gap, int) or isinstance(gap, bool):
        raise TypeError(f"gap must be an int, got {type(gap).__name__}")
    if gap < 0:
        raise ValueError(f"gap must be non-negative, got {gap}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must lie strictly inside (0, 1), got {train_fraction!r}. "
            "The default of 0.30 fits on a small slice and holds out the rest."
        )
    if train_fraction > 0.5 and not allow_large_train:
        raise ValueError(
            f"train_fraction={train_fraction!r} allocates more than half the data to "
            "fitting, which defeats the purpose of this library: the held-out set must be "
            "the larger of the two. Reduce train_fraction to 0.5 or below, or pass "
            "allow_large_train=True if you are deliberately reproducing a conventional "
            "70/30 split for comparison."
        )
    if n < 2:
        raise ValueError(f"n must be at least 2 to form a train and a test range, got {n}")


def fraction_split(
    n: int,
    train_fraction: float = _DEFAULT_TRAIN_FRACTION,
    *,
    gap: int = 0,
    allow_large_train: bool = False,
    offset: int = 0,
) -> Split:
    """Split ``n`` ordered bars into a small contiguous train range and a large test range.

    The training range takes the first ``floor(n * train_fraction)`` bars. The next ``gap``
    bars are discarded. Everything after that is held out.

    Why the gap exists: features and labels near the boundary overlap in time. A feature
    computed from a 20-bar window at the last training bar reads the same prices as a
    feature at the first test bar, and a label defined over the next 5 bars at the end of
    training resolves inside the test range. Without an embargo, the two sides share
    observations and the held-out Sharpe is inflated by construction. Set ``gap`` to at
    least the longer of the feature window and the label horizon.

    Args:
        n: Number of bars in the series.
        train_fraction: Fraction of ``n`` used for fitting. Must lie in ``(0, 1)`` and, by
            default, must not exceed 0.5.
        gap: Number of bars discarded between the train and test ranges (the embargo).
        allow_large_train: Escape hatch permitting ``train_fraction > 0.5``.
        offset: Value added to every returned index, so the split can address a block inside
            a longer series.

    Returns:
        A :class:`Split` with both ranges non-empty.

    Raises:
        TypeError: If ``n``, ``gap``, or ``offset`` is not an ``int``.
        ValueError: If ``train_fraction`` is outside ``(0, 1)``; if it exceeds 0.5 without
            ``allow_large_train``; if ``gap`` or ``offset`` is negative; or if ``n`` is too
            small to leave at least one bar on each side after the embargo.
    """
    _validate_common(n, train_fraction, gap, allow_large_train)
    if not isinstance(offset, int) or isinstance(offset, bool):
        raise TypeError(f"offset must be an int, got {type(offset).__name__}")
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")

    train_length = int(n * train_fraction)
    if train_length < 1:
        raise ValueError(
            f"train_fraction={train_fraction!r} over n={n} bars yields an empty training "
            "range; increase n or train_fraction"
        )
    test_start = train_length + gap
    if test_start >= n:
        raise ValueError(
            f"n={n} bars cannot accommodate a {train_length}-bar training range plus a "
            f"{gap}-bar embargo and still leave a held-out range; increase n or reduce gap"
        )
    return Split(
        train_start=offset,
        train_stop=offset + train_length,
        test_start=offset + test_start,
        test_stop=offset + n,
        gap=gap,
    )


def walk_forward_periods(
    n: int,
    n_periods: int,
    train_fraction: float = _DEFAULT_TRAIN_FRACTION,
    *,
    gap: int = 0,
    allow_large_train: bool = False,
) -> list[Split]:
    """Cut ``n`` bars into ``n_periods`` disjoint blocks and split each one independently.

    A single split has a single outcome, and a single outcome is a coin flip dressed as
    evidence. Carving history into disjoint blocks and requiring the configuration to work
    in each of them turns one observation into several, and makes a lucky boundary visible
    instead of decisive. Blocks do not overlap, so no bar is ever used twice.

    Block boundaries are placed by rounding ``i * n / n_periods``, which distributes any
    remainder across the blocks rather than dumping it into the last one.

    Args:
        n: Number of bars in the series.
        n_periods: Number of disjoint blocks. Must be at least 1.
        train_fraction: Fraction of each block used for fitting.
        gap: Embargo applied inside every block, between its train and test ranges.
        allow_large_train: Escape hatch permitting ``train_fraction > 0.5``.

    Returns:
        A list of ``n_periods`` :class:`Split` objects in chronological order.

    Raises:
        TypeError: If ``n``, ``n_periods``, or ``gap`` is not an ``int``.
        ValueError: If ``n_periods`` is below 1, or if any block is too short to hold a
            train range, the embargo, and a test range.
    """
    if not isinstance(n_periods, int) or isinstance(n_periods, bool):
        raise TypeError(f"n_periods must be an int, got {type(n_periods).__name__}")
    if n_periods < 1:
        raise ValueError(f"n_periods must be at least 1, got {n_periods}")
    _validate_common(n, train_fraction, gap, allow_large_train)

    bounds = [round(i * n / n_periods) for i in range(n_periods + 1)]
    splits: list[Split] = []
    for index in range(n_periods):
        start, stop = bounds[index], bounds[index + 1]
        block_length = stop - start
        try:
            splits.append(
                fraction_split(
                    block_length,
                    train_fraction,
                    gap=gap,
                    allow_large_train=allow_large_train,
                    offset=start,
                )
            )
        except ValueError as exc:
            raise ValueError(
                f"period {index} spans bars [{start}, {stop}) which is too short to split: {exc}"
            ) from exc
    return splits
