"""Deterministic property tests for metrics, splits, and causality."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given
from hypothesis import strategies as st

from holdout_first import (
    LookaheadError,
    assert_causal,
    equity_curve,
    fraction_split,
    total_return,
    walk_forward_periods,
)

RETURNS = st.lists(
    st.floats(
        min_value=-0.05,
        max_value=0.05,
        allow_nan=False,
        allow_infinity=False,
    ),
    min_size=2,
    max_size=200,
).map(lambda values: np.asarray(values, dtype=np.float64))


@st.composite
def split_cases(draw: st.DrawFn) -> tuple[int, int, float, int, int]:
    """Generate parameters for which every walk-forward block is splittable."""
    n_periods = draw(st.integers(min_value=1, max_value=20))
    n = draw(st.integers(min_value=4 * n_periods, max_value=200))
    train_fraction = draw(
        st.floats(
            min_value=0.25,
            max_value=0.50,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    bounds = [round(index * n / n_periods) for index in range(n_periods + 1)]
    max_gap = min(
        stop - start - int((stop - start) * train_fraction) - 1
        for start, stop in zip(bounds[:-1], bounds[1:], strict=True)
    )
    gap = draw(st.integers(min_value=0, max_value=max_gap))
    offset = draw(st.integers(min_value=0, max_value=200))
    return n, n_periods, train_fraction, gap, offset


@dataclass(frozen=True)
class LaggedRule:
    """A causal rule whose position at i depends only on the price at i - 1."""

    n_parameters: int = 0

    def positions(self, prices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        positions = np.zeros(prices.size, dtype=np.float64)
        positions[1:] = np.sign(prices[:-1] - prices[0])
        return positions


@dataclass(frozen=True)
class OneBarLookaheadRule:
    """A non-causal rule whose position at i reads the price at i + 1."""

    n_parameters: int = 0

    def positions(self, prices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        positions = np.zeros(prices.size, dtype=np.float64)
        positions[:-1] = np.sign(prices[1:] - prices[:-1])
        return positions


@given(returns=RETURNS)
def test_equity_curve_has_one_base_and_matches_total_return(
    returns: npt.NDArray[np.float64],
) -> None:
    curve = equity_curve(returns)
    wealth_with_base = np.concatenate(([1.0], curve))

    assert wealth_with_base[0] == 1.0
    assert wealth_with_base[-1] == pytest.approx(1.0 + total_return(returns))


@given(case=split_cases())
def test_splits_follow_documented_exact_boundaries(
    case: tuple[int, int, float, int, int],
) -> None:
    n, n_periods, train_fraction, gap, offset = case

    direct = fraction_split(n, train_fraction, gap=gap, offset=offset)
    direct_train_length = int(n * train_fraction)
    assert direct.train_start == offset
    assert direct.train_stop == offset + direct_train_length
    assert direct.test_start == direct.train_stop + gap
    assert direct.test_stop == offset + n

    bounds = [round(index * n / n_periods) for index in range(n_periods + 1)]
    splits = walk_forward_periods(n, n_periods, train_fraction, gap=gap)
    assert len(splits) == n_periods

    for index, split in enumerate(splits):
        start, stop = bounds[index], bounds[index + 1]
        block_length = stop - start
        assert split == fraction_split(block_length, train_fraction, gap=gap, offset=start)
        assert split.train_start == start
        assert split.train_stop == start + int(block_length * train_fraction)
        assert split.test_start == split.train_stop + gap
        assert split.test_stop == stop
        assert 0 <= split.train_start < split.train_stop
        assert split.train_stop <= split.test_start < split.test_stop <= n
        if index:
            assert splits[index - 1].test_stop == split.train_start

    assert splits[0].train_start == 0
    assert splits[-1].test_stop == n


@given(returns=RETURNS)
def test_assert_causal_accepts_lagged_rule_and_rejects_one_bar_lookahead(
    returns: npt.NDArray[np.float64],
) -> None:
    positive_steps = np.abs(returns) + 1e-6
    prices = 100.0 + np.concatenate(([0.0, 1.0], 2.0 + np.cumsum(positive_steps)))

    expected = LaggedRule().positions(prices)
    assert assert_causal(LaggedRule(), prices) == pytest.approx(expected)
    with pytest.raises(LookaheadError):
        assert_causal(OneBarLookaheadRule(), prices)
