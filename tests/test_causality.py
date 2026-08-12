"""Prefix-invariance detection: known-causal strategies pass, known-peeking ones are named."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest

from holdout_first.causality import LookaheadError, assert_causal, coerce_positions
from holdout_first.splits import walk_forward_periods
from holdout_first.synthetic import MomentumRule, OverfittedLookup, PeekingRule, make_panel

PRICES = make_panel(3, n_instruments=1, n_bars=200)["SYN_00"]


class CentredAverage:
    """A rule whose moving average is centred on the current bar, so it reads ahead."""

    def __init__(self, half_window: int = 5) -> None:
        self.half_window = half_window
        self.n_parameters = 1

    def positions(self, prices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        out = np.zeros(prices.size, dtype=np.float64)
        half = self.half_window
        for index in range(half, prices.size - half):
            centred = float(np.mean(prices[index - half : index + half + 1]))
            out[index] = 1.0 if prices[index] > centred else -1.0
        return out


class FullSampleZScore:
    """A rule normalising by statistics of the entire sample, including the future."""

    def __init__(self) -> None:
        self.n_parameters = 1

    def positions(self, prices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.sign(prices - float(np.mean(prices)))


class WrongLength:
    """A rule returning fewer positions than bars."""

    def __init__(self) -> None:
        self.n_parameters = 0

    def positions(self, prices: npt.NDArray[np.float64]) -> Sequence[float]:
        return [0.0] * (prices.size - 1)


class OutOfRange:
    """A rule returning a levered position the protocol does not allow."""

    def __init__(self) -> None:
        self.n_parameters = 0

    def positions(self, prices: npt.NDArray[np.float64]) -> Sequence[float]:
        return [3.0] * prices.size


def test_causal_momentum_rule_passes_and_returns_full_positions() -> None:
    positions = assert_causal(MomentumRule(lookback=20), PRICES)
    assert positions.shape == PRICES.shape
    assert set(np.unique(positions)).issubset({-1.0, 0.0, 1.0})


def test_flat_strategy_is_trivially_causal() -> None:
    class Flat:
        n_parameters = 0

        def positions(self, prices: npt.NDArray[np.float64]) -> Sequence[float]:
            return [0.0] * prices.size

    assert np.all(assert_causal(Flat(), PRICES) == 0.0)


def test_fitted_lookup_table_is_causal_even_though_it_is_overfitted() -> None:
    panel = make_panel(3, n_instruments=1, n_bars=200)
    splits = walk_forward_periods(200, 2, 0.30)
    strategy = OverfittedLookup.fit(panel, splits)
    positions = assert_causal(strategy, PRICES)
    assert positions.shape == PRICES.shape


def test_peeking_rule_is_caught_at_the_last_bar_of_the_prefix() -> None:
    with pytest.raises(LookaheadError) as info:
        assert_causal(PeekingRule(), PRICES, prefix_fractions=(0.5,))
    error = info.value
    assert error.prefix_length == 100
    assert error.index == 99
    assert error.prefix_value == 0.0
    assert error.full_value != 0.0


def test_peeking_rule_index_tracks_the_prefix_that_exposed_it() -> None:
    with pytest.raises(LookaheadError) as info:
        assert_causal(PeekingRule(), PRICES, prefix_fractions=(0.25,))
    assert info.value.index == 49


def test_centred_average_leaks_by_half_a_window() -> None:
    with pytest.raises(LookaheadError) as info:
        assert_causal(CentredAverage(half_window=5), PRICES, prefix_fractions=(0.5,))
    # The prefix has no data past bar 99, so bars 95..99 differ; the first is 95.
    assert info.value.index == 95


def test_full_sample_normalisation_is_caught_inside_the_prefix() -> None:
    with pytest.raises(LookaheadError) as info:
        assert_causal(FullSampleZScore(), PRICES, prefix_fractions=(0.5,))
    error = info.value
    # The whole-sample mean shifts when later bars arrive, so a bar that sat above the
    # prefix mean can sit below the full-sample mean. The reported bar is that crossing.
    assert 0 <= error.index < error.prefix_length
    assert error.prefix_value != error.full_value


def test_lookahead_error_message_names_the_bar_and_both_values() -> None:
    with pytest.raises(LookaheadError, match="look-ahead detected at bar 99"):
        assert_causal(PeekingRule(), PRICES, prefix_fractions=(0.5,))


def test_lookahead_error_preserves_all_public_fields_and_exact_message() -> None:
    error = LookaheadError(3, 7, -1.0, 1.0)
    assert (error.index, error.prefix_length, error.prefix_value, error.full_value) == (
        3,
        7,
        -1.0,
        1.0,
    )
    assert str(error) == (
        "look-ahead detected at bar 3: the strategy returned -1.0 when shown the first 7 "
        "bars but 1.0 when shown the full series. A causal strategy cannot revise a past "
        "position after seeing future data."
    )


def test_duplicate_prefix_fractions_are_evaluated_once() -> None:
    calls: list[int] = []

    class Counting:
        n_parameters = 0

        def positions(self, prices: npt.NDArray[np.float64]) -> Sequence[float]:
            calls.append(prices.size)
            return [0.0] * prices.size

    assert_causal(Counting(), PRICES, prefix_fractions=(0.5, 0.5, 0.5005))
    # One full run plus a single truncated run, because all three fractions round to 100.
    assert calls == [200, 100]


def test_tolerance_admits_floating_point_reordering_but_not_real_changes() -> None:
    class Drifting:
        n_parameters = 0

        def positions(self, prices: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return np.full(prices.size, 1e-9 if prices.size == 200 else 0.0)

    assert_causal(Drifting(), PRICES, prefix_fractions=(0.5,), tolerance=1e-6)
    with pytest.raises(LookaheadError):
        assert_causal(Drifting(), PRICES, prefix_fractions=(0.5,), tolerance=1e-12)


def test_assert_causal_rejects_an_object_without_positions() -> None:
    with pytest.raises(TypeError, match="does not implement Strategy"):
        assert_causal(object(), PRICES)  # type: ignore[arg-type]


def test_assert_causal_names_an_object_missing_positions() -> None:
    with pytest.raises(TypeError, match=r"^object does not implement Strategy"):
        assert_causal(object(), PRICES)  # type: ignore[arg-type]


def test_assert_causal_rejects_a_wrong_length_result() -> None:
    with pytest.raises(ValueError, match="one position per bar"):
        assert_causal(WrongLength(), PRICES)


def test_assert_causal_rejects_positions_outside_the_unit_range() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        assert_causal(OutOfRange(), PRICES)


@pytest.mark.parametrize("fractions", [(), (0.0,), (1.0,), (-0.2,)])
def test_assert_causal_rejects_invalid_prefix_fractions(fractions: tuple[float, ...]) -> None:
    with pytest.raises(ValueError):
        assert_causal(MomentumRule(), PRICES, prefix_fractions=fractions)


def test_assert_causal_rejects_a_series_too_short_to_truncate() -> None:
    with pytest.raises(ValueError, match="at least 4 bars"):
        assert_causal(MomentumRule(), [100.0, 101.0, 102.0])


def test_assert_causal_rejects_non_positive_prices() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        assert_causal(MomentumRule(), [100.0, 101.0, -1.0, 102.0])


def test_assert_causal_rejects_zero_price_and_names_its_index() -> None:
    with pytest.raises(ValueError, match=r"index 2 is np.float64\(0.0\)"):
        assert_causal(MomentumRule(), [100.0, 101.0, 0.0, 102.0])


def test_assert_causal_rejects_two_dimensional_prices() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        assert_causal(MomentumRule(), np.full((4, 2), 100.0))


def test_assert_causal_rejects_non_finite_prices() -> None:
    with pytest.raises(ValueError, match="finite"):
        assert_causal(MomentumRule(), [100.0, 101.0, float("inf"), 102.0])


def test_assert_causal_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        assert_causal(MomentumRule(), PRICES, tolerance=-1.0)


def test_coerce_positions_accepts_a_plain_list() -> None:
    result = coerce_positions([0.0, 1.0, -1.0], 3)
    assert result.dtype == np.float64
    assert result.tolist() == [0.0, 1.0, -1.0]


def test_coerce_positions_rejects_non_numeric_output() -> None:
    with pytest.raises(TypeError, match="sequence of floats"):
        coerce_positions("not positions", 3)


def test_coerce_positions_rejects_two_dimensional_output() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        coerce_positions([[0.0, 1.0], [1.0, 0.0]], 2)


def test_coerce_positions_rejects_non_finite_output() -> None:
    with pytest.raises(ValueError, match="finite"):
        coerce_positions([0.0, float("nan")], 2)
