"""Focused public-contract checks discovered by mutation testing."""

from __future__ import annotations

import numpy as np
import pytest

from holdout_first import evaluate
from holdout_first.causality import assert_causal, coerce_positions
from holdout_first.splits import Split, walk_forward_periods
from holdout_first.synthetic import MomentumRule, OverfittedLookup, make_panel


def test_public_array_results_are_float64_for_integer_inputs() -> None:
    assert MomentumRule(1).positions([100, 101]).dtype == np.float64
    panel = make_panel(1, n_instruments=1, n_bars=2)
    assert panel["SYN_00"].dtype == np.float64
    assert coerce_positions([0, 1], 2).dtype == np.float64


def test_momentum_accepts_one_bar_and_a_one_bar_lookback() -> None:
    assert MomentumRule(1).positions([100.0]).tolist() == [0.0]


def test_price_validation_reports_exact_bad_index() -> None:
    with pytest.raises(ValueError, match=r"index 1 is np.float64\(0.0\)"):
        MomentumRule().positions([100.0, 0.0])


def test_position_validation_reports_exact_bad_indices() -> None:
    with pytest.raises(ValueError, match=r"index 1 is np.float64\(nan\)"):
        coerce_positions([0.0, float("nan")], 2)
    with pytest.raises(ValueError, match=r"index 1 is np.float64\(2.0\)"):
        coerce_positions([0.0, 2.0], 2)


def test_zero_prices_are_rejected_by_causality_and_panel_validation() -> None:
    prices = [100.0, 101.0, 0.0, 102.0]
    with pytest.raises(ValueError, match="strictly positive"):
        assert_causal(MomentumRule(), prices)
    with pytest.raises(ValueError, match="strictly positive"):
        evaluate(MomentumRule(), {"x": np.array(prices * 50)})


def test_zero_tolerance_is_supported() -> None:
    assert assert_causal(MomentumRule(1), [100.0, 101.0, 102.0, 103.0], tolerance=0.0).shape == (4,)


def test_walk_forward_smallest_period_count_has_exact_bounds() -> None:
    assert walk_forward_periods(2, 1, 0.5) == [
        Split(train_start=0, train_stop=1, test_start=1, test_stop=2, gap=0)
    ]


def test_lookup_training_uses_each_selected_observation_once() -> None:
    prices = np.array(
        [
            100,
            102,
            101,
            105,
            103,
            108,
            107,
            109,
            106,
            110,
            111,
            108,
            112,
            115,
            113,
            117,
            116,
            120,
            119,
            121,
            123,
            122,
            125,
            124,
            128,
        ],
        dtype=float,
    )
    split = Split(0, 24, 24, 25, 0)
    fitted = OverfittedLookup.fit({"x": prices}, [split])
    active = np.flatnonzero(fitted.table)
    assert active.tolist() == [42, 50, 94]
    assert fitted.table[active].tolist() == [1.0, 1.0, -1.0]
