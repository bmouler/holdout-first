"""The synthetic panel is reproducible, and the reference strategies behave as advertised."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from holdout_first.splits import Split, walk_forward_periods
from holdout_first.synthetic import MomentumRule, OverfittedLookup, PeekingRule, make_panel


def test_same_seed_produces_an_identical_panel() -> None:
    first = make_panel(7, n_instruments=2, n_bars=300)
    second = make_panel(7, n_instruments=2, n_bars=300)
    assert first.keys() == second.keys()
    assert all(np.array_equal(first[name], second[name]) for name in first)


def test_different_seeds_produce_different_panels() -> None:
    assert not np.array_equal(
        make_panel(7, n_instruments=1, n_bars=300)["SYN_00"],
        make_panel(8, n_instruments=1, n_bars=300)["SYN_00"],
    )


def test_panel_shape_and_names() -> None:
    panel = make_panel(1, n_instruments=3, n_bars=250)
    assert list(panel) == ["SYN_00", "SYN_01", "SYN_02"]
    assert all(series.shape == (250,) for series in panel.values())


def test_make_panel_defaults_are_part_of_the_public_contract() -> None:
    panel = make_panel(17)
    assert list(panel) == [f"SYN_{index:02d}" for index in range(5)]
    assert all(series.shape == (3000,) for series in panel.values())
    assert panel["SYN_00"][0] == pytest.approx(101.52632496183232)


def test_make_panel_matches_the_documented_seeded_ar1_process() -> None:
    panel = make_panel(
        123,
        n_instruments=2,
        n_bars=6,
        drift_persistence=0.5,
        drift_scale=0.002,
        noise_scale=0.003,
        initial_price=50.0,
    )
    expected = {
        "SYN_00": [
            50.017646468424445,
            49.90644347714161,
            49.93772684384383,
            50.00886210184742,
            49.887969884639766,
            50.17014916686158,
        ],
        "SYN_01": [
            49.702656247925916,
            49.92900467470913,
            50.22339355505434,
            50.55836393838513,
            50.69819838999663,
            50.661329067866475,
        ],
    }
    for name, prices in panel.items():
        assert prices.dtype == np.float64
        assert prices == pytest.approx(expected[name])


def test_make_panel_accepts_the_smallest_documented_shape() -> None:
    panel = make_panel(0, n_instruments=1, n_bars=2)
    assert list(panel) == ["SYN_00"]
    assert panel["SYN_00"].shape == (2,)


@pytest.mark.parametrize("name", ["n_instruments", "n_bars"])
def test_make_panel_reports_each_non_integer_count(name: str) -> None:
    with pytest.raises(TypeError, match=rf"^{name} must be an int, got float$"):
        make_panel(0, **{name: 1.5})  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["drift_scale", "noise_scale", "initial_price"])
def test_make_panel_reports_each_invalid_scale(name: str) -> None:
    with pytest.raises(ValueError, match=rf"^{name} must be finite and positive"):
        make_panel(0, **{name: 0.0})  # type: ignore[arg-type]


def test_panel_prices_are_strictly_positive_and_finite() -> None:
    panel = make_panel(2, n_instruments=4, n_bars=1000)
    for series in panel.values():
        assert np.all(series > 0.0)
        assert np.all(np.isfinite(series))


def test_instruments_within_a_panel_are_not_copies_of_each_other() -> None:
    panel = make_panel(5, n_instruments=2, n_bars=500)
    assert not np.array_equal(panel["SYN_00"], panel["SYN_01"])


def test_zero_persistence_drift_still_produces_a_valid_panel() -> None:
    panel = make_panel(4, n_instruments=1, n_bars=100, drift_persistence=0.0)
    assert np.all(panel["SYN_00"] > 0.0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_instruments": 0}, "n_instruments"),
        ({"n_bars": 1}, "n_bars"),
        ({"drift_persistence": 1.0}, "drift_persistence"),
        ({"drift_persistence": -0.1}, "drift_persistence"),
        ({"drift_scale": 0.0}, "drift_scale"),
        ({"noise_scale": -1.0}, "noise_scale"),
        ({"initial_price": 0.0}, "initial_price"),
    ],
)
def test_make_panel_rejects_invalid_arguments(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make_panel(0, **kwargs)  # type: ignore[arg-type]


def test_make_panel_rejects_a_non_integer_seed() -> None:
    with pytest.raises(TypeError, match="seed must be an int"):
        make_panel(1.5)  # type: ignore[arg-type]


def test_momentum_rule_declares_one_parameter() -> None:
    assert MomentumRule(lookback=20).n_parameters == 1
    assert MomentumRule().lookback == 20


def test_momentum_rule_is_flat_until_its_window_is_full() -> None:
    prices = make_panel(3, n_instruments=1, n_bars=100)["SYN_00"]
    positions = MomentumRule(lookback=20).positions(prices)
    assert np.all(positions[:20] == 0.0)
    assert np.any(positions[20:] != 0.0)


def test_momentum_rule_holds_the_sign_of_the_trailing_window() -> None:
    prices = np.array([100.0, 101.0, 102.0, 99.0, 98.0], dtype=np.float64)
    positions = MomentumRule(lookback=2).positions(prices)
    # bar 2: 102 vs 100 -> long. bar 3: 99 vs 101 -> short. bar 4: 98 vs 102 -> short.
    assert positions.tolist() == [0.0, 0.0, 1.0, -1.0, -1.0]


def test_momentum_rule_returns_all_zeros_for_a_series_shorter_than_the_window() -> None:
    positions = MomentumRule(lookback=20).positions(np.full(10, 100.0))
    assert positions.tolist() == [0.0] * 10


def test_momentum_rule_positions_are_discrete() -> None:
    prices = make_panel(3, n_instruments=1, n_bars=500)["SYN_00"]
    positions = MomentumRule(lookback=20).positions(prices)
    assert set(np.unique(positions)).issubset({-1.0, 0.0, 1.0})


@pytest.mark.parametrize("lookback", [0, -5])
def test_momentum_rule_rejects_a_non_positive_lookback(lookback: int) -> None:
    with pytest.raises(ValueError, match="lookback"):
        MomentumRule(lookback=lookback)


def test_momentum_rule_rejects_a_non_integer_lookback() -> None:
    with pytest.raises(TypeError, match="lookback must be an int"):
        MomentumRule(lookback=20.0)  # type: ignore[arg-type]


def test_reference_strategies_reject_two_dimensional_prices() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        MomentumRule().positions(np.full((2, 2), 100.0))


def test_reference_strategies_reject_non_finite_prices() -> None:
    with pytest.raises(ValueError, match="finite"):
        MomentumRule().positions([100.0, float("nan")])


def test_reference_strategies_reject_non_positive_prices() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        MomentumRule().positions([100.0, 0.0])


def test_overfitted_lookup_declares_one_parameter_per_bucket() -> None:
    panel = make_panel(3, n_instruments=1, n_bars=400)
    splits = walk_forward_periods(400, 2, 0.30)
    strategy = OverfittedLookup.fit(panel, splits)
    assert strategy.n_parameters == OverfittedLookup.n_buckets == 128
    assert strategy.table.shape == (128,)


def test_overfitted_lookup_fit_is_deterministic() -> None:
    panel = make_panel(3, n_instruments=2, n_bars=400)
    splits = walk_forward_periods(400, 2, 0.30)
    first = OverfittedLookup.fit(panel, splits).table
    second = OverfittedLookup.fit(panel, splits).table
    assert np.array_equal(first, second)


def test_overfitted_lookup_positions_stay_within_the_unit_range() -> None:
    panel = make_panel(3, n_instruments=1, n_bars=400)
    splits = walk_forward_periods(400, 2, 0.30)
    positions = OverfittedLookup.fit(panel, splits).positions(panel["SYN_00"])
    assert np.all(np.abs(positions) <= 1.0)
    assert positions.shape == (400,)


def test_overfitted_lookup_trades_far_more_often_than_the_momentum_rule() -> None:
    panel = make_panel(3, n_instruments=1, n_bars=1000)
    splits = walk_forward_periods(1000, 2, 0.30)
    table = OverfittedLookup.fit(panel, splits)
    lookup_changes = int(np.count_nonzero(np.diff(table.positions(panel["SYN_00"]))))
    momentum_changes = int(np.count_nonzero(np.diff(MomentumRule(20).positions(panel["SYN_00"]))))
    assert lookup_changes > 5 * momentum_changes


def test_overfitted_lookup_stays_flat_through_feature_warmup() -> None:
    strategy = OverfittedLookup(np.ones(OverfittedLookup.n_buckets))
    assert strategy.positions(np.full(21, 100.0)).tolist() == [0.0] * 21


def test_overfitted_lookup_matches_a_hand_checked_training_fixture() -> None:
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
        dtype=np.float64,
    )
    strategy = OverfittedLookup.fit(
        {"X": prices}, [Split(train_start=0, train_stop=24, test_start=24, test_stop=25, gap=0)]
    )
    active = np.flatnonzero(strategy.table)
    assert active.tolist() == [42, 50, 94]
    assert strategy.table[active].tolist() == [1.0, 1.0, -1.0]
    assert strategy.positions(prices).tolist() == [0.0] * 21 + [1.0, -1.0, 1.0, 0.0]


def test_overfitted_lookup_rejects_a_wrong_shaped_table() -> None:
    with pytest.raises(ValueError, match=r"shape \(128,\)"):
        OverfittedLookup(np.zeros(10))


def test_overfitted_lookup_rejects_levered_table_entries() -> None:
    table = np.zeros(OverfittedLookup.n_buckets)
    table[0] = 2.0
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        OverfittedLookup(table)


def test_overfitted_lookup_rejects_non_finite_table_entries() -> None:
    table = np.zeros(OverfittedLookup.n_buckets)
    table[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        OverfittedLookup(table)


def test_overfitted_lookup_fit_requires_splits() -> None:
    with pytest.raises(ValueError, match="at least one Split"):
        OverfittedLookup.fit(make_panel(3, n_instruments=1, n_bars=200), [])


def test_overfitted_lookup_fit_requires_a_panel() -> None:
    with pytest.raises(ValueError, match="at least one instrument"):
        OverfittedLookup.fit({}, walk_forward_periods(200, 2, 0.30))


def test_overfitted_lookup_fit_rejects_training_ranges_shorter_than_the_warmup() -> None:
    panel = make_panel(3, n_instruments=1, n_bars=30)
    splits = walk_forward_periods(30, 1, 0.30)
    with pytest.raises(ValueError, match="feature warm-up"):
        OverfittedLookup.fit(panel, splits)


def test_overfitted_lookup_ignores_training_ranges_outside_the_series() -> None:
    panel = make_panel(3, n_instruments=1, n_bars=100)
    splits = [
        walk_forward_periods(100, 1, 0.30)[0],
        Split(train_start=100, train_stop=120, test_start=120, test_stop=140, gap=0),
    ]
    fitted = OverfittedLookup.fit(panel, splits)
    assert fitted.table.shape == (OverfittedLookup.n_buckets,)


def test_peeking_rule_declares_no_parameters() -> None:
    assert PeekingRule().n_parameters == 0


def test_peeking_rule_returns_the_sign_of_the_next_move() -> None:
    prices: npt.NDArray[np.float64] = np.array([100.0, 101.0, 99.0, 99.0], dtype=np.float64)
    assert PeekingRule().positions(prices).tolist() == [1.0, -1.0, 0.0, 0.0]


def test_peeking_rule_is_profitable_by_construction() -> None:
    prices = make_panel(3, n_instruments=1, n_bars=300)["SYN_00"]
    positions = PeekingRule().positions(prices)
    forward = prices[1:] / prices[:-1] - 1.0
    assert float(np.sum(positions[:-1] * forward)) > 0.0


def test_peeking_rule_requires_at_least_two_bars() -> None:
    with pytest.raises(ValueError, match="at least 2 bars"):
        PeekingRule().positions([100.0])
