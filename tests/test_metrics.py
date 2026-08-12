"""Metric correctness against values computed by hand, plus the documented error paths."""

from __future__ import annotations

import math

import numpy as np
import pytest

from holdout_first import metrics as m


def test_total_return_compounds_geometrically() -> None:
    # 1.10 * 0.95 * 1.02 = 1.0659
    assert m.total_return([0.10, -0.05, 0.02]) == pytest.approx(0.0659, abs=1e-12)


def test_total_return_of_single_period_is_that_period() -> None:
    assert m.total_return([0.07]) == pytest.approx(0.07)


def test_equity_curve_starts_from_one_and_compounds() -> None:
    curve = m.equity_curve([0.10, -0.05])
    assert curve == pytest.approx([1.10, 1.045])


def test_annualized_return_is_geometric_not_arithmetic() -> None:
    # Four periods of +10% at two periods per year annualises to exactly 1.1**2 - 1.
    assert m.annualized_return([0.1, 0.1, 0.1, 0.1], 2.0) == pytest.approx(0.21)


def test_annualized_return_over_exactly_one_year_equals_total_return() -> None:
    assert m.annualized_return([0.2, 0.0], 2.0) == pytest.approx(m.total_return([0.2, 0.0]))


def test_annualized_return_is_below_arithmetic_mean_when_returns_vary() -> None:
    returns = [0.5, -0.5]
    arithmetic = float(np.mean(returns)) * 2.0
    assert m.annualized_return(returns, 2.0) < arithmetic


def test_annualized_volatility_uses_sample_std_and_sqrt_time() -> None:
    returns = [0.01, 0.03, -0.01, 0.05]
    expected = float(np.std(returns, ddof=1)) * math.sqrt(252.0)
    assert m.annualized_volatility(returns, 252.0) == pytest.approx(expected)
    # ddof=1 is strictly larger than the population figure, which is the common bug.
    assert m.annualized_volatility(returns, 252.0) > float(np.std(returns)) * math.sqrt(252.0)


def test_annualized_volatility_of_constant_series_is_zero() -> None:
    assert m.annualized_volatility([0.01, 0.01, 0.01], 252.0) == 0.0


def test_sharpe_annualization_matches_hand_computation() -> None:
    returns = [0.02, 0.02, 0.02, 0.06]
    # excess = returns - 0.04/4 = [0.01, 0.01, 0.01, 0.05]; mean 0.02, ddof=1 std 0.02.
    assert m.sharpe(returns, 4.0, risk_free_rate=0.04) == pytest.approx(2.0)


def test_sharpe_scales_by_sqrt_of_periods_per_year_exactly_once() -> None:
    returns = [0.01, -0.02, 0.03, 0.00, 0.015]
    daily = m.sharpe(returns, 1.0)
    assert m.sharpe(returns, 252.0) == pytest.approx(daily * math.sqrt(252.0))


def test_sharpe_of_zero_dispersion_series_is_zero() -> None:
    assert m.sharpe([0.01, 0.01, 0.01], 252.0) == 0.0


def test_sharpe_sign_follows_mean_excess_return() -> None:
    assert m.sharpe([0.02, -0.01, 0.03], 252.0) > 0.0
    assert m.sharpe([-0.02, 0.01, -0.03], 252.0) < 0.0


def test_max_drawdown_is_a_non_negative_magnitude() -> None:
    # equity 1.10, 1.045, 1.0659; peak 1.10; worst decline 1 - 1.045/1.10 = 0.05.
    assert m.max_drawdown([0.10, -0.05, 0.02]) == pytest.approx(0.05)


def test_max_drawdown_counts_a_decline_in_the_very_first_period() -> None:
    assert m.max_drawdown([-0.2, 0.1]) == pytest.approx(0.2)


def test_max_drawdown_of_monotonically_rising_curve_is_zero() -> None:
    assert m.max_drawdown([0.01, 0.02, 0.03]) == 0.0


def test_max_drawdown_measures_from_the_running_peak_not_the_start() -> None:
    # equity 2.0 then 1.0: the decline is measured from the peak of 2.0, so 50%.
    assert m.max_drawdown([1.0, -0.5]) == pytest.approx(0.5)


def test_hit_rate_excludes_flat_periods() -> None:
    assert m.hit_rate([0.1, 0.0, -0.05, 0.02]) == pytest.approx(2.0 / 3.0)


def test_hit_rate_is_nan_when_every_period_is_flat() -> None:
    assert math.isnan(m.hit_rate([0.0, 0.0, 0.0]))


def test_turnover_charges_the_opening_position_and_not_the_final_exit() -> None:
    # 0 -> 0.5 -> 0.5 -> -1.0 -> 0.0 costs 0.5 + 0 + 1.5 + 1.0 = 3.0.
    assert m.turnover([0.5, 0.5, -1.0, 0.0]) == pytest.approx(3.0)


def test_turnover_of_one_full_round_trip_is_two() -> None:
    assert m.turnover([1.0, 1.0, 1.0, 0.0]) == pytest.approx(2.0)


def test_turnover_of_a_permanently_flat_path_is_zero() -> None:
    assert m.turnover([0.0, 0.0, 0.0]) == 0.0


def test_trade_count_counts_changes_including_the_opening_position() -> None:
    assert m.trade_count([0.5, 0.5, -1.0, 0.0]) == 3


def test_trade_count_ignores_dust_below_tolerance() -> None:
    assert m.trade_count([1.0, 1.0 + 1e-15, 1.0]) == 1
    assert m.trade_count([1.0, 0.9, 1.0], tolerance=0.5) == 1


def test_strategy_returns_are_aligned_one_bar_forward() -> None:
    prices = [100.0, 110.0, 99.0, 99.0]
    positions = [1.0, -1.0, 0.5, 0.0]
    # price moves +10%, -10%, 0%; held positions 1, -1, 0.5.
    assert m.strategy_returns(positions, prices) == pytest.approx([0.1, 0.1, 0.0])


def test_strategy_returns_default_to_zero_fees() -> None:
    prices = [100.0, 110.0, 99.0, 99.0]
    positions = [1.0, -1.0, 0.5, 0.0]
    assert m.strategy_returns(positions, prices) == pytest.approx(
        m.strategy_returns(positions, prices, fees=0.0)
    )


def test_strategy_returns_charge_fees_per_unit_of_turnover() -> None:
    prices = [100.0, 110.0, 99.0, 99.0]
    positions = [1.0, -1.0, 0.5, 0.0]
    # turnover per step: |1-0|=1, |-1-1|=2, |0.5-(-1)|=1.5 at one percent each.
    assert m.strategy_returns(positions, prices, fees=0.01) == pytest.approx([0.09, 0.08, -0.015])


def test_strategy_returns_of_a_flat_position_are_zero() -> None:
    result = m.strategy_returns([0.0, 0.0, 0.0], [100.0, 120.0, 90.0], fees=0.01)
    assert result == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize(
    "returns",
    [[], [[0.1, 0.2], [0.3, 0.4]], [0.1, float("nan")], [0.1, float("inf")], [0.1, -1.0]],
)
def test_metrics_reject_malformed_return_series(returns: object) -> None:
    with pytest.raises(ValueError):
        m.total_return(returns)


def test_annualized_volatility_requires_two_observations() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        m.annualized_volatility([0.01], 252.0)


def test_sharpe_requires_two_observations() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        m.sharpe([0.01], 252.0)


@pytest.mark.parametrize("periods_per_year", [0.0, -1.0, float("nan"), float("inf")])
def test_metrics_reject_invalid_annualization_factor(periods_per_year: float) -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        m.sharpe([0.01, 0.02], periods_per_year)


def test_sharpe_rejects_non_finite_risk_free_rate() -> None:
    with pytest.raises(ValueError, match="risk_free_rate"):
        m.sharpe([0.01, 0.02], 252.0, risk_free_rate=float("nan"))


def test_turnover_rejects_positions_outside_the_unit_range() -> None:
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        m.turnover([0.5, 1.5])


def test_turnover_rejects_two_dimensional_positions() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        m.turnover([[0.0, 0.5], [0.5, 0.0]])


def test_turnover_rejects_non_finite_positions() -> None:
    with pytest.raises(ValueError, match="finite"):
        m.turnover([0.0, float("nan")])


def test_trade_count_rejects_negative_tolerance() -> None:
    with pytest.raises(ValueError, match="tolerance"):
        m.trade_count([0.5, 1.0], tolerance=-1.0)


def test_strategy_returns_reject_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal length"):
        m.strategy_returns([1.0, 0.0], [100.0, 101.0, 102.0])


def test_strategy_returns_reject_non_positive_prices() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        m.strategy_returns([1.0, 0.0], [100.0, 0.0])


def test_strategy_returns_reject_negative_fees() -> None:
    with pytest.raises(ValueError, match="fees"):
        m.strategy_returns([1.0, 0.0], [100.0, 101.0], fees=-0.001)


def test_strategy_returns_rejects_two_dimensional_prices() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        m.strategy_returns([0.0, 0.0], [[100.0, 101.0]])


def test_strategy_returns_rejects_non_finite_prices() -> None:
    with pytest.raises(ValueError, match="finite"):
        m.strategy_returns([0.0, 0.0], [100.0, float("inf")])


def test_strategy_returns_require_at_least_two_bars() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        m.strategy_returns([1.0], [100.0])
