"""Parameter-budget arithmetic, especially at the boundary where the inequality is decided."""

from __future__ import annotations

import math

import pytest

from holdout_first.budget import parameter_budget


def test_exactly_meeting_the_ratio_passes() -> None:
    verdict = parameter_budget(100, 2, min_trades_per_parameter=50.0)
    assert verdict.trades_per_parameter == 50.0
    assert verdict.required_trades == 100.0
    assert verdict.passed is True


def test_one_trade_short_of_the_ratio_fails() -> None:
    verdict = parameter_budget(99, 2, min_trades_per_parameter=50.0)
    assert verdict.trades_per_parameter == 49.5
    assert verdict.passed is False


def test_one_trade_above_the_ratio_passes() -> None:
    assert parameter_budget(101, 2, min_trades_per_parameter=50.0).passed is True


def test_default_requirement_is_fifty_trades_per_parameter() -> None:
    assert parameter_budget(50, 1).passed is True
    assert parameter_budget(49, 1).passed is False


def test_zero_parameters_pass_with_an_infinite_ratio() -> None:
    verdict = parameter_budget(0, 0)
    assert math.isinf(verdict.trades_per_parameter)
    assert verdict.required_trades == 0.0
    assert verdict.passed is True
    assert (verdict.n_trades, verdict.n_parameters, verdict.min_trades_per_parameter) == (
        0,
        0,
        50.0,
    )


def test_zero_trades_with_parameters_always_fails() -> None:
    assert parameter_budget(0, 1).passed is False


def test_describe_reports_the_numbers_behind_the_verdict() -> None:
    text = parameter_budget(2133, 128).describe()
    assert "fail" in text
    assert "2133 trades" in text
    assert "128 parameters" in text
    assert "6400 trades" in text


def test_describe_renders_the_infinite_ratio_without_formatting_errors() -> None:
    assert "inf" in parameter_budget(10, 0).describe()


def test_a_larger_parameter_count_needs_proportionally_more_trades() -> None:
    assert parameter_budget(500, 10).passed is True
    assert parameter_budget(500, 11).passed is False


@pytest.mark.parametrize(("n_trades", "n_parameters"), [(-1, 1), (10, -1)])
def test_negative_counts_are_rejected(n_trades: int, n_parameters: int) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        parameter_budget(n_trades, n_parameters)
    with pytest.raises(
        ValueError,
        match=(
            r"n_parameters must be non-negative, got -1\. "
            r"Count every value that was chosen by looking at data\."
        ),
    ):
        parameter_budget(10, -1)


@pytest.mark.parametrize("value", [0.0, -5.0, float("nan"), float("inf")])
def test_invalid_minimum_ratio_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="min_trades_per_parameter"):
        parameter_budget(100, 1, min_trades_per_parameter=value)


def test_positive_minimum_below_one_is_accepted() -> None:
    assert parameter_budget(1, 2, min_trades_per_parameter=0.5).passed is True


def test_non_integer_counts_are_rejected() -> None:
    with pytest.raises(TypeError, match="n_trades must be an int"):
        parameter_budget(100.0, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="n_parameters must be an int"):
        parameter_budget(100, 1.0)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"got float$"):
        parameter_budget(100.0, 1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match=r"got float$"):
        parameter_budget(100, 1.0)  # type: ignore[arg-type]


def test_booleans_are_not_accepted_as_counts() -> None:
    with pytest.raises(TypeError):
        parameter_budget(True, 1)  # type: ignore[arg-type]
