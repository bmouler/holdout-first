"""End-to-end harness behaviour, including each survival rule failing on its own."""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
import pytest
from conftest import GAP, N_PERIODS, TRAIN_FRACTION

from holdout_first import evaluate
from holdout_first.harness import Report
from holdout_first.synthetic import MomentumRule, OverfittedLookup, PeekingRule

RULE_NAMES = ("causality", "parameter_budget", "test_sharpe_positive", "sharpe_retention")


def run(strategy: object, panel: dict[str, npt.NDArray[np.float64]], **kwargs: object) -> Report:
    """Evaluate with the demo's split configuration and any per-test overrides."""
    settings: dict[str, object] = {
        "n_periods": N_PERIODS,
        "train_fraction": TRAIN_FRACTION,
        "gap": GAP,
    }
    settings.update(kwargs)
    return evaluate(strategy, panel, **settings)  # type: ignore[arg-type]


class InflatedParameters(MomentumRule):
    """The honest rule, but declaring far more parameters than it can pay for."""

    def __init__(self) -> None:
        super().__init__(lookback=20)
        self.n_parameters = 1000


class InvertedMomentum(MomentumRule):
    """The honest rule with its sign flipped, so training performance is negative."""

    def positions(self, prices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return -super().positions(prices)


def test_honest_strategy_survives(panel: dict[str, npt.NDArray[np.float64]]) -> None:
    report = run(MomentumRule(lookback=20), panel)
    assert report.survived is True
    assert report.failed_rules == ()
    assert tuple(rule.name for rule in report.rules) == RULE_NAMES


def test_report_has_one_cell_per_instrument_and_period(
    panel: dict[str, npt.NDArray[np.float64]],
) -> None:
    report = run(MomentumRule(lookback=20), panel)
    assert len(report.cells) == len(panel) * N_PERIODS
    assert report.instruments == tuple(panel)
    assert {cell.period for cell in report.cells} == {0, 1, 2}


def test_held_out_segments_are_larger_than_training_segments(
    panel: dict[str, npt.NDArray[np.float64]],
) -> None:
    report = run(MomentumRule(lookback=20), panel)
    assert all(cell.test.n_bars > cell.train.n_bars for cell in report.cells)
    assert all(cell.split.test_start - cell.split.train_stop == GAP for cell in report.cells)


def test_overfitted_strategy_is_rejected_and_says_why(
    panel: dict[str, npt.NDArray[np.float64]], overfitted: OverfittedLookup
) -> None:
    report = run(overfitted, panel)
    assert report.survived is False
    failed = {rule.name for rule in report.failed_rules}
    assert "parameter_budget" in failed
    assert "sharpe_retention" in failed
    assert report.rule("causality").passed is True


def test_overfitted_training_sharpe_far_exceeds_held_out_sharpe(
    panel: dict[str, npt.NDArray[np.float64]], overfitted: OverfittedLookup
) -> None:
    report = run(overfitted, panel)
    mean_train = float(np.mean([cell.train.sharpe for cell in report.cells]))
    mean_test = float(np.mean([cell.test.sharpe for cell in report.cells]))
    assert mean_train > 1.0
    assert mean_test < 0.5 * mean_train


def test_causality_rule_fails_in_isolation(panel: dict[str, npt.NDArray[np.float64]]) -> None:
    report = run(PeekingRule(), panel)
    assert report.survived is False
    assert tuple(rule.name for rule in report.failed_rules) == ("causality",)
    assert "look-ahead detected on" in report.rule("causality").detail


def test_evaluate_reports_look_ahead_instead_of_raising(
    panel: dict[str, npt.NDArray[np.float64]],
) -> None:
    report = run(PeekingRule(), panel)
    assert "at bar" in report.rule("causality").detail
    assert report.rule("causality").observed == 0.0
    assert report.rule("causality").threshold == float(len(panel))


def test_parameter_budget_rule_fails_in_isolation(
    panel: dict[str, npt.NDArray[np.float64]],
) -> None:
    report = run(InflatedParameters(), panel)
    assert report.survived is False
    assert tuple(rule.name for rule in report.failed_rules) == ("parameter_budget",)
    assert report.budget.n_parameters == 1000
    assert report.budget.required_trades == 50000.0


def test_test_sharpe_positive_rule_fails_in_isolation(
    panel: dict[str, npt.NDArray[np.float64]],
) -> None:
    report = run(MomentumRule(lookback=20), panel, min_positive_test_fraction=1.0)
    assert report.survived is False
    assert tuple(rule.name for rule in report.failed_rules) == ("test_sharpe_positive",)
    assert 0.0 < report.rule("test_sharpe_positive").observed < 1.0


def test_sharpe_retention_rule_fails_in_isolation(
    panel: dict[str, npt.NDArray[np.float64]],
) -> None:
    report = run(MomentumRule(lookback=20), panel, min_sharpe_retention=1.0)
    assert report.survived is False
    assert tuple(rule.name for rule in report.failed_rules) == ("sharpe_retention",)
    assert report.rule("sharpe_retention").observed < 1.0


def test_non_positive_training_sharpe_fails_retention_with_a_specific_reason(
    panel: dict[str, npt.NDArray[np.float64]],
) -> None:
    report = run(InvertedMomentum(), panel)
    verdict = report.rule("sharpe_retention")
    assert verdict.passed is False
    assert verdict.observed == 0.0
    assert "not positive" in verdict.detail


def test_fees_reduce_held_out_returns(panel: dict[str, npt.NDArray[np.float64]]) -> None:
    free = run(MomentumRule(lookback=20), panel)
    costed = run(MomentumRule(lookback=20), panel, fees=0.001)
    assert free.settings.fees == 0.0
    assert all(
        costed_cell.test.total_return < free_cell.test.total_return
        for free_cell, costed_cell in zip(free.cells, costed.cells, strict=True)
    )


def test_settings_are_recorded_on_the_report(panel: dict[str, npt.NDArray[np.float64]]) -> None:
    report = run(MomentumRule(lookback=20), panel, periods_per_year=52.0)
    assert report.settings.periods_per_year == 52.0
    assert report.settings.train_fraction == TRAIN_FRACTION
    assert report.settings.gap == GAP
    assert report.n_parameters == 1
    assert report.strategy_name == "MomentumRule"


def test_annualization_factor_scales_every_sharpe(
    panel: dict[str, npt.NDArray[np.float64]],
) -> None:
    daily = run(MomentumRule(lookback=20), panel, periods_per_year=252.0)
    weekly = run(MomentumRule(lookback=20), panel, periods_per_year=52.0)
    ratio = np.sqrt(252.0 / 52.0)
    assert daily.cells[0].test.sharpe == pytest.approx(weekly.cells[0].test.sharpe * ratio)


def test_report_serialises_to_json(panel: dict[str, npt.NDArray[np.float64]]) -> None:
    payload = json.loads(json.dumps(run(MomentumRule(lookback=20), panel).to_dict()))
    assert payload["survived"] is True
    assert payload["strategy"] == "MomentumRule"
    assert len(payload["cells"]) == len(panel) * N_PERIODS
    assert [rule["name"] for rule in payload["rules"]] == list(RULE_NAMES)


def test_format_text_states_the_verdict_and_lists_every_rule(
    panel: dict[str, npt.NDArray[np.float64]],
) -> None:
    text = run(MomentumRule(lookback=20), panel).format_text()
    assert "verdict: SURVIVED" in text
    assert all(name in text for name in RULE_NAMES)
    assert "SYN_00" in text


def test_format_text_names_the_failing_rules(panel: dict[str, npt.NDArray[np.float64]]) -> None:
    text = run(InflatedParameters(), panel).format_text()
    assert "verdict: REJECTED (failing rules: parameter_budget)" in text
    assert "[FAIL] parameter_budget" in text


def test_rule_lookup_rejects_an_unknown_name(
    panel: dict[str, npt.NDArray[np.float64]],
) -> None:
    report = run(MomentumRule(lookback=20), panel)
    with pytest.raises(KeyError, match="no rule named"):
        report.rule("deflated_sharpe")


def test_panel_must_not_be_empty() -> None:
    with pytest.raises(ValueError, match="at least one instrument"):
        evaluate(MomentumRule(), {})


def test_panel_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="panel must be a mapping"):
        evaluate(MomentumRule(), [1.0, 2.0])  # type: ignore[arg-type]


def test_panel_must_be_rectangular() -> None:
    with pytest.raises(ValueError, match="same number of bars"):
        evaluate(MomentumRule(), {"a": np.full(200, 100.0), "b": np.full(199, 100.0)})


def test_panel_prices_must_be_positive() -> None:
    prices = np.full(200, 100.0)
    prices[7] = -1.0
    with pytest.raises(ValueError, match="strictly positive"):
        evaluate(MomentumRule(), {"a": prices})


def test_panel_prices_must_be_finite() -> None:
    prices = np.full(200, 100.0)
    prices[7] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate(MomentumRule(), {"a": prices})


def test_strategy_must_declare_an_integer_parameter_count(
    panel: dict[str, npt.NDArray[np.float64]],
) -> None:
    class Undeclared:
        n_parameters = "one"

        def positions(self, prices: npt.NDArray[np.float64]) -> Sequence[float]:
            return [0.0] * prices.size

    with pytest.raises(TypeError, match="n_parameters must be an int"):
        run(Undeclared(), panel)


def test_strategy_must_implement_positions(panel: dict[str, npt.NDArray[np.float64]]) -> None:
    with pytest.raises(TypeError, match="does not implement Strategy"):
        run(object(), panel)


@pytest.mark.parametrize("value", [0.0, 1.5, -0.2])
def test_supermajority_threshold_must_lie_in_the_unit_interval(
    panel: dict[str, npt.NDArray[np.float64]], value: float
) -> None:
    with pytest.raises(ValueError, match="min_positive_test_fraction"):
        run(MomentumRule(), panel, min_positive_test_fraction=value)


def test_retention_threshold_must_lie_in_the_unit_interval(
    panel: dict[str, npt.NDArray[np.float64]],
) -> None:
    with pytest.raises(ValueError, match="min_sharpe_retention"):
        run(MomentumRule(), panel, min_sharpe_retention=0.0)


def test_fees_must_be_non_negative(panel: dict[str, npt.NDArray[np.float64]]) -> None:
    with pytest.raises(ValueError, match="fees"):
        run(MomentumRule(), panel, fees=-0.1)


def test_periods_per_year_must_be_positive(panel: dict[str, npt.NDArray[np.float64]]) -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        run(MomentumRule(), panel, periods_per_year=0.0)


def test_evaluate_refuses_a_conventional_large_training_split(
    panel: dict[str, npt.NDArray[np.float64]],
) -> None:
    with pytest.raises(ValueError, match="allow_large_train"):
        run(MomentumRule(), panel, train_fraction=0.70)


def test_evaluate_accepts_a_large_training_split_via_the_escape_hatch(
    panel: dict[str, npt.NDArray[np.float64]],
) -> None:
    report = run(MomentumRule(), panel, train_fraction=0.70, allow_large_train=True)
    assert all(cell.train.n_bars > cell.test.n_bars for cell in report.cells)


def test_evaluate_rejects_segments_that_are_too_short() -> None:
    panel = {"a": np.cumprod(np.full(40, 1.001)) * 100.0}
    with pytest.raises(ValueError, match="at least 3 bars"):
        evaluate(MomentumRule(), panel, n_periods=6, train_fraction=0.30)
