"""The validation harness: run a strategy across a panel and apply the survival rules.

:func:`evaluate` takes a user-supplied strategy and a mapping of instrument name to price
array, cuts each instrument into disjoint walk-forward blocks, and reports train and
held-out statistics for every instrument-period cell. It then applies four survival rules.
All four must pass. Each one reports its own verdict together with the numbers that produced
it, so a failure states which rule failed and by how much rather than emitting a single
unhelpful boolean.

The rules are:

1. **causality** - every instrument passes the prefix-invariance test.
2. **parameter_budget** - trades observed in the training segments divided by the declared
   parameter count meets the minimum.
3. **test_sharpe_positive** - a supermajority of held-out cells have a positive Sharpe
   ratio. Consistency across instruments and periods is the evidence; a single spectacular
   cell is not.
4. **sharpe_retention** - mean held-out Sharpe is at least a fixed fraction of mean training
   Sharpe. Some decay is normal. A collapse means the training figure was describing the
   sample rather than the process.

There is no multiplicity testing anywhere in this module, and none is wanted. No probability
of backtest overfitting, no deflated Sharpe ratio, no Bonferroni, no false discovery rate,
no reality check. Validation here is expressed as held-out performance across instruments
and periods, which is a statement about the world rather than an adjustment to a p-value.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from . import metrics as m
from .budget import BudgetVerdict, parameter_budget
from .causality import (
    LookaheadError,
    _assert_causal_at_prefixes,
    _default_prefix_lengths,
    coerce_positions,
)
from .protocol import Strategy
from .splits import Split, walk_forward_periods

__all__ = [
    "CellResult",
    "EvaluationSettings",
    "Report",
    "RuleVerdict",
    "SegmentMetrics",
    "evaluate",
]

_MIN_SEGMENT_BARS = 3


@dataclass(frozen=True, slots=True)
class SegmentMetrics:
    """Performance statistics for one contiguous segment of one instrument.

    Attributes:
        n_bars: Bars in the segment.
        n_trades: Position changes inside the segment, counting the opening position.
        total_return: Compounded return over the segment.
        annualized_return: Compound annual growth rate.
        annualized_volatility: Sample standard deviation scaled by ``sqrt(periods_per_year)``.
        sharpe: Annualised Sharpe ratio.
        max_drawdown: Largest peak-to-trough decline, as a non-negative magnitude.
        hit_rate: Fraction of non-zero periods that were positive, or ``nan`` if the segment
            never held a position.
        turnover: Total absolute position change inside the segment.
    """

    n_bars: int
    n_trades: int
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    max_drawdown: float
    hit_rate: float
    turnover: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping, with ``nan`` rendered as ``None``."""
        return {
            "n_bars": self.n_bars,
            "n_trades": self.n_trades,
            "total_return": _jsonable(self.total_return),
            "annualized_return": _jsonable(self.annualized_return),
            "annualized_volatility": _jsonable(self.annualized_volatility),
            "sharpe": _jsonable(self.sharpe),
            "max_drawdown": _jsonable(self.max_drawdown),
            "hit_rate": _jsonable(self.hit_rate),
            "turnover": _jsonable(self.turnover),
        }


@dataclass(frozen=True, slots=True)
class CellResult:
    """Train and held-out statistics for one instrument in one walk-forward period.

    Attributes:
        instrument: Instrument name, as keyed in the panel.
        period: Zero-based walk-forward period index, in chronological order.
        split: The index ranges this cell was computed from.
        train: Statistics over the training segment.
        test: Statistics over the held-out segment.
    """

    instrument: str
    period: int
    split: Split
    train: SegmentMetrics
    test: SegmentMetrics

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping of the cell."""
        return {
            "instrument": self.instrument,
            "period": self.period,
            "train_range": [self.split.train_start, self.split.train_stop],
            "test_range": [self.split.test_start, self.split.test_stop],
            "gap": self.split.gap,
            "train": self.train.to_dict(),
            "test": self.test.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class RuleVerdict:
    """One survival rule, its verdict, and the numbers behind it.

    Attributes:
        name: Stable machine-readable rule identifier.
        passed: Whether the rule was satisfied.
        observed: The quantity the rule measured.
        threshold: The value ``observed`` had to reach.
        detail: A sentence stating what was measured and against what.
    """

    name: str
    passed: bool
    observed: float
    threshold: float
    detail: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping of the verdict."""
        return {
            "name": self.name,
            "passed": self.passed,
            "observed": _jsonable(self.observed),
            "threshold": _jsonable(self.threshold),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class EvaluationSettings:
    """The configuration a report was produced under, recorded so results are reproducible.

    Attributes:
        n_periods: Number of disjoint walk-forward blocks per instrument.
        train_fraction: Fraction of each block used for fitting.
        gap: Embargo bars discarded between train and test inside every block.
        periods_per_year: Annualisation factor used for every Sharpe and volatility figure.
        fees: Cost per unit of turnover.
        min_trades_per_parameter: Parameter-budget requirement.
        min_positive_test_fraction: Supermajority threshold for positive held-out Sharpe.
        min_sharpe_retention: Minimum ratio of mean held-out Sharpe to mean training Sharpe.
    """

    n_periods: int
    train_fraction: float
    gap: int
    periods_per_year: float
    fees: float
    min_trades_per_parameter: float
    min_positive_test_fraction: float
    min_sharpe_retention: float

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping of the settings."""
        return {
            "n_periods": self.n_periods,
            "train_fraction": self.train_fraction,
            "gap": self.gap,
            "periods_per_year": self.periods_per_year,
            "fees": self.fees,
            "min_trades_per_parameter": self.min_trades_per_parameter,
            "min_positive_test_fraction": self.min_positive_test_fraction,
            "min_sharpe_retention": self.min_sharpe_retention,
        }


@dataclass(frozen=True, slots=True)
class Report:
    """The complete outcome of an evaluation.

    Attributes:
        strategy_name: Class name of the evaluated strategy.
        n_parameters: Parameter count the strategy declared.
        settings: The configuration this report was produced under.
        cells: One :class:`CellResult` per instrument and period, in panel order.
        rules: One :class:`RuleVerdict` per survival rule, in the order they are applied.
        budget: The parameter-budget verdict, also surfaced as a rule.
        survived: ``True`` only if every rule passed.
    """

    strategy_name: str
    n_parameters: int
    settings: EvaluationSettings
    cells: tuple[CellResult, ...]
    rules: tuple[RuleVerdict, ...]
    budget: BudgetVerdict
    survived: bool
    instruments: tuple[str, ...] = ()

    @property
    def failed_rules(self) -> tuple[RuleVerdict, ...]:
        """The rules that did not pass, in application order."""
        return tuple(rule for rule in self.rules if not rule.passed)

    def rule(self, name: str) -> RuleVerdict:
        """Look up a single rule verdict by name.

        Args:
            name: The rule identifier, for example ``"sharpe_retention"``.

        Returns:
            The matching :class:`RuleVerdict`.

        Raises:
            KeyError: If no rule with that name exists in this report.
        """
        for verdict in self.rules:
            if verdict.name == name:
                return verdict
        available = ", ".join(verdict.name for verdict in self.rules)
        raise KeyError(f"no rule named {name!r}; available rules are: {available}")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping of the entire report."""
        return {
            "strategy": self.strategy_name,
            "n_parameters": self.n_parameters,
            "instruments": list(self.instruments),
            "settings": self.settings.to_dict(),
            "cells": [cell.to_dict() for cell in self.cells],
            "rules": [verdict.to_dict() for verdict in self.rules],
            "budget": {
                "n_trades": self.budget.n_trades,
                "n_parameters": self.budget.n_parameters,
                "trades_per_parameter": _jsonable(self.budget.trades_per_parameter),
                "required_trades": _jsonable(self.budget.required_trades),
                "min_trades_per_parameter": self.budget.min_trades_per_parameter,
                "passed": self.budget.passed,
            },
            "survived": self.survived,
        }

    def format_text(self) -> str:
        """Render the report as plain text suitable for a terminal or a log."""
        settings = self.settings
        lines = [
            f"holdout-first report: {self.strategy_name}",
            f"  declared parameters : {self.n_parameters}",
            f"  instruments         : {len(self.instruments)} ({', '.join(self.instruments)})",
            f"  walk-forward periods: {settings.n_periods}",
            f"  train fraction      : {settings.train_fraction:.2f} "
            f"(held out {1.0 - settings.train_fraction:.2f})",
            f"  embargo gap         : {settings.gap} bars",
            f"  periods per year    : {settings.periods_per_year:g}",
            f"  fees per turnover   : {settings.fees:g}",
            "",
            "  instrument     period  train_shp  test_shp  test_ret  test_mdd  test_trd",
        ]
        for cell in self.cells:
            lines.append(
                f"  {cell.instrument:<14} {cell.period:>6}  "
                f"{cell.train.sharpe:>9.3f}  {cell.test.sharpe:>8.3f}  "
                f"{cell.test.total_return:>8.3f}  {cell.test.max_drawdown:>8.3f}  "
                f"{cell.test.n_trades:>8}"
            )
        lines.append("")
        lines.append("  rules")
        for verdict in self.rules:
            mark = "pass" if verdict.passed else "FAIL"
            lines.append(f"  [{mark}] {verdict.name:<21} {verdict.detail}")
        lines.append("")
        if self.survived:
            lines.append("  verdict: SURVIVED")
        else:
            reasons = ", ".join(verdict.name for verdict in self.failed_rules)
            lines.append(f"  verdict: REJECTED (failing rules: {reasons})")
        return "\n".join(lines)


def _jsonable(value: float) -> float | None:
    """Map non-finite floats to ``None`` so the result survives ``json.dumps``."""
    number = float(value)
    return number if math.isfinite(number) else None


def _validate_panel(panel: Mapping[str, npt.ArrayLike]) -> dict[str, npt.NDArray[np.float64]]:
    if not isinstance(panel, Mapping):
        raise TypeError(
            f"panel must be a mapping of name to price array, got {type(panel).__name__}"
        )
    if not panel:
        raise ValueError("panel must contain at least one instrument")
    validated: dict[str, npt.NDArray[np.float64]] = {}
    lengths: set[int] = set()
    for name, prices in panel.items():
        if not isinstance(name, str):
            raise TypeError(f"panel keys must be str instrument names, got {type(name).__name__}")
        array = np.asarray(prices, dtype=np.float64)
        if array.ndim != 1:
            raise ValueError(f"panel[{name!r}] must be one-dimensional, got shape {array.shape}")
        if array.size:
            minimum_price = float(np.min(array))
            maximum_price = float(np.max(array))
            if not np.isfinite(minimum_price) or not np.isfinite(maximum_price):
                raise ValueError(f"panel[{name!r}] must be finite; found nan or inf")
            if minimum_price <= 0.0:
                worst = int(np.argmin(array))
                raise ValueError(
                    f"panel[{name!r}] must be strictly positive; index {worst} is {array[worst]!r}"
                )
        validated[name] = array
        lengths.add(array.size)
    if len(lengths) != 1:
        raise ValueError(
            "every instrument in the panel must have the same number of bars, got lengths "
            f"{sorted(lengths)}. Align the panel before evaluating, so that a walk-forward "
            "period means the same stretch of history for every instrument."
        )
    return validated


def evaluate(
    strategy: Strategy,
    panel: Mapping[str, npt.ArrayLike],
    *,
    n_periods: int = 3,
    train_fraction: float = 0.30,
    gap: int = 0,
    periods_per_year: float = 252.0,
    fees: float = 0.0,
    min_trades_per_parameter: float = 50.0,
    min_positive_test_fraction: float = 2.0 / 3.0,
    min_sharpe_retention: float = 0.5,
    allow_large_train: bool = False,
) -> Report:
    """Evaluate a strategy against the held-out-first survival rules.

    The strategy is run once per instrument on the full price series, which is the causal
    thing to do: at any bar it sees exactly the history a live system would have. The
    resulting position path is then sliced into the walk-forward train and test segments.
    The harness never fits anything; whatever fitting the configuration embodies happened
    before it was handed over.

    Args:
        strategy: An object satisfying :class:`holdout_first.protocol.Strategy`.
        panel: Mapping of instrument name to a one-dimensional array of strictly positive
            prices. All instruments must have the same number of bars, so that a period
            index refers to the same stretch of history everywhere.
        n_periods: Number of disjoint walk-forward blocks per instrument.
        train_fraction: Fraction of each block used for fitting. Defaults to 0.30, and
            values above 0.5 are rejected unless ``allow_large_train`` is set.
        gap: Embargo bars discarded between train and test inside each block.
        periods_per_year: Annualisation factor. 252 assumes daily bars.
        fees: Cost per unit of turnover, defaulting to zero so the baseline is cost-free.
        min_trades_per_parameter: Parameter-budget requirement, applied to trades observed
            in the training segments.
        min_positive_test_fraction: Fraction of instrument-period cells that must show a
            positive held-out Sharpe ratio. Defaults to two thirds.
        min_sharpe_retention: Minimum ratio of mean held-out Sharpe to mean training Sharpe.
            Defaults to 0.5.
        allow_large_train: Escape hatch permitting ``train_fraction > 0.5``.

    Returns:
        A frozen :class:`Report`. A strategy that peeks at the future does not raise here:
        the causality rule fails and the report says which instrument and which bar, so a
        single call can surface every problem at once. Use
        :func:`holdout_first.causality.assert_causal` directly if you want the exception.

    Raises:
        TypeError: If the panel is not a mapping, or the strategy does not implement the
            protocol.
        ValueError: If the panel is empty, ragged, or contains non-positive prices; if any
            threshold is outside its valid range; or if the series is too short to form
            ``n_periods`` blocks with usable train and test segments.
    """
    prices_by_name = _validate_panel(panel)
    if not hasattr(strategy, "positions"):
        raise TypeError(
            f"{type(strategy).__name__} does not implement Strategy: no 'positions' method"
        )
    declared = getattr(strategy, "n_parameters", None)
    if not isinstance(declared, int) or isinstance(declared, bool):
        raise TypeError(
            f"{type(strategy).__name__}.n_parameters must be an int, got "
            f"{type(declared).__name__}. Declare the number of values that were chosen by "
            "looking at data."
        )
    for label, value in (
        ("min_positive_test_fraction", min_positive_test_fraction),
        ("min_sharpe_retention", min_sharpe_retention),
    ):
        number = float(value)
        if not math.isfinite(number) or not 0.0 < number <= 1.0:
            raise ValueError(f"{label} must lie in (0, 1], got {value!r}")
    annualization = float(periods_per_year)
    if not math.isfinite(annualization) or annualization <= 0.0:
        raise ValueError(f"periods_per_year must be finite and positive, got {periods_per_year!r}")
    cost = float(fees)
    if not math.isfinite(cost) or cost < 0.0:
        raise ValueError(f"fees must be finite and non-negative, got {fees!r}")

    n_bars = next(iter(prices_by_name.values())).size
    splits = walk_forward_periods(
        n_bars,
        n_periods,
        train_fraction,
        gap=gap,
        allow_large_train=allow_large_train,
    )
    for index, split in enumerate(splits):
        if split.train_length < _MIN_SEGMENT_BARS or split.test_length < _MIN_SEGMENT_BARS:
            raise ValueError(
                f"period {index} yields a {split.train_length}-bar train segment and a "
                f"{split.test_length}-bar test segment; each needs at least "
                f"{_MIN_SEGMENT_BARS} bars. Supply more history, reduce n_periods, or "
                "reduce gap."
            )

    prefix_lengths = _default_prefix_lengths(n_bars)
    bounds = tuple(segment for split in splits for segment in (split.train_slice, split.test_slice))
    measured_return_segments = np.full(n_bars - 1, -1, dtype=np.intp)
    for segment_index, segment in enumerate(bounds):
        measured_return_segments[segment.start : segment.stop - 1] = segment_index

    evaluated: list[tuple[str, npt.NDArray[np.float64], npt.NDArray[np.float64]]] = []
    causality_failures: list[str] = []
    for name, prices in prices_by_name.items():
        try:
            positions = _assert_causal_at_prefixes(strategy, prices, prefix_lengths)
        except LookaheadError as exc:
            causality_failures.append(f"{name} at bar {exc.index}")
            positions = coerce_positions(
                strategy.positions(prices), prices.size, label=f"positions({name})"
            )
        returns = m.strategy_returns(positions, prices, fees=cost)
        invalid_returns = ~np.isfinite(returns)
        np.logical_or(invalid_returns, returns <= -1.0, out=invalid_returns)
        np.logical_and(
            invalid_returns,
            measured_return_segments >= 0,
            out=invalid_returns,
        )
        if np.any(invalid_returns):
            first_segment = int(np.min(measured_return_segments[invalid_returns]))
            segment = bounds[first_segment]
            m._as_returns(returns[segment.start : segment.stop - 1])
        evaluated.append((name, positions, returns))

    position_segments = tuple(
        positions[segment.start : segment.stop]
        for _, positions, _ in evaluated
        for segment in bounds
    )
    return_segments = tuple(
        returns[segment.start : segment.stop - 1]
        for _, _, returns in evaluated
        for segment in bounds
    )
    summaries = m._segment_summaries(position_segments, return_segments, annualization)
    cells: list[CellResult] = []
    summaries_per_instrument = len(bounds)
    for instrument_index, (name, _, _) in enumerate(evaluated):
        offset = instrument_index * summaries_per_instrument
        for period, split in enumerate(splits):
            train_summary = summaries[offset + period * 2]
            test_summary = summaries[offset + period * 2 + 1]
            cells.append(
                CellResult(
                    instrument=name,
                    period=period,
                    split=split,
                    train=SegmentMetrics(split.train_length, *train_summary),
                    test=SegmentMetrics(split.test_length, *test_summary),
                )
            )

    settings = EvaluationSettings(
        n_periods=n_periods,
        train_fraction=float(train_fraction),
        gap=gap,
        periods_per_year=annualization,
        fees=cost,
        min_trades_per_parameter=float(min_trades_per_parameter),
        min_positive_test_fraction=float(min_positive_test_fraction),
        min_sharpe_retention=float(min_sharpe_retention),
    )

    n_instruments = len(prices_by_name)
    causality_rule = RuleVerdict(
        name="causality",
        passed=not causality_failures,
        observed=float(n_instruments - len(causality_failures)),
        threshold=float(n_instruments),
        detail=(
            f"prefix-invariance held on all {n_instruments} instrument(s)"
            if not causality_failures
            else "look-ahead detected on " + "; ".join(causality_failures)
        ),
    )

    train_trades = sum(cell.train.n_trades for cell in cells)
    budget_verdict = parameter_budget(
        train_trades, declared, min_trades_per_parameter=float(min_trades_per_parameter)
    )
    budget_rule = RuleVerdict(
        name="parameter_budget",
        passed=budget_verdict.passed,
        observed=budget_verdict.trades_per_parameter,
        threshold=budget_verdict.min_trades_per_parameter,
        detail=budget_verdict.describe(),
    )

    test_sharpes = np.array([cell.test.sharpe for cell in cells], dtype=np.float64)
    train_sharpes = np.array([cell.train.sharpe for cell in cells], dtype=np.float64)
    n_positive = int(np.count_nonzero(test_sharpes > 0.0))
    positive_fraction = n_positive / test_sharpes.size
    positive_rule = RuleVerdict(
        name="test_sharpe_positive",
        passed=positive_fraction >= float(min_positive_test_fraction),
        observed=positive_fraction,
        threshold=float(min_positive_test_fraction),
        detail=(
            f"{n_positive}/{test_sharpes.size} held-out cells have a positive Sharpe "
            f"({positive_fraction:.3f}), need {float(min_positive_test_fraction):.3f}"
        ),
    )

    mean_train = float(np.mean(train_sharpes))
    mean_test = float(np.mean(test_sharpes))
    if mean_train <= 0.0:
        retention = 0.0
        retention_rule = RuleVerdict(
            name="sharpe_retention",
            passed=False,
            observed=retention,
            threshold=float(min_sharpe_retention),
            detail=(
                f"mean training Sharpe is {mean_train:.3f}, which is not positive: there is "
                "no in-sample performance to retain, so the configuration has nothing to "
                "degrade from"
            ),
        )
    else:
        retention = mean_test / mean_train
        retention_rule = RuleVerdict(
            name="sharpe_retention",
            passed=retention >= float(min_sharpe_retention),
            observed=retention,
            threshold=float(min_sharpe_retention),
            detail=(
                f"mean held-out Sharpe {mean_test:.3f} against mean training Sharpe "
                f"{mean_train:.3f} is a retention of {retention:.3f}, need "
                f"{float(min_sharpe_retention):.3f}"
            ),
        )

    rules = (causality_rule, budget_rule, positive_rule, retention_rule)
    return Report(
        strategy_name=type(strategy).__name__,
        n_parameters=declared,
        settings=settings,
        cells=tuple(cells),
        rules=rules,
        budget=budget_verdict,
        survived=all(verdict.passed for verdict in rules),
        instruments=tuple(prices_by_name),
    )
