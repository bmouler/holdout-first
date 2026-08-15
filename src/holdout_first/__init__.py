"""holdout-first: fit on a small slice, prove it on the large remainder.

The public surface is small on purpose. Wrap your rule in an object satisfying
:class:`~holdout_first.protocol.Strategy`, hand it to :func:`~holdout_first.harness.evaluate`
with a panel of price arrays, and read the report.
"""

from __future__ import annotations

from .budget import BudgetVerdict, parameter_budget
from .causality import LookaheadError, assert_causal, coerce_positions
from .harness import (
    CellResult,
    EvaluationSettings,
    Report,
    RuleVerdict,
    SegmentMetrics,
    evaluate,
)
from .metrics import (
    annualized_return,
    annualized_volatility,
    equity_curve,
    hit_rate,
    max_drawdown,
    sharpe,
    strategy_returns,
    total_return,
    trade_count,
    turnover,
)
from .protocol import Strategy
from .splits import Split, fraction_split, walk_forward_periods

__version__ = "1.0.1"

__all__ = [
    "BudgetVerdict",
    "CellResult",
    "EvaluationSettings",
    "LookaheadError",
    "Report",
    "RuleVerdict",
    "SegmentMetrics",
    "Split",
    "Strategy",
    "__version__",
    "annualized_return",
    "annualized_volatility",
    "assert_causal",
    "coerce_positions",
    "equity_curve",
    "evaluate",
    "fraction_split",
    "hit_rate",
    "max_drawdown",
    "parameter_budget",
    "sharpe",
    "strategy_returns",
    "total_return",
    "trade_count",
    "turnover",
    "walk_forward_periods",
]
