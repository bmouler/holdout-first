"""Command-line entry point. ``holdout-first demo`` runs the full three-strategy comparison.

The demo is the fastest honest answer to "what does this library actually do". It builds a
synthetic panel from a seed, puts an honest one-parameter rule and a 128-parameter fitted
lookup table through the same harness, and then runs a rule that reads the next bar straight
into the causality check. The exit code is 1 unless all three behave as they should, so the
demo doubles as a smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from typing import cast

from .causality import LookaheadError, assert_causal
from .harness import Report, evaluate
from .splits import walk_forward_periods
from .synthetic import MomentumRule, OverfittedLookup, PeekingRule, make_panel

__all__ = ["build_parser", "main", "run_demo"]

_N_PERIODS = 3
_TRAIN_FRACTION = 0.30
_GAP = 5
_LOOKBACK = 20


def run_demo(seed: int) -> tuple[Report, Report, LookaheadError | None]:
    """Run the honest, overfitted, and peeking strategies against one synthetic panel.

    Args:
        seed: Seed passed to :func:`holdout_first.synthetic.make_panel`.

    Returns:
        A tuple of the honest strategy's report, the overfitted strategy's report, and the
        :class:`holdout_first.causality.LookaheadError` raised by the peeking rule, or
        ``None`` if the peeking rule somehow escaped detection.

    Raises:
        TypeError: If ``seed`` is not an ``int``.
    """
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")

    panel = make_panel(seed)
    n_bars = len(next(iter(panel.values())))
    splits = walk_forward_periods(n_bars, _N_PERIODS, _TRAIN_FRACTION, gap=_GAP)

    honest = evaluate(
        MomentumRule(lookback=_LOOKBACK),
        panel,
        n_periods=_N_PERIODS,
        train_fraction=_TRAIN_FRACTION,
        gap=_GAP,
        fees=0.0,
    )
    overfitted = evaluate(
        OverfittedLookup.fit(panel, splits),
        panel,
        n_periods=_N_PERIODS,
        train_fraction=_TRAIN_FRACTION,
        gap=_GAP,
        fees=0.0,
    )

    caught: LookaheadError | None = None
    try:
        assert_causal(PeekingRule(), next(iter(panel.values())))
    except LookaheadError as exc:
        caught = exc
    return honest, overfitted, caught


def _demo_succeeded(honest: Report, overfitted: Report, caught: LookaheadError | None) -> bool:
    return honest.survived and not overfitted.survived and caught is not None


def _render_text(
    seed: int, honest: Report, overfitted: Report, caught: LookaheadError | None
) -> str:
    lines = [
        f"holdout-first demo (seed {seed})",
        "",
        "1. honest strategy: one free parameter, a momentum lookback",
        "",
        honest.format_text(),
        "",
        "2. overfitted strategy: a 128-entry lookup table fitted to the training segments",
        "",
        overfitted.format_text(),
        "",
        "3. non-causal strategy: reads the next bar",
        "",
    ]
    if caught is None:
        lines.append("  no LookaheadError was raised, which means the check is broken")
    else:
        lines.append(f"  LookaheadError raised: {caught}")
        lines.append(
            f"  first divergent bar: {caught.index} "
            f"(prefix length {caught.prefix_length}, "
            f"prefix said {caught.prefix_value:g}, full series said {caught.full_value:g})"
        )
    lines.append("")
    if _demo_succeeded(honest, overfitted, caught):
        lines.append(
            "outcome: the parsimonious rule survived, the overfitted rule was rejected, and "
            "the peeking rule never reached a performance number."
        )
    else:
        lines.append("outcome: the demo did not behave as documented.")
    return "\n".join(lines)


def _render_json(
    seed: int, honest: Report, overfitted: Report, caught: LookaheadError | None
) -> str:
    payload: dict[str, object] = {
        "seed": seed,
        "honest": honest.to_dict(),
        "overfitted": overfitted.to_dict(),
        "peeking": (
            None
            if caught is None
            else {
                "error": "LookaheadError",
                "index": caught.index,
                "prefix_length": caught.prefix_length,
                "prefix_value": caught.prefix_value,
                "full_value": caught.full_value,
            }
        ),
        "demo_succeeded": _demo_succeeded(honest, overfitted, caught),
    }
    return json.dumps(payload, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``holdout-first`` command.

    Returns:
        A parser with a single ``demo`` subcommand.
    """
    parser = argparse.ArgumentParser(
        prog="holdout-first",
        description=(
            "Validation harness that fits on a small slice and demands survival on the "
            "large held-out remainder."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser(
        "demo",
        help="run the honest, overfitted, and non-causal reference strategies",
        description=(
            "Build a synthetic panel from a seed and run three reference strategies through "
            "the harness. Exits 0 only if the honest strategy survives, the overfitted one "
            "is rejected, and the non-causal one raises LookaheadError."
        ),
    )
    demo.add_argument("--seed", type=int, default=11, help="panel seed (default: 11)")
    demo.add_argument(
        "--json", action="store_true", help="emit the full report as JSON instead of text"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface.

    Args:
        argv: Argument vector excluding the program name. Defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` if the demo behaved as documented, ``1`` otherwise.
    """
    args = build_parser().parse_args(argv)
    seed = cast(int, args.seed)
    use_json = cast(bool, args.json)
    honest, overfitted, caught = run_demo(seed)
    if use_json:
        print(_render_json(seed, honest, overfitted, caught))
    else:
        print(_render_text(seed, honest, overfitted, caught))
    return 0 if _demo_succeeded(honest, overfitted, caught) else 1


if __name__ == "__main__":
    sys.exit(main())
