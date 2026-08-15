"""Deterministic end-to-end benchmark for the documented holdout evaluation pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from typing import Any

from holdout_first import evaluate
from holdout_first.synthetic import MomentumRule, make_panel

EXPECTED_CHECKSUM = "b97c08eeec26c31a91bae5d15f991fc52c364066e30ebce4c9d0b3fc4da1c379"
SEED = 1729
N_INSTRUMENTS = 24
N_BARS = 20_000
N_PERIODS = 3
GAP = 20
LOOKBACK = 20


def _checksum(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.samples < 11:
        parser.error("--samples must be at least 11")
    if args.warmups < 1:
        parser.error("--warmups must be at least 1")

    # Preparing observations and configuration is deliberately outside the timed region.
    panel = make_panel(seed=SEED, n_instruments=N_INSTRUMENTS, n_bars=N_BARS)
    strategy = MomentumRule(lookback=LOOKBACK)

    def materialize() -> dict[str, Any]:
        return evaluate(
            strategy,
            panel,
            n_periods=N_PERIODS,
            gap=GAP,
            periods_per_year=252.0,
            fees=0.0,
        ).to_dict()

    reference = materialize()
    checksum = _checksum(reference)
    if checksum != EXPECTED_CHECKSUM:
        raise AssertionError(
            f"public result checksum changed: expected {EXPECTED_CHECKSUM}, got {checksum}"
        )
    for _ in range(args.warmups):
        if materialize() != reference:
            raise AssertionError("evaluation result changed during warmup")

    samples: list[float] = []
    for _ in range(args.samples):
        started = time.perf_counter_ns()
        payload = materialize()
        elapsed = (time.perf_counter_ns() - started) / 1_000_000_000.0
        if payload != reference:
            raise AssertionError("evaluation result changed between timed samples")
        samples.append(elapsed)

    result = {
        "benchmark": "holdout_evaluate",
        "checksum": checksum,
        "dimensions": {
            "bars_per_instrument": N_BARS,
            "gap": GAP,
            "instruments": N_INSTRUMENTS,
            "lookback": LOOKBACK,
            "periods": N_PERIODS,
            "result_cells": N_INSTRUMENTS * N_PERIODS,
        },
        "max_seconds": max(samples),
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "samples": args.samples,
        "warmups": args.warmups,
    }
    if args.json:
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    else:
        print(json.dumps(result, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
