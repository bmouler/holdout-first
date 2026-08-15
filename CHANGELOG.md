# Changelog

## [Unreleased]

- Reduced redundant panel, causality, strategy-return, and metric work without changing report output.
- Reused metric scratch arrays and balanced large batches across caller-assisted worker tasks.
- Preserved scalar annualized-return rounding across platforms while vectorizing the remaining metrics.
- Added an exact-checksum end-to-end benchmark; three paired trials measured 3.154x–3.168x speedups.


## [1.0.0] - 2026-08-12

First stable release.

- Validation harness that fits on a small slice and demands survival on the large held-out remainder.
- Added a deterministic property-based suite for metric, split, and causality invariants.
- Documented mutation testing with 1,430 of 1,650 mutants killed (86.67%) and all 220 remaining mutants reviewed as equivalent.
- Adopted strict mypy checking across the package.
- Expanded CI to Linux and macOS on Python 3.11–3.13.
