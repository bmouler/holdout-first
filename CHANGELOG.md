# Changelog

## [Unreleased]

- Reduced redundant panel and causality validation and shared metric work across aligned walk-forward segments without changing report output.
- Preserved scalar annualized-return rounding across platforms while batching the remaining metric work.
- Added a deterministic end-to-end `evaluate(...).to_dict()` benchmark with exact output checksums.


## [1.0.0] - 2026-08-12

First stable release.

- Validation harness that fits on a small slice and demands survival on the large held-out remainder.
- Added a deterministic property-based suite for metric, split, and causality invariants.
- Documented mutation testing with 1,430 of 1,650 mutants killed (86.67%) and all 220 remaining mutants reviewed as equivalent.
- Adopted strict mypy checking across the package.
- Expanded CI to Linux and macOS on Python 3.11–3.13.
