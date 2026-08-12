"""Shared fixtures. Everything is generated from a fixed seed, so nothing here is random."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest

from holdout_first.splits import walk_forward_periods
from holdout_first.synthetic import OverfittedLookup, make_panel

DEMO_SEED = 11
N_PERIODS = 3
TRAIN_FRACTION = 0.30
GAP = 5


@pytest.fixture(scope="session")
def panel() -> dict[str, npt.NDArray[np.float64]]:
    """The same synthetic panel the demo uses."""
    return make_panel(DEMO_SEED)


@pytest.fixture(scope="session")
def overfitted(panel: dict[str, npt.NDArray[np.float64]]) -> OverfittedLookup:
    """A lookup table fitted to the training segments of that panel."""
    n_bars = next(iter(panel.values())).size
    splits = walk_forward_periods(n_bars, N_PERIODS, TRAIN_FRACTION, gap=GAP)
    return OverfittedLookup.fit(panel, splits)
