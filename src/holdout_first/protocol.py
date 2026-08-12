"""The user-facing contract that every strategy passed to the harness must satisfy.

``holdout-first`` is not a backtesting engine. It never builds signals, never matches
orders, and never fetches data. It wraps a callable that you supply and subjects it to a
validation regime that is deliberately hostile to overfitting.

The single hard requirement on that callable is *causality*: the position produced for bar
``t`` may depend only on prices at bars ``0 .. t``. This is not an honour-system request.
:func:`holdout_first.causality.assert_causal` re-runs your strategy on truncated prefixes of
the same series and compares the results bar by bar. A strategy that peeks at the future
produces different early positions once later data is appended, and the harness raises
:class:`holdout_first.causality.LookaheadError` naming the first index that moved.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

__all__ = ["Strategy"]


@runtime_checkable
class Strategy(Protocol):
    """A causal mapping from a price series to a target position per bar.

    Attributes:
        n_parameters: The number of free parameters the configuration spends. This is the
            honest count of every value that was chosen by looking at data: lookbacks,
            thresholds, lookup-table entries, blend weights. It feeds
            :func:`holdout_first.budget.parameter_budget`, which requires a minimum number
            of observed trades per parameter. Understating it defeats the check, and the
            only person deceived is you.
    """

    n_parameters: int

    def positions(
        self, prices: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64] | Sequence[float]:
        """Return the target position for every bar in ``prices``.

        Args:
            prices: A one-dimensional array of strictly positive prices, ordered oldest to
                newest.

        Returns:
            A sequence of the same length as ``prices``. Element ``t`` is the target
            position held from the close of bar ``t`` to the close of bar ``t + 1``,
            expressed as a signed fraction of capital in ``[-1, 1]``. The final element is
            never applied to a return, because there is no bar after the last one; it is
            still validated for range and finiteness.

        Raises:
            The implementation may raise anything it likes. The harness does not swallow
            strategy exceptions, on the principle that a strategy that cannot run is a
            louder failure than a strategy that quietly returns zeros.

        Note:
            Element ``t`` MUST be computable from ``prices[: t + 1]`` alone. Vectorised
            implementations are welcome, but a centred rolling window, a full-sample
            normalisation, a ``numpy.gradient`` call, or a backward-filled series will all
            leak the future and will be caught.
        """
        ...
