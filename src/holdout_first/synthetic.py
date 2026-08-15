"""Synthetic price panels and three reference strategies used by the demo and the tests.

Nothing in this module is a trading recommendation and none of it has ever been traded. The
price process is a toy: a slowly mean-reverting drift buried in independent noise. It exists
so that the harness has something with a known, modest, genuinely persistent edge to detect,
and so that every example in this repository is reproducible from a seed with no data files
and no network.

Three reference strategies are provided, one for each thing the harness is meant to show:

* :class:`MomentumRule` - honest and parsimonious. One free parameter, a lookback. It has a
  real but unspectacular edge on this process, and it survives.
* :class:`OverfittedLookup` - a 128-entry lookup table fitted to the training segments. It
  is perfectly causal and it memorises the training noise, so it posts a strong training
  Sharpe and nothing out of sample. It is rejected.
* :class:`PeekingRule` - reads the next bar's price. It is rejected before any performance
  number is computed, by the prefix-invariance test.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np
import numpy.typing as npt

from .splits import Split

__all__ = [
    "MomentumRule",
    "OverfittedLookup",
    "PeekingRule",
    "make_panel",
]

_MAGNITUDE_EDGES: tuple[float, ...] = (-1.5, -0.75, -0.25, 0.0, 0.25, 0.75, 1.5)
_SIGN_LOOKBACK = 4
_VOLATILITY_WINDOW = 20


def make_panel(
    seed: int,
    *,
    n_instruments: int = 5,
    n_bars: int = 3000,
    drift_persistence: float = 0.985,
    drift_scale: float = 0.0010,
    noise_scale: float = 0.008,
    initial_price: float = 100.0,
) -> dict[str, npt.NDArray[np.float64]]:
    """Generate a reproducible panel of synthetic price series.

    Each instrument follows ``r_t = mu_t + noise_scale * z_t`` where ``mu_t`` is a
    stationary AR(1) process with persistence ``drift_persistence`` and unconditional
    standard deviation ``drift_scale``, and ``z_t`` is standard normal. Prices are the
    compounded series scaled by ``initial_price``, so they are strictly positive by
    construction.

    The AR(1) drift is what makes a slow momentum rule work here. It is persistent enough to
    be detectable over a few dozen bars and small enough relative to the noise that
    detection is imperfect, which is the regime where overfitting is tempting and the
    harness is worth having.

    Args:
        seed: Seed for ``numpy.random.default_rng``. The same seed always produces the same
            panel.
        n_instruments: Number of independent series. Must be at least 1.
        n_bars: Bars per series. Must be at least 2.
        drift_persistence: AR(1) coefficient in ``[0, 1)``.
        drift_scale: Unconditional standard deviation of the drift, per bar.
        noise_scale: Standard deviation of the independent noise, per bar.
        initial_price: Starting price for every series. Must be positive.

    Returns:
        A dict mapping ``"SYN_00"``, ``"SYN_01"``, ... to ``float64`` price arrays of length
        ``n_bars``, all with identical length as the harness requires.

    Raises:
        TypeError: If ``seed``, ``n_instruments``, or ``n_bars`` is not an ``int``.
        ValueError: If any count is too small, ``drift_persistence`` is outside ``[0, 1)``,
            or any scale or the initial price is not positive and finite.
    """
    for label, value in (("seed", seed), ("n_instruments", n_instruments), ("n_bars", n_bars)):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"{label} must be an int, got {type(value).__name__}")
    if n_instruments < 1:
        raise ValueError(f"n_instruments must be at least 1, got {n_instruments}")
    if n_bars < 2:
        raise ValueError(f"n_bars must be at least 2, got {n_bars}")
    if not 0.0 <= drift_persistence < 1.0:
        raise ValueError(f"drift_persistence must lie in [0, 1), got {drift_persistence!r}")
    for scale_label, scale_value in (
        ("drift_scale", drift_scale),
        ("noise_scale", noise_scale),
        ("initial_price", initial_price),
    ):
        number = float(scale_value)
        if not np.isfinite(number) or number <= 0.0:
            raise ValueError(f"{scale_label} must be finite and positive, got {scale_value!r}")

    rng = np.random.default_rng(seed)
    innovation_scale = float(drift_scale) * np.sqrt(1.0 - drift_persistence**2)
    panel: dict[str, npt.NDArray[np.float64]] = {}
    for index in range(n_instruments):
        drift_shocks = rng.standard_normal(n_bars) * innovation_scale
        drift = np.empty(n_bars, dtype=np.float64)
        drift[0] = rng.standard_normal() * float(drift_scale)
        for step in range(1, n_bars):
            drift[step] = drift_persistence * drift[step - 1] + drift_shocks[step]
        returns = drift + rng.standard_normal(n_bars) * float(noise_scale)
        panel[f"SYN_{index:02d}"] = np.asarray(
            float(initial_price) * np.cumprod(1.0 + returns), dtype=np.float64
        )
    return panel


def _as_prices(prices: npt.ArrayLike, minimum: int) -> npt.NDArray[np.float64]:
    array = np.asarray(prices, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"prices must be one-dimensional, got shape {array.shape}")
    if array.size < minimum:
        raise ValueError(f"prices must contain at least {minimum} bars, got {array.size}")
    minimum_price = float(np.min(array))
    maximum_price = float(np.max(array))
    if not np.isfinite(minimum_price) or not np.isfinite(maximum_price):
        raise ValueError("prices must be finite; found nan or inf")
    if minimum_price <= 0.0:
        worst = int(np.argmin(array))
        raise ValueError(f"prices must be strictly positive; index {worst} is {array[worst]!r}")
    return array


class MomentumRule:
    """Honest one-parameter momentum rule: hold the sign of the trailing ``lookback`` return.

    The position at bar ``t`` is ``sign(prices[t] / prices[t - lookback] - 1)``, and zero for
    the first ``lookback`` bars where the window is not yet full. There is nothing else: no
    volatility target, no filter, no stop, no regime switch. Each of those would be another
    parameter, and the parameter budget would want another 50 trades to pay for it.

    Attributes:
        lookback: Number of bars in the momentum window.
        n_parameters: Always 1.
    """

    def __init__(self, lookback: int = 20) -> None:
        """Initialise the rule.

        Args:
            lookback: Momentum window length in bars. Must be at least 1.

        Raises:
            TypeError: If ``lookback`` is not an ``int``.
            ValueError: If ``lookback`` is below 1.
        """
        if not isinstance(lookback, int) or isinstance(lookback, bool):
            raise TypeError(f"lookback must be an int, got {type(lookback).__name__}")
        if lookback < 1:
            raise ValueError(f"lookback must be at least 1, got {lookback}")
        self.lookback = lookback
        self.n_parameters = 1

    def positions(self, prices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the sign of the trailing momentum at every bar.

        Args:
            prices: One-dimensional array of strictly positive prices.

        Returns:
            An array of ``-1.0``, ``0.0``, or ``1.0``, one per bar.

        Raises:
            ValueError: If ``prices`` is not a valid one-dimensional positive series.
        """
        array = _as_prices(prices, minimum=1)
        out = np.zeros(array.size, dtype=np.float64)
        if array.size <= self.lookback:
            return out
        momentum = out[self.lookback :]
        np.divide(array[self.lookback :], array[: -self.lookback], out=momentum)
        np.subtract(momentum, 1.0, out=momentum)
        np.sign(momentum, out=momentum)
        return out


def _lookup_features(prices: npt.NDArray[np.float64]) -> npt.NDArray[np.intp]:
    """Map every bar to a bucket index using only information available at that bar.

    The bucket combines the signs of the last :data:`_SIGN_LOOKBACK` returns with a bucketed
    volatility-scaled magnitude of the most recent return. Both components read a trailing
    window only, so the mapping is prefix-invariant and passes the causality test. Bars
    inside the warm-up period map to ``-1``, meaning "no bucket".
    """
    n = prices.size
    buckets = np.full(n, -1, dtype=np.intp)
    warmup = max(_SIGN_LOOKBACK, _VOLATILITY_WINDOW) + 1
    if n <= warmup:
        return buckets
    returns = np.zeros(n, dtype=np.float64)
    returns[1:] = prices[1:] / prices[:-1] - 1.0

    signs = np.zeros(n, dtype=np.intp)
    for lag in range(_SIGN_LOOKBACK):
        signs = signs * 2 + (np.roll(returns, lag) > 0.0).astype(np.intp)
    signs[:warmup] = 0

    windows = np.asarray(
        np.lib.stride_tricks.sliding_window_view(returns, _VOLATILITY_WINDOW),
        dtype=np.float64,
    )
    volatility = np.zeros(n, dtype=np.float64)
    volatility[_VOLATILITY_WINDOW:] = np.std(windows[:-1], axis=1, ddof=1)
    scaled = np.zeros(n, dtype=np.float64)
    active = volatility > 0.0
    scaled[active] = returns[active] / volatility[active]
    magnitude = np.asarray(
        np.searchsorted(np.asarray(_MAGNITUDE_EDGES), scaled, side="right"), dtype=np.intp
    )

    buckets[warmup:] = signs[warmup:] * (len(_MAGNITUDE_EDGES) + 1) + magnitude[warmup:]
    return buckets


class OverfittedLookup:
    """A 128-entry lookup table fitted to the training segments. Causal, and still worthless.

    Each bar is mapped to one of ``2 ** 4 * 8 = 128`` buckets by the signs of its last four
    returns and a bucketed volatility-scaled magnitude of the latest return. The table stores
    the sign of the mean next-bar return observed in each bucket across the training
    segments, and the strategy simply looks the bucket up.

    This is the failure mode the library exists to catch. Nothing about it is dishonest: the
    fit uses training data only, the positions are strictly causal, and the training Sharpe
    is genuinely high. It is also entirely a description of the training noise, because 128
    free parameters divided into a few thousand training bars leaves a few dozen
    observations per parameter. The parameter budget says so before the held-out numbers do,
    and the held-out numbers then say it again.

    Attributes:
        table: One signed position per bucket.
        n_parameters: Number of buckets, that is 128.
    """

    n_buckets: int = (2**_SIGN_LOOKBACK) * (len(_MAGNITUDE_EDGES) + 1)

    def __init__(self, table: npt.ArrayLike) -> None:
        """Initialise from a fitted table.

        Args:
            table: A sequence of :attr:`n_buckets` positions in ``[-1, 1]``.

        Raises:
            ValueError: If the table is the wrong shape or leaves ``[-1, 1]``.
        """
        array = np.asarray(table, dtype=np.float64)
        if array.shape != (self.n_buckets,):
            raise ValueError(f"table must have shape ({self.n_buckets},), got {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError("table must be finite; found nan or inf")
        if np.any(np.abs(array) > 1.0):
            raise ValueError("table entries must lie in [-1, 1]")
        self.table = array
        self.n_parameters = int(self.n_buckets)

    @classmethod
    def fit(
        cls,
        panel: Mapping[str, npt.ArrayLike],
        splits: Iterable[Split],
    ) -> OverfittedLookup:
        """Fit the table on the training segments of every instrument.

        Args:
            panel: Mapping of instrument name to price array.
            splits: The walk-forward splits whose training ranges define the fitting data.

        Returns:
            A fitted :class:`OverfittedLookup`.

        Raises:
            ValueError: If the panel is empty, a price series is invalid, or the training
                ranges contain no usable bars.
        """
        split_list = list(splits)
        if not split_list:
            raise ValueError("splits must contain at least one Split")
        if not panel:
            raise ValueError("panel must contain at least one instrument")

        totals = np.zeros(cls.n_buckets, dtype=np.float64)
        counts = np.zeros(cls.n_buckets, dtype=np.int64)
        for prices in panel.values():
            array = _as_prices(prices, minimum=2)
            buckets = _lookup_features(array)
            forward = np.empty(array.size, dtype=np.float64)
            forward[:-1] = array[1:] / array[:-1] - 1.0
            forward[-1] = 0.0
            for split in split_list:
                stop = min(split.train_stop, array.size - 1)
                if stop <= split.train_start:
                    continue
                window = buckets[split.train_start : stop]
                usable = window >= 0
                selected = window[usable]
                np.add.at(totals, selected, forward[split.train_start : stop][usable])
                np.add.at(counts, selected, 1)
        if not counts.any():
            raise ValueError(
                "no training bars produced a bucket; the training ranges are shorter than "
                "the feature warm-up period"
            )
        means = np.zeros(cls.n_buckets, dtype=np.float64)
        seen = counts > 0
        means[seen] = totals[seen] / counts[seen]
        return cls(np.sign(means))

    def positions(self, prices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the fitted position for every bar's bucket.

        Args:
            prices: One-dimensional array of strictly positive prices.

        Returns:
            An array of positions in ``[-1, 1]``, one per bar, zero during warm-up.

        Raises:
            ValueError: If ``prices`` is not a valid one-dimensional positive series.
        """
        array = _as_prices(prices, minimum=1)
        buckets = _lookup_features(array)
        out = np.zeros(array.size, dtype=np.float64)
        active = buckets >= 0
        out[active] = self.table[buckets[active]]
        return out


class PeekingRule:
    """A deliberately non-causal rule that reads the next bar. Used to demonstrate detection.

    The position at bar ``t`` is the sign of the return from ``t`` to ``t + 1``, which is
    perfect foresight one bar ahead. The final bar has no successor, so it is held flat.

    That final-bar detail is exactly what gives the rule away. Truncate the series at ``k``
    bars and the position at ``k - 1`` is zero; supply the whole series and the same bar
    carries a signed position. :func:`holdout_first.causality.assert_causal` compares the two
    runs and reports index ``k - 1``.

    Attributes:
        n_parameters: Always 0. The rule has nothing to fit, which is a useful reminder that
            a low parameter count is not by itself a virtue.
    """

    def __init__(self) -> None:
        """Initialise the rule. It has no configuration."""
        self.n_parameters = 0

    def positions(self, prices: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the sign of each bar's forward return, which is not knowable at that bar.

        Args:
            prices: One-dimensional array of strictly positive prices.

        Returns:
            An array of ``-1.0``, ``0.0``, or ``1.0``, one per bar, with the last bar flat.

        Raises:
            ValueError: If ``prices`` is not a valid one-dimensional positive series.
        """
        array = _as_prices(prices, minimum=2)
        out = np.zeros(array.size, dtype=np.float64)
        out[:-1] = np.sign(array[1:] / array[:-1] - 1.0)
        return out
