"""Higuchi Fractal Dimension (HFD) estimation for time series.

Implements Higuchi's algorithm (Higuchi 1988):
- Multi-scale curve length estimation across delay factors k = 1 ... k_max
- Linear log-log fit: ln(<L(k)>) ~ D * ln(1/k)
- Stability sweep helper for data-driven k_max selection
"""

from typing import Any
import numpy as np

from criticalitylens.measures.base import MeasureResult, register


def higuchi_fd(signal: np.ndarray, kmax: int = 10) -> tuple[float, float]:
    """Compute Higuchi Fractal Dimension of a 1D time series.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.
    kmax : int, default 10
        Maximum interval delay factor k.

    Returns
    -------
    d : float
        Higuchi fractal dimension.
    r2 : float
        R^2 of log-log linear fit.
    """
    n = len(signal)
    if n < kmax * 2 or kmax < 2:
        return float("nan"), float("nan")

    lengths = np.zeros(kmax)

    for k in range(1, kmax + 1):
        l_mk = np.zeros(k)
        for m in range(k):
            # Indices for sub-series
            idx = np.arange(m, n, k)
            n_pts = len(idx)
            if n_pts < 2:
                continue

            diffs = np.abs(np.diff(signal[idx]))
            # Normalization factor: (N - 1) / (floor((N - m) / k) * k^2)
            norm = (n - 1) / (int(np.floor((n - 1 - m) / k)) * (k**2))
            l_mk[m] = np.sum(diffs) * norm

        lengths[k - 1] = np.mean(l_mk)

    valid = lengths > 0
    if np.sum(valid) < 3:
        return float("nan"), float("nan")

    k_vals = np.arange(1, kmax + 1)[valid]
    log_inv_k = np.log(1.0 / k_vals)
    log_l = np.log(lengths[valid])

    poly = np.polyfit(log_inv_k, log_l, 1)
    d = float(poly[0])

    fit_vals = np.polyval(poly, log_inv_k)
    ss_tot = np.sum((log_l - np.mean(log_l)) ** 2)
    ss_res = np.sum((log_l - fit_vals) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return d, r2


def find_optimal_kmax(signal: np.ndarray, k_range: range = range(4, 30)) -> int:
    """Perform a stability sweep to select the optimal kmax parameter.

    Identifies the scale where the estimated fractal dimension stabilizes
    (plateau region).

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.
    k_range : range, default range(4, 30)
        Candidate kmax values.

    Returns
    -------
    int
        Selected kmax value.
    """
    dims = [higuchi_fd(signal, kmax=k)[0] for k in k_range]
    valid_dims = np.array(dims)
    if len(valid_dims) < 3 or np.all(np.isnan(valid_dims)):
        return 10

    # Pick kmax where difference with subsequent k stabilizes
    diffs = np.abs(np.diff(valid_dims))
    min_diff_idx = int(np.nanargmin(diffs))
    return list(k_range)[min_diff_idx]


@register("higuchi")
@register("higuchi_fd")
class HiguchiMeasure:
    """Higuchi Fractal Dimension measure."""

    @property
    def name(self) -> str:
        return "higuchi_fd"

    def compute(
        self,
        signal: np.ndarray,
        fs: float,
        config: dict[str, Any] | None = None,
    ) -> MeasureResult:
        cfg = config or {}
        kmax = int(cfg.get("kmax", 10))

        if len(signal) < kmax * 4:
            return MeasureResult(
                value=float("nan"),
                status="excluded",
                reason="duration_too_short",
                meta={"n_samples": len(signal)},
            )

        d, r2 = higuchi_fd(signal, kmax=kmax)
        if np.isnan(d):
            return MeasureResult(
                value=float("nan"),
                status="failed",
                reason="hfd_estimation_failure",
                meta={},
            )

        return MeasureResult(
            value=d,
            status="ok",
            reason="",
            meta={"kmax": kmax, "r2": r2},
        )
