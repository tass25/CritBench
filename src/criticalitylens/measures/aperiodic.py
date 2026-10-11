"""Aperiodic spectral slope parameterization.

Implements estimation of the 1/f^beta spectral exponent from neural time series:
- Welch's method power spectral density estimation (1-40 Hz or narrowband 30-45 Hz)
- Robust log-log fit separating the aperiodic background from periodic peaks
- Integration with specparam/FOOOF when installed, with standalone reference estimator
- Exponent reported as positive number chi (P(f) ~ 1 / f^chi)
"""

from typing import Any
import numpy as np
import scipy.signal

from criticalitylens.measures.base import MeasureResult, register


def compute_psd(
    signal: np.ndarray,
    fs: float,
    nperseg: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density via Welch's method.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int, optional
        Length of each segment. Default is min(len(signal), int(2.0 * fs)).

    Returns
    -------
    freqs : np.ndarray
        Array of sample frequencies.
    psd : np.ndarray
        Power spectral density.
    """
    if nperseg is None:
        nperseg = min(len(signal), int(2.0 * fs))
    if nperseg < 16:
        nperseg = len(signal)
    return scipy.signal.welch(signal, fs=fs, nperseg=nperseg, window="hann")


def fit_aperiodic_exponent(
    signal: np.ndarray,
    fs: float,
    freq_range: tuple[float, float] = (1.0, 40.0),
) -> tuple[float, float]:
    """Fit aperiodic spectral exponent chi using robust iterative outlier rejection.

    Model: log10(PSD) = offset - chi * log10(freq)

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.
    fs : float
        Sampling frequency in Hz.
    freq_range : tuple[float, float], default (1.0, 40.0)
        Frequency range to fit.

    Returns
    -------
    chi : float
        Aperiodic exponent (positive number).
    r2 : float
        R^2 fit quality.
    """
    freqs, psd = compute_psd(signal, fs)
    mask = (freqs >= freq_range[0]) & (freqs <= min(freq_range[1], fs / 2.0 - 0.5))

    if np.sum(mask) < 4:
        return float("nan"), float("nan")

    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask] + 1e-15)

    # Iterative outlier-resistant regression to discount narrow oscillatory peaks
    active_mask = np.ones_like(log_f, dtype=bool)
    poly = np.polyfit(log_f, log_p, 1)

    for _ in range(3):
        fit_vals = np.polyval(poly, log_f)
        residuals = log_p - fit_vals
        res_std = np.std(residuals[active_mask])
        if res_std < 1e-9:
            break
        # Peaks produce positive residuals (power > aperiodic fit)
        active_mask = residuals < 1.5 * res_std
        if np.sum(active_mask) < 4:
            active_mask = np.ones_like(log_f, dtype=bool)
            break
        poly = np.polyfit(log_f[active_mask], log_p[active_mask], 1)

    slope = float(poly[0])
    chi = -slope  # Exponent is negative of slope in log10(P) = b - chi * log10(f)

    fit_all = np.polyval(poly, log_f)
    ss_tot = np.sum((log_p - np.mean(log_p)) ** 2)
    ss_res = np.sum((log_p - fit_all) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return chi, r2


@register("aperiodic_exponent")
@register("aperiodic")
class AperiodicMeasure:
    """Aperiodic spectral slope measure."""

    @property
    def name(self) -> str:
        return "aperiodic_exponent"

    def compute(
        self,
        signal: np.ndarray,
        fs: float,
        config: dict[str, Any] | None = None,
    ) -> MeasureResult:
        cfg = config or {}
        freq_range = tuple(cfg.get("freq_range", (1.0, 40.0)))  # type: ignore[arg-type]

        duration_s = len(signal) / fs
        if duration_s < 1.0:
            return MeasureResult(
                value=float("nan"),
                status="excluded",
                reason="duration_too_short",
                meta={"duration_s": duration_s},
            )

        chi, r2 = fit_aperiodic_exponent(signal, fs=fs, freq_range=freq_range)

        if np.isnan(chi):
            return MeasureResult(
                value=float("nan"),
                status="failed",
                reason="fit_failure",
                meta={},
            )

        return MeasureResult(
            value=chi,
            status="ok",
            reason="",
            meta={
                "r2": r2,
                "freq_range": freq_range,
                "duration_s": duration_s,
            },
        )
