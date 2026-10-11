"""Detrended fluctuation analysis (DFA) of amplitude envelopes.

Implements DFA according to the locked scientific specifications:
- Zero-phase FIR band-pass in theta (4-8 Hz), alpha (8-13 Hz), and beta (13-30 Hz) separately
- Conditioning on an identified spectral peak above the aperiodic fit
- Hilbert amplitude envelope extraction
- Integrated fluctuation computation across log-spaced windows (1s to min(25s, length/10))
- Undefined without peak: returns NaN with reason 'no_peak'
"""

from typing import Any
import numpy as np
import scipy.signal

from criticalitylens.measures.base import MeasureResult, register


BANDS: dict[str, tuple[float, float]] = {
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}


def raw_dfa(
    signal: np.ndarray,
    fs: float,
    window_min_s: float = 1.0,
    window_max_s: float | None = None,
    n_windows: int = 20,
) -> tuple[float, np.ndarray, np.ndarray, float]:
    """Compute Detrended Fluctuation Analysis exponent on a 1D time series.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.
    fs : float
        Sampling frequency in Hz.
    window_min_s : float, default 1.0
        Minimum window size in seconds.
    window_max_s : float, optional
        Maximum window size in seconds. If None, defaults to min(25.0, len(signal)/(10*fs)).
    n_windows : int, default 20
        Number of log-spaced window scales.

    Returns
    -------
    alpha : float
        DFA scaling exponent (slope of log(F) vs log(scale)).
    scales_s : np.ndarray
        Window scales in seconds.
    fluctuations : np.ndarray
        Root-mean-square fluctuations F(s).
    r2 : float
        Coefficient of determination of the log-log linear fit.
    """
    n_samples = len(signal)
    if n_samples < 32:
        return float("nan"), np.array([]), np.array([]), float("nan")

    if window_max_s is None:
        window_max_s = min(25.0, n_samples / (10.0 * fs))

    min_scale = max(4, int(np.round(window_min_s * fs)))
    max_scale = max(min_scale + 4, int(np.round(window_max_s * fs)))

    if max_scale <= min_scale:
        return float("nan"), np.array([]), np.array([]), float("nan")

    # Generate log-spaced integer scales
    scales = np.unique(
        np.round(np.geomspace(min_scale, max_scale, num=n_windows)).astype(int)
    )

    # 1. Mean-centered cumulative sum (profile)
    mean_val = np.mean(signal)
    profile = np.cumsum(signal - mean_val)

    fluctuations = []
    valid_scales = []

    for s in scales:
        n_segments = n_samples // s
        if n_segments < 1:
            continue

        # Forward and backward segments to use full series
        forward = profile[: n_segments * s].reshape(n_segments, s)
        backward = profile[n_samples - n_segments * s :].reshape(n_segments, s)
        segments = np.vstack([forward, backward])

        t = np.arange(s)
        # Linear detrend per segment: fit y = a*t + b
        # Vectorized polynomial fit for degree 1
        t_mean = np.mean(t)
        t_centered = t - t_mean
        denom = np.sum(t_centered**2)

        slopes = np.dot(segments, t_centered) / denom
        intercepts = np.mean(segments, axis=1) - slopes * t_mean

        trends = slopes[:, None] * t[None, :] + intercepts[:, None]
        residuals = segments - trends
        f_s = np.sqrt(np.mean(residuals**2))

        fluctuations.append(f_s)
        valid_scales.append(s)

    if len(valid_scales) < 3:
        return float("nan"), np.array([]), np.array([]), float("nan")

    scales_arr = np.array(valid_scales, dtype=float)
    flucts_arr = np.array(fluctuations, dtype=float)

    # Log10-log10 linear regression
    log_scales = np.log10(scales_arr)
    log_flucts = np.log10(flucts_arr)

    poly = np.polyfit(log_scales, log_flucts, 1)
    alpha = float(poly[0])

    # Compute R^2
    fit_vals = np.polyval(poly, log_scales)
    ss_tot = np.sum((log_flucts - np.mean(log_flucts)) ** 2)
    ss_res = np.sum((log_flucts - fit_vals) ** 2)
    r2 = float(1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0

    return alpha, scales_arr / fs, flucts_arr, r2


def has_spectral_peak(
    signal: np.ndarray,
    fs: float,
    band: tuple[float, float],
    threshold_sd: float = 1.0,
) -> bool:
    """Determine whether a prominent spectral peak exists above the aperiodic slope in a band.

    Parameters
    ----------
    signal : np.ndarray
        Signal time series.
    fs : float
        Sampling frequency in Hz.
    band : tuple[float, float]
        Frequency range (f_low, f_high).
    threshold_sd : float, default 1.0
        Threshold in standard deviations of spectral residuals to qualify as a peak.

    Returns
    -------
    bool
        True if an oscillatory peak exists in the band above threshold.
    """
    nperseg = min(len(signal), int(4.0 * fs))
    if nperseg < 16:
        return False

    freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=nperseg)
    # Fit broadband aperiodic baseline between 1 and 40 Hz
    mask = (freqs >= 1.0) & (freqs <= min(40.0, fs / 2.0 - 1.0))
    if np.sum(mask) < 5:
        return False

    log_f = np.log10(freqs[mask])
    log_p = np.log10(psd[mask] + 1e-12)

    poly = np.polyfit(log_f, log_p, 1)
    baseline = np.polyval(poly, log_f)
    residuals = log_p - baseline

    fit_freqs = freqs[mask]
    residual_std = float(np.std(residuals))
    if residual_std < 1e-9:
        return False

    # Height requirement combines both standard deviation threshold and minimum
    # log10 power height (0.25 in log10 power ~ 1.78x power ratio) to reject
    # minor Welch windowing ripples on pure white/pink noise.
    required_height = max(threshold_sd * residual_std, 0.25)
    peaks, _ = scipy.signal.find_peaks(
        residuals,
        height=required_height,
        prominence=residual_std,
    )
    if len(peaks) == 0:
        return False

    peak_freqs = fit_freqs[peaks]
    in_band = (peak_freqs >= band[0]) & (peak_freqs <= band[1])
    return bool(np.any(in_band))




@register("dfa")
class DFAMeasure:
    """Detrended Fluctuation Analysis of amplitude envelopes."""

    @property
    def name(self) -> str:
        return "dfa"

    def compute(
        self,
        signal: np.ndarray,
        fs: float,
        config: dict[str, Any] | None = None,
    ) -> MeasureResult:
        cfg = config or {}
        band_name = str(cfg.get("band", "alpha"))
        min_duration_s = float(cfg.get("min_duration_s", 180.0))
        peak_threshold_sd = float(cfg.get("peak_threshold_sd", 1.0))

        duration_s = len(signal) / fs
        if duration_s < min_duration_s:
            return MeasureResult(
                value=float("nan"),
                status="excluded",
                reason="duration_too_short",
                meta={"duration_s": duration_s, "min_duration_s": min_duration_s},
            )

        band = BANDS.get(band_name)
        if band is None:
            if "band_limits" in cfg:
                band = tuple(cfg["band_limits"])  # type: ignore[assignment]
            else:
                return MeasureResult(
                    value=float("nan"),
                    status="failed",
                    reason=f"unknown_band_{band_name}",
                    meta={},
                )

        # Peak presence check
        if not has_spectral_peak(signal, fs, band, threshold_sd=peak_threshold_sd):
            return MeasureResult(
                value=float("nan"),
                status="undefined",
                reason="no_peak",
                meta={"band": band_name, "band_limits": band},
            )

        # Zero-phase FIR band-pass filter
        numtaps = min(len(signal) // 3, int(3.0 * fs))
        if numtaps % 2 == 0:
            numtaps += 1

        try:
            fir_coeff = scipy.signal.firwin(
                numtaps,
                [band[0], band[1]],
                pass_zero=False,
                fs=fs,
            )
            filtered = scipy.signal.filtfilt(fir_coeff, [1.0], signal)
            envelope = np.abs(scipy.signal.hilbert(filtered))
        except Exception as e:
            return MeasureResult(
                value=float("nan"),
                status="failed",
                reason=f"filter_error_{e}",
                meta={},
            )

        alpha, scales, flucts, r2 = raw_dfa(envelope, fs=fs)
        return MeasureResult(
            value=alpha,
            status="ok",
            reason="",
            meta={
                "r2": r2,
                "band": band_name,
                "band_limits": band,
                "duration_s": duration_s,
            },
        )
