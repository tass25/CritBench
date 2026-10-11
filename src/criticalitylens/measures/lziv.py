"""Normalised Lempel-Ziv complexity (LZ76) for neural time series.

Implements median-binarised normalised complexity:
- Median thresholding to convert continuous signal to binary sequence {0, 1}
- Lempel-Ziv 1976 vocabulary decomposition
- Asymptotic normalization C_norm = C * log2(N) / N
- Broadband (1-45 Hz) and narrowband (30-45 Hz) filtering options
"""

from typing import Any
import numpy as np
import scipy.signal

from criticalitylens.measures.base import MeasureResult, register


def lziv_complexity_norm(signal: np.ndarray) -> float:
    """Compute normalized Lempel-Ziv (LZ76) complexity of a 1D signal.

    The continuous signal is median-binarized and normalized such that
    uncorrelated random noise approaches 1.0.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.

    Returns
    -------
    float
        Normalized LZ complexity C_norm.
    """
    n = len(signal)
    if n < 4:
        return float("nan")

    # Median binarization
    median_val = np.median(signal)
    binary_str = "".join("1" if x > median_val else "0" for x in signal)

    # Pure Python LZ76 algorithm
    i = 0
    c = 1
    k = 1
    k_max = 1

    while i + k <= n:
        # Check if binary_str[i:i+k] appeared in binary_str[0:i+k-1]
        sub = binary_str[i : i + k]
        history = binary_str[0 : i + k - 1]

        if sub in history:
            k += 1
        else:
            c += 1
            i += k
            k = 1

    # Asymptotic normalization (Lempel & Ziv 1976 / Zhang et al. 1999)
    # C_norm = c * log2(n) / n
    norm_factor = (n / np.log2(n)) if n > 1 else 1.0
    return float(c / norm_factor)


@register("lziv")
@register("lempel_ziv")
class LempelZivMeasure:
    """Lempel-Ziv normalized complexity measure."""

    @property
    def name(self) -> str:
        return "lziv"

    def compute(
        self,
        signal: np.ndarray,
        fs: float,
        config: dict[str, Any] | None = None,
    ) -> MeasureResult:
        cfg = config or {}
        band = cfg.get("band", "broadband")

        filtered = signal
        if band == "broadband":
            # 1 - 45 Hz bandpass
            sos = scipy.signal.butter(4, [1.0, min(45.0, fs / 2.0 - 1.0)], btype="bandpass", fs=fs, output="sos")
            filtered = scipy.signal.sosfiltfilt(sos, signal)
        elif band == "narrowband":
            # 30 - 45 Hz bandpass
            sos = scipy.signal.butter(4, [30.0, min(45.0, fs / 2.0 - 1.0)], btype="bandpass", fs=fs, output="sos")
            filtered = scipy.signal.sosfiltfilt(sos, signal)

        c_norm = lziv_complexity_norm(filtered)
        if np.isnan(c_norm):
            return MeasureResult(
                value=float("nan"),
                status="failed",
                reason="computation_failure",
                meta={},
            )

        return MeasureResult(
            value=c_norm,
            status="ok",
            reason="",
            meta={"band": band, "n_samples": len(signal)},
        )
