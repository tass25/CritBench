"""Known-answer tests for DFA against fractional Gaussian noise and colored noise."""

import numpy as np
import pytest

from criticalitylens.measures.base import get_measure
from criticalitylens.measures.dfa import DFAMeasure, raw_dfa


def generate_colored_noise(
    beta: float,
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate 1/f^beta noise via Fourier filtering.

    Parameters
    ----------
    beta : float
        Spectral exponent (0 = white, 1 = pink, 2 = Brownian).
    n_samples : int
        Number of time points.
    fs : float
        Sampling frequency in Hz.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Synthetic noise time series with zero mean and unit variance.
    """
    white = rng.standard_normal(n_samples)
    fft_vals = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    # Avoid division by zero at DC component
    scaling = np.ones_like(freqs)
    scaling[1:] = 1.0 / (freqs[1:] ** (beta / 2.0))
    scaling[0] = 0.0

    colored_fft = fft_vals * scaling
    colored = np.fft.irfft(colored_fft, n=n_samples)
    return (colored - np.mean(colored)) / np.std(colored)


class TestDFAKnownAnswer:
    """Verification against known analytical scaling behaviors."""

    def test_white_noise_scaling(self) -> None:
        """White noise (beta=0) must exhibit DFA exponent close to 0.5."""
        rng = np.random.default_rng(42)
        fs = 250.0
        n_samples = int(fs * 180)  # 3 minutes
        signal = generate_colored_noise(beta=0.0, n_samples=n_samples, fs=fs, rng=rng)

        alpha, _, _, r2 = raw_dfa(signal, fs=fs, window_min_s=1.0, window_max_s=18.0)
        assert 0.44 <= alpha <= 0.56, f"Expected alpha ~ 0.5 for white noise, got {alpha:.3f}"
        assert r2 > 0.98, f"Expected linear log-log fit r2 > 0.98, got {r2:.3f}"

    def test_pink_noise_scaling(self) -> None:
        """Pink noise (beta=1) must exhibit DFA exponent close to 1.0."""
        rng = np.random.default_rng(123)
        fs = 250.0
        n_samples = int(fs * 180)
        signal = generate_colored_noise(beta=1.0, n_samples=n_samples, fs=fs, rng=rng)

        alpha, _, _, r2 = raw_dfa(signal, fs=fs, window_min_s=1.0, window_max_s=18.0)
        assert 0.90 <= alpha <= 1.10, f"Expected alpha ~ 1.0 for pink noise, got {alpha:.3f}"
        assert r2 > 0.98, f"Expected linear log-log fit r2 > 0.98, got {r2:.3f}"

    def test_brownian_noise_scaling(self) -> None:
        """Brownian noise (beta=2) must exhibit DFA exponent close to 1.5."""
        rng = np.random.default_rng(999)
        fs = 250.0
        n_samples = int(fs * 180)
        signal = generate_colored_noise(beta=2.0, n_samples=n_samples, fs=fs, rng=rng)

        alpha, _, _, r2 = raw_dfa(signal, fs=fs, window_min_s=1.0, window_max_s=18.0)
        assert 1.40 <= alpha <= 1.60, f"Expected alpha ~ 1.5 for Brownian noise, got {alpha:.3f}"
        assert r2 > 0.98, f"Expected linear log-log fit r2 > 0.98, got {r2:.3f}"

    def test_absence_of_peak_returns_undefined(self) -> None:
        """Broadband DFA without an oscillatory peak must return status='undefined' and reason='no_peak'."""
        rng = np.random.default_rng(7)
        fs = 250.0
        n_samples = int(fs * 180)
        # Pure white noise has no prominent alpha peak above aperiodic spectrum
        signal = rng.standard_normal(n_samples)

        measure = DFAMeasure()
        result = measure.compute(
            signal,
            fs=fs,
            config={"band": "alpha", "min_duration_s": 180.0, "peak_threshold_sd": 1.5},
        )
        assert result.status == "undefined"
        assert result.reason == "no_peak"
        assert np.isnan(result.value)

    def test_with_alpha_oscillation_peak(self) -> None:
        """Signal with strong alpha peak produces status='ok' and positive alpha exponent."""
        rng = np.random.default_rng(101)
        fs = 250.0
        n_samples = int(fs * 180)
        t = np.arange(n_samples) / fs

        # Alpha burst modulated by slow 1/f amplitude
        slow_amp = np.abs(generate_colored_noise(beta=1.0, n_samples=n_samples, fs=fs, rng=rng))
        oscillation = slow_amp * np.sin(2 * np.pi * 10.0 * t)
        noise = 0.2 * rng.standard_normal(n_samples)
        signal = oscillation + noise

        measure = DFAMeasure()
        result = measure.compute(
            signal,
            fs=fs,
            config={"band": "alpha", "min_duration_s": 180.0, "peak_threshold_sd": 1.0},
        )
        assert result.status == "ok"
        assert not np.isnan(result.value)
        assert 0.5 <= result.value <= 1.5
