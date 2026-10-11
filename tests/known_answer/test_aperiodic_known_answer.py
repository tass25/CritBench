"""Known-answer tests for aperiodic exponent recovery."""

import numpy as np
import pytest

from criticalitylens.measures.aperiodic import AperiodicMeasure, fit_aperiodic_exponent


def generate_synthetic_psd_signal(
    true_exponent: float,
    n_samples: int,
    fs: float,
    rng: np.random.Generator,
    peak_freq: float | None = None,
    peak_amp: float = 0.0,
) -> np.ndarray:
    """Generate synthetic time series with known spectral exponent and optional oscillatory peak."""
    white = rng.standard_normal(n_samples)
    fft_vals = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)

    scaling = np.ones_like(freqs)
    scaling[1:] = 1.0 / (freqs[1:] ** (true_exponent / 2.0))
    scaling[0] = 0.0

    colored = np.fft.irfft(fft_vals * scaling, n=n_samples)
    signal = (colored - np.mean(colored)) / np.std(colored)

    if peak_freq is not None and peak_amp > 0:
        t = np.arange(n_samples) / fs
        signal += peak_amp * np.sin(2 * np.pi * peak_freq * t)

    return signal


class TestAperiodicKnownAnswer:
    """Test recovery of spectral exponent against synthetic 1/f signals."""

    @pytest.mark.parametrize("target_beta", [1.0, 1.5, 2.0])
    def test_pure_colored_noise_recovery(self, target_beta: float) -> None:
        """Aperiodic fit must recover true spectral exponent within 0.15 tolerance."""
        rng = np.random.default_rng(int(target_beta * 100))
        fs = 250.0
        n_samples = int(fs * 30)  # 30-second epoch
        signal = generate_synthetic_psd_signal(target_beta, n_samples, fs, rng)

        exponent, r2 = fit_aperiodic_exponent(signal, fs=fs, freq_range=(1.0, 40.0))
        assert abs(exponent - target_beta) <= 0.15, (
            f"Expected exponent ~ {target_beta}, got {exponent:.3f} (r2={r2:.3f})"
        )
        assert r2 > 0.90

    def test_noise_with_alpha_peak(self) -> None:
        """Aperiodic fit with an added 10 Hz alpha peak must not bias the underlying exponent."""
        target_beta = 1.5
        rng = np.random.default_rng(42)
        fs = 250.0
        n_samples = int(fs * 60)
        signal = generate_synthetic_psd_signal(
            true_exponent=target_beta,
            n_samples=n_samples,
            fs=fs,
            rng=rng,
            peak_freq=10.0,
            peak_amp=1.5,
        )

        measure = AperiodicMeasure()
        result = measure.compute(signal, fs=fs, config={"freq_range": (1.0, 40.0)})

        assert result.status == "ok"
        assert abs(result.value - target_beta) <= 0.25, (
            f"Expected exponent ~ {target_beta} despite peak, got {result.value:.3f}"
        )
