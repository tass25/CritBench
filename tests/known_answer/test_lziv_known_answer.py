"""Known-answer tests for normalised Lempel-Ziv complexity."""

import numpy as np
import pytest

from criticalitylens.measures.lziv import LempelZivMeasure, lziv_complexity_norm


class TestLZIVKnownAnswer:
    """Verification of asymptotic bounds for normalized LZ complexity."""

    def test_white_noise_approaches_one(self) -> None:
        """Normalized LZ complexity of random uncorrelated white noise approaches 1.0."""
        rng = np.random.default_rng(42)
        n_samples = 10000
        signal = rng.standard_normal(n_samples)

        c_norm = lziv_complexity_norm(signal)
        # Normalised LZ for random binary sequences asymptotically converges to 1.0
        assert 0.90 <= c_norm <= 1.10, f"Expected c_norm ~ 1.0 for white noise, got {c_norm:.3f}"

    def test_periodic_sine_signal_low_complexity(self) -> None:
        """Periodic sine waves exhibit low complexity (c_norm < 0.15)."""
        fs = 250.0
        t = np.arange(int(fs * 30)) / fs
        signal = np.sin(2 * np.pi * 5.0 * t)

        c_norm = lziv_complexity_norm(signal)
        assert c_norm < 0.15, f"Expected periodic sine c_norm < 0.15, got {c_norm:.3f}"

    def test_measure_contract(self) -> None:
        """Verify LempelZivMeasure executes and populates MeasureResult correctly."""
        rng = np.random.default_rng(123)
        fs = 250.0
        signal = rng.standard_normal(int(fs * 30))

        measure = LempelZivMeasure()
        result = measure.compute(signal, fs=fs, config={"band": "broadband"})

        assert result.status == "ok"
        assert not np.isnan(result.value)
        assert 0.65 <= result.value <= 1.15
