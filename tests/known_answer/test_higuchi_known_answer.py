"""Known-answer tests for Higuchi fractal dimension."""

import numpy as np
import pytest

from criticalitylens.measures.higuchi import HiguchiMeasure, higuchi_fd


class TestHiguchiKnownAnswer:
    """Verification of Higuchi Fractal Dimension on canonical signals."""

    def test_white_noise_approaches_two(self) -> None:
        """Pure white noise has theoretical fractal dimension D ~ 2.0."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal(8000)

        d, r2 = higuchi_fd(signal, kmax=10)
        assert 1.85 <= d <= 2.10, f"Expected D ~ 2.0 for white noise, got {d:.3f}"
        assert r2 > 0.95

    def test_pure_sine_wave_approaches_one(self) -> None:
        """Differentiable 1D smooth sine curve has topological dimension D ~ 1.0."""
        fs = 500.0
        t = np.arange(int(fs * 20)) / fs
        signal = np.sin(2 * np.pi * 2.0 * t)

        d, r2 = higuchi_fd(signal, kmax=10)
        assert 1.00 <= d <= 1.15, f"Expected D ~ 1.0 for sine wave, got {d:.3f}"

    def test_brownian_motion_approaches_one_point_five(self) -> None:
        """Brownian motion (cumulative sum of white noise) has theoretical D ~ 1.5."""
        rng = np.random.default_rng(77)
        white = rng.standard_normal(8000)
        brownian = np.cumsum(white)

        d, r2 = higuchi_fd(brownian, kmax=10)
        assert 1.40 <= d <= 1.60, f"Expected D ~ 1.5 for Brownian motion, got {d:.3f}"

    def test_measure_contract(self) -> None:
        """HiguchiMeasure executes cleanly and yields valid MeasureResult."""
        rng = np.random.default_rng(10)
        fs = 250.0
        signal = rng.standard_normal(int(fs * 30))

        measure = HiguchiMeasure()
        result = measure.compute(signal, fs=fs, config={"kmax": 10})

        assert result.status == "ok"
        assert not np.isnan(result.value)
        assert 1.80 <= result.value <= 2.10
