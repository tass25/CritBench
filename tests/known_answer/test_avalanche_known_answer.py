"""Known-answer tests for neuronal avalanche kappa index and branching ratio."""

import numpy as np
import pytest

from criticalitylens.measures.avalanche import (
    AvalancheBranchingMeasure,
    AvalancheKappaMeasure,
    calculate_branching_ratio,
    calculate_kappa,
    detect_avalanches,
)


def simulate_simple_branching_process(
    m: float,
    n_steps: int,
    drive_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate a pure Galton-Watson branching process with Poisson drive.

    Parameters
    ----------
    m : float
        Branching parameter.
    n_steps : int
        Number of discrete time steps.
    drive_rate : float
        Mean Poisson spontaneous external drive.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of activity counts across time bins.
    """
    activity = np.zeros(n_steps, dtype=int)
    activity[0] = rng.poisson(drive_rate)

    for t in range(1, n_steps):
        # Offspring generation from previous activity
        offspring = 0
        if activity[t - 1] > 0:
            offspring = rng.poisson(m * activity[t - 1])
        # External drive
        drive = rng.poisson(drive_rate)
        activity[t] = offspring + drive

    return activity


class TestAvalancheKnownAnswer:
    """Verification against simulated branching processes."""

    @pytest.mark.parametrize("true_m", [0.70, 0.90, 0.98])
    def test_branching_ratio_recovery(self, true_m: float) -> None:
        """Branching ratio estimator recovers parameter m under full observation."""
        rng = np.random.default_rng(42)
        n_steps = 30000
        drive_rate = 0.05  # sparse drive to ensure well-separated avalanches
        activity = simulate_simple_branching_process(true_m, n_steps, drive_rate, rng)

        # Directly compute branching ratio from active transitions
        branching_ratio = calculate_branching_ratio(activity)
        assert abs(branching_ratio - true_m) <= 0.08, (
            f"Expected branching ratio ~ {true_m}, got {branching_ratio:.3f}"
        )

    def test_kappa_near_one_at_criticality(self) -> None:
        """Kappa index at critical branching (m ~ 0.99 - 1.0) is close to 1.0."""
        rng = np.random.default_rng(101)
        n_steps = 50000
        drive_rate = 0.02
        activity = simulate_simple_branching_process(0.99, n_steps, drive_rate, rng)

        sizes, durations = detect_avalanches(activity)
        assert len(sizes) > 100, "Should detect sufficient avalanches"

        kappa = calculate_kappa(sizes, reference_tau=1.5)
        # At criticality, kappa should be close to 1.0 (+- 0.15)
        assert 0.85 <= kappa <= 1.15, f"Expected kappa ~ 1.0 at criticality, got {kappa:.3f}"

    def test_insufficient_events_returns_undefined(self) -> None:
        """Signal with insufficient excursions returns status='undefined' and reason='insufficient_events'."""
        fs = 250.0
        signal = np.zeros(int(fs * 30))  # flat signal has no threshold crossings

        measure = AvalancheKappaMeasure()
        result = measure.compute(signal, fs=fs, config={})
        assert result.status == "undefined"
        assert result.reason == "insufficient_events"
        assert np.isnan(result.value)
