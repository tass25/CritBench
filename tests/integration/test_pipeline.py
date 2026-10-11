"""Integration tests: multi-measure extraction on neural data fixtures."""

import numpy as np
import pytest

from criticalitylens.measures import get_measure, list_measures


def test_all_registered_measures_execute_on_fixture() -> None:
    """Every registered measure executes on a synthetic recording fixture without unhandled exceptions."""
    fs = 250.0
    duration_s = 180.0
    n_samples = int(fs * duration_s)
    t = np.arange(n_samples) / fs

    rng = np.random.default_rng(42)
    # Synthetic EEG fixture with slow baseline, 10 Hz alpha oscillation, and noise
    osc = 0.5 * np.sin(2 * np.pi * 10.0 * t)
    noise = rng.standard_normal(n_samples)
    fixture_signal = osc + noise

    measure_names = list_measures()
    assert len(measure_names) >= 6

    for name in measure_names:
        measure_cls = get_measure(name)
        measure = measure_cls()

        # Execute measure
        res = measure.compute(fixture_signal, fs=fs, config={})

        # Must always return a valid MeasureResult record, never raise unhandled error
        assert hasattr(res, "value")
        assert hasattr(res, "status")
        assert res.status in {"ok", "undefined", "excluded", "failed"}
        assert hasattr(res, "reason")
        assert hasattr(res, "meta")
