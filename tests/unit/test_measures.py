"""Unit tests for the measure interface and registry."""

import numpy as np
import pytest

from criticalitylens.config.exceptions import MeasureUndefined
from criticalitylens.measures import (
    AperiodicMeasure,
    DFAMeasure,
    HiguchiMeasure,
    LempelZivMeasure,
    Measure,
    MeasureResult,
    clear_registry,
    get_measure,
    list_measures,
    register,
)
from criticalitylens.measures.base import _REGISTRY
from criticalitylens.measures.higuchi import find_optimal_kmax
from criticalitylens.surrogates.iaaft import iaaft_surrogate
from criticalitylens.surrogates.phase_randomized import phase_randomized_surrogate


def test_registry_contains_core_measures() -> None:
    """Verify that all core criticality measures are registered."""
    measures = list_measures()
    expected = [
        "aperiodic",
        "aperiodic_exponent",
        "avalanche_branching",
        "avalanche_kappa",
        "dfa",
        "higuchi",
        "higuchi_fd",
        "lempel_ziv",
        "lziv",
    ]
    for m in expected:
        assert m in measures, f"Measure '{m}' missing from registry"


def test_get_measure_unknown_raises() -> None:
    """Requesting an unregistered measure must raise MeasureUndefined."""
    with pytest.raises(MeasureUndefined):
        get_measure("non_existent_measure")


def test_measure_protocol_compliance() -> None:
    """Verify registered classes conform to the Measure protocol."""
    for name in list_measures():
        cls = get_measure(name)
        instance = cls()
        assert isinstance(instance, Measure)
        assert hasattr(instance, "name")
        assert callable(instance.compute)


def test_measure_result_immutability() -> None:
    """Verify that MeasureResult is frozen and immutable."""
    res = MeasureResult(value=1.23, status="ok", reason="", meta={"test": True})
    assert res.value == 1.23
    assert res.status == "ok"
    assert res.reason == ""
    with pytest.raises(Exception):  # Frozen dataclass raises FrozenInstanceError
        res.value = 4.56  # type: ignore[misc]


def test_direct_registration_and_clear() -> None:
    """Test registering a measure class directly and clearing registry."""
    dummy_name = "dummy_test_measure"

    class DummyMeasure:
        @property
        def name(self) -> str:
            return dummy_name

        def compute(self, signal: np.ndarray, fs: float, config: dict) -> MeasureResult:
            return MeasureResult(value=0.0, status="ok", reason="", meta={})

    register(dummy_name, DummyMeasure)
    assert dummy_name in list_measures()
    assert get_measure(dummy_name) is DummyMeasure

    # Clean up dummy from registry
    _REGISTRY.pop(dummy_name, None)


def test_higuchi_find_optimal_kmax() -> None:
    """Test kmax stability sweep helper."""
    rng = np.random.default_rng(42)
    sig = rng.standard_normal(2000)
    k = find_optimal_kmax(sig, k_range=range(4, 15))
    assert 4 <= k <= 15


def test_dfa_edge_cases() -> None:
    """Verify DFA handling of short durations and unknown bands."""
    measure = DFAMeasure()
    sig = np.zeros(100)

    # Duration too short
    res_short = measure.compute(sig, fs=250.0, config={"min_duration_s": 180.0})
    assert res_short.status == "excluded"
    assert res_short.reason == "duration_too_short"

    # Unknown band
    res_band = measure.compute(sig, fs=250.0, config={"band": "gamma_unknown", "min_duration_s": 0.1})
    assert res_band.status == "failed"
    assert "unknown_band" in res_band.reason


def test_aperiodic_duration_too_short() -> None:
    """Aperiodic fit on extremely short signal is excluded."""
    measure = AperiodicMeasure()
    sig = np.array([1.0, 2.0])
    res = measure.compute(sig, fs=250.0)
    assert res.status == "excluded"
    assert res.reason == "duration_too_short"


def test_lziv_narrowband_variant() -> None:
    """Lempel-Ziv narrow-band filtering computation."""
    rng = np.random.default_rng(42)
    sig = rng.standard_normal(250 * 5)
    measure = LempelZivMeasure()
    res = measure.compute(sig, fs=250.0, config={"band": "narrowband"})
    assert res.status == "ok"
    assert not np.isnan(res.value)


def test_surrogates_default_rng() -> None:
    """Surrogates run cleanly when rng is None."""
    sig = np.random.randn(500)
    surr1 = phase_randomized_surrogate(sig)
    surr2 = iaaft_surrogate(sig, max_iter=20)
    assert len(surr1) == len(sig)
    assert len(surr2) == len(sig)
