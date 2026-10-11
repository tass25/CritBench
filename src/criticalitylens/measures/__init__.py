"""Criticality measures behind a unified interface."""

from criticalitylens.measures.aperiodic import AperiodicMeasure
from criticalitylens.measures.avalanche import (
    AvalancheBranchingMeasure,
    AvalancheKappaMeasure,
)
from criticalitylens.measures.base import (
    Measure,
    MeasureResult,
    clear_registry,
    get_measure,
    list_measures,
    register,
)
from criticalitylens.measures.dfa import DFAMeasure
from criticalitylens.measures.higuchi import HiguchiMeasure
from criticalitylens.measures.lziv import LempelZivMeasure

__all__ = [
    "Measure",
    "MeasureResult",
    "register",
    "get_measure",
    "list_measures",
    "clear_registry",
    "DFAMeasure",
    "AperiodicMeasure",
    "LempelZivMeasure",
    "HiguchiMeasure",
    "AvalancheKappaMeasure",
    "AvalancheBranchingMeasure",
]
