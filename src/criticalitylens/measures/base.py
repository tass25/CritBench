"""Measure protocol, MeasureResult record, and measure registry."""

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

from criticalitylens.config.exceptions import MeasureUndefined


@dataclass(frozen=True)
class MeasureResult:
    """Result record produced by a criticality measure computation.

    Parameters
    ----------
    value : float
        Scalar output of the measure.
    status : str
        Execution status ('ok', 'rejected', 'failed').
    reason : str
        Explanation if status is not 'ok', empty otherwise.
    meta : dict[str, object]
        Additional metadata and intermediate parameters.
    """

    value: float
    status: str
    reason: str
    meta: dict[str, object]


@runtime_checkable
class Measure(Protocol):
    """Protocol defining the interface for criticality measures."""

    @property
    def name(self) -> str:
        """Measure identifier.

        Returns
        -------
        str
            Unique name of the measure.
        """
        ...

    def compute(
        self,
        signal: np.ndarray,
        fs: float,
        config: dict[str, object],
    ) -> MeasureResult:
        """Compute the criticality measure on the input signal.

        Parameters
        ----------
        signal : numpy.ndarray
            Input neural signal time series.
        fs : float
            Sampling frequency in Hertz.
        config : dict[str, object]
            Measure configuration parameters.

        Returns
        -------
        MeasureResult
            Computed measure value, status, and metadata.
        """
        ...


_REGISTRY: dict[str, type[Measure]] = {}


def register(
    name: str,
    measure_cls: type[Measure] | None = None,
) -> Any:
    """Register a measure class in the registry.

    Parameters
    ----------
    name : str
        Unique identifier for the measure.
    measure_cls : type[Measure], optional
        Measure class to register. If None, returns a decorator.

    Returns
    -------
    type[Measure] or Callable[[type[Measure]], type[Measure]]
        Registered measure class or a decorator that registers the class.
    """
    if measure_cls is not None:
        _REGISTRY[name] = measure_cls
        return measure_cls

    def decorator(cls: type[Measure]) -> type[Measure]:
        _REGISTRY[name] = cls
        return cls

    return decorator


def get_measure(name: str) -> type[Measure]:
    """Retrieve a registered measure class by name.

    Parameters
    ----------
    name : str
        Unique identifier of the measure.

    Returns
    -------
    type[Measure]
        The registered measure class.

    Raises
    ------
    MeasureUndefined
        If no measure is registered under the given name.
    """
    if name not in _REGISTRY:
        raise MeasureUndefined(f"Measure '{name}' is not registered")
    return _REGISTRY[name]
