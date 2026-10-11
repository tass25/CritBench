"""Neuronal avalanche analysis: kappa index and branching ratio.

Implements neuronal avalanche metrics according to the locked scientific specifications:
- Event detection: negative excursions beyond 3 SD of z-scored channel
- Temporal binning with bin width equal to the average inter-event interval (IEI)
- Avalanche definition: contiguous non-empty bins flanked by quiescent bins
- Kappa index (kappa): Shew et al. 2009 metric comparing empirical avalanche size CDF
  to the critical reference power-law CDF (tau = 1.5)
- Branching ratio (sigma): mean ratio of event counts in successive bins within avalanches
"""

from typing import Any
import numpy as np

from criticalitylens.measures.base import MeasureResult, register


def detect_events(signal: np.ndarray, threshold_sd: float = 3.0) -> np.ndarray:
    """Detect negative local peak excursions exceeding threshold SD.

    Parameters
    ----------
    signal : np.ndarray
        1D continuous neural time series.
    threshold_sd : float, default 3.0
        Z-score threshold for event identification.

    Returns
    -------
    np.ndarray
        Integer indices of detected event peaks.
    """
    std = np.std(signal)
    if std < 1e-12:
        return np.array([], dtype=int)

    mean = np.mean(signal)
    z = (signal - mean) / std

    # Identify negative excursions beyond threshold
    is_below = z < -threshold_sd
    if not np.any(is_below):
        return np.array([], dtype=int)

    # Local minimum peak detection
    peaks = []
    n = len(z)
    for i in range(1, n - 1):
        if is_below[i] and z[i] <= z[i - 1] and z[i] <= z[i + 1]:
            peaks.append(i)

    return np.array(peaks, dtype=int)


def bin_events(event_indices: np.ndarray, n_samples: int, bin_width: int) -> np.ndarray:
    """Discretize event indices into discrete time bins of fixed width.

    Parameters
    ----------
    event_indices : np.ndarray
        Sample indices of events.
    n_samples : int
        Total length of the continuous signal in samples.
    bin_width : int
        Width of each time bin in samples.

    Returns
    -------
    np.ndarray
        Counts of events per time bin.
    """
    if len(event_indices) == 0 or bin_width < 1:
        return np.zeros(1, dtype=int)

    n_bins = int(np.ceil(n_samples / bin_width))
    bins = np.zeros(n_bins, dtype=int)
    bin_ids = event_indices // bin_width
    valid_ids = bin_ids[bin_ids < n_bins]

    for b in valid_ids:
        bins[b] += 1

    return bins


def detect_avalanches(activity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract avalanche sizes and durations from binned event counts.

    Parameters
    ----------
    activity : np.ndarray
        Array of event counts per time bin.

    Returns
    -------
    sizes : np.ndarray
        Total event counts for each identified avalanche.
    durations : np.ndarray
        Duration in number of bins for each identified avalanche.
    """
    sizes = []
    durations = []

    in_avalanche = False
    current_size = 0
    current_duration = 0

    for count in activity:
        if count > 0:
            in_avalanche = True
            current_size += count
            current_duration += 1
        else:
            if in_avalanche:
                sizes.append(current_size)
                durations.append(current_duration)
                in_avalanche = False
                current_size = 0
                current_duration = 0

    # Capture trailing avalanche if signal terminates actively
    if in_avalanche:
        sizes.append(current_size)
        durations.append(current_duration)

    return np.array(sizes, dtype=int), np.array(durations, dtype=int)


def calculate_branching_ratio(activity: np.ndarray) -> float:
    """Compute the avalanche branching ratio sigma.

    Calculated as the mean branching parameter E[N_{t+1} | N_t] / N_t across
    all active time steps N_t > 0 (including extinction transitions N_{t+1} = 0).

    Parameters
    ----------
    activity : np.ndarray
        Event counts per time bin.

    Returns
    -------
    float
        Branching ratio estimate.
    """
    if len(activity) < 2:
        return float("nan")

    active_idx = np.where(activity[:-1] > 0)[0]
    if len(active_idx) < 5:
        return float("nan")

    n_t = activity[active_idx].astype(float)
    n_next = activity[active_idx + 1].astype(float)

    denom = np.sum(n_t)
    if denom == 0:
        return float("nan")

    return float(np.sum(n_next) / denom)



def calculate_kappa(
    sizes: np.ndarray,
    reference_tau: float = 1.5,
    n_points: int = 10,
) -> float:
    """Compute the avalanche kappa index (Shew et al. 2009).

    kappa = 1 + (1 / n_points) * sum_{k=1}^{n_points} (F_data(beta_k) - F_ref(beta_k))

    Parameters
    ----------
    sizes : np.ndarray
        Array of avalanche sizes.
    reference_tau : float, default 1.5
        Reference critical power-law exponent.
    n_points : int, default 10
        Number of logarithmic evaluation points.

    Returns
    -------
    float
        Kappa index.
    """
    if len(sizes) < 10:
        return float("nan")

    s_min = float(np.min(sizes))
    s_max = float(np.max(sizes))

    if s_max <= s_min:
        return 1.0

    # Logarithmically spaced test points
    betas = np.geomspace(s_min, s_max, num=n_points)

    # Empirical CDF: F_data(beta) = P(S < beta)
    n_total = len(sizes)
    f_data = np.array([np.sum(sizes < b) / n_total for b in betas])

    # Reference CDF for power law P(S) ~ S^(-tau)
    # CDF: (S^(1-tau) - S_min^(1-tau)) / (S_max^(1-tau) - S_min^(1-tau))
    p = 1.0 - reference_tau
    denom = (s_max**p) - (s_min**p)
    if abs(denom) < 1e-12:
        f_ref = np.linspace(0.0, 1.0, n_points)
    else:
        f_ref = ((betas**p) - (s_min**p)) / denom
        f_ref = np.clip(f_ref, 0.0, 1.0)

    # Shew et al. 2009 kappa equation
    diff = f_data - f_ref
    kappa = 1.0 + (1.0 / n_points) * np.sum(diff)
    return float(kappa)


@register("avalanche_kappa")
class AvalancheKappaMeasure:
    """Avalanche kappa index measure."""

    @property
    def name(self) -> str:
        return "avalanche_kappa"

    def compute(
        self,
        signal: np.ndarray,
        fs: float,
        config: dict[str, Any] | None = None,
    ) -> MeasureResult:
        cfg = config or {}
        threshold_sd = float(cfg.get("threshold_sd", 3.0))

        events = detect_events(signal, threshold_sd=threshold_sd)
        if len(events) < 10:
            return MeasureResult(
                value=float("nan"),
                status="undefined",
                reason="insufficient_events",
                meta={"event_count": len(events)},
            )

        iei = np.diff(events)
        mean_iei = int(max(1, np.round(np.mean(iei))))

        activity = bin_events(events, len(signal), bin_width=mean_iei)
        sizes, durations = detect_avalanches(activity)

        if len(sizes) < 10:
            return MeasureResult(
                value=float("nan"),
                status="undefined",
                reason="insufficient_events",
                meta={"n_avalanches": len(sizes)},
            )

        kappa = calculate_kappa(sizes, reference_tau=1.5)
        return MeasureResult(
            value=kappa,
            status="ok",
            reason="",
            meta={"n_avalanches": len(sizes), "mean_iei_samples": mean_iei},
        )


@register("avalanche_branching")
class AvalancheBranchingMeasure:
    """Avalanche branching ratio measure."""

    @property
    def name(self) -> str:
        return "avalanche_branching"

    def compute(
        self,
        signal: np.ndarray,
        fs: float,
        config: dict[str, Any] | None = None,
    ) -> MeasureResult:
        cfg = config or {}
        threshold_sd = float(cfg.get("threshold_sd", 3.0))

        events = detect_events(signal, threshold_sd=threshold_sd)
        if len(events) < 10:
            return MeasureResult(
                value=float("nan"),
                status="undefined",
                reason="insufficient_events",
                meta={"event_count": len(events)},
            )

        iei = np.diff(events)
        mean_iei = int(max(1, np.round(np.mean(iei))))

        activity = bin_events(events, len(signal), bin_width=mean_iei)
        sigma = calculate_branching_ratio(activity)

        return MeasureResult(
            value=sigma,
            status="ok",
            reason="",
            meta={"mean_iei_samples": mean_iei},
        )
