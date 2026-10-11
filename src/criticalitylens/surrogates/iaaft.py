"""Iterative Amplitude Adjusted Fourier Transform (IAAFT) surrogates.

Generates surrogates that simultaneously preserve:
1. The exact empirical amplitude distribution (via rank replacement).
2. The power spectral density / autocorrelation function (via iterative Fourier projection).
(Schreiber & Schmitz 1996).
"""

import numpy as np


def iaaft_surrogate(
    signal: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-4,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate an IAAFT surrogate of a 1D real signal.

    Parameters
    ----------
    signal : np.ndarray
        1D input signal.
    max_iter : int, default 200
        Maximum iteration steps.
    tol : float, default 1e-4
        Convergence tolerance on relative spectrum error.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Surrogate signal matching original sorted amplitude distribution and power spectrum.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(signal)
    sorted_orig = np.sort(signal)

    orig_fft = np.fft.rfft(signal)
    target_amplitudes = np.abs(orig_fft)
    target_power = target_amplitudes**2

    # Initial condition: random permutation of original data
    current = rng.permutation(signal)

    for _ in range(max_iter):
        # 1. Fourier transform of current series
        curr_fft = np.fft.rfft(current)
        curr_phases = np.angle(curr_fft)

        # 2. Adjust amplitudes to target Fourier spectrum while keeping current phases
        adjusted_fft = target_amplitudes * np.exp(1j * curr_phases)
        adjusted_signal = np.fft.irfft(adjusted_fft, n=n)

        # 3. Rank replacement: substitute sorted original amplitudes according to ranks
        ranks = np.argsort(np.argsort(adjusted_signal))
        new_current = sorted_orig[ranks]

        # 4. Convergence check
        new_power = np.abs(np.fft.rfft(new_current)) ** 2
        rel_err = np.mean(np.abs(new_power - target_power) / (np.mean(target_power) + 1e-12))
        current = new_current

        if rel_err < tol:
            break

    return current
