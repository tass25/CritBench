"""Phase-randomised Fourier surrogates.

Generates spectrum-preserving surrogates by preserving Fourier amplitudes
while shuffling phases uniformly on [0, 2pi) subject to Hermitian symmetry.
"""

import numpy as np


def phase_randomized_surrogate(
    signal: np.ndarray,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate a phase-randomized surrogate of a 1D real time series.

    Preserves the Fourier power spectrum and autocorrelation structure
    while destroying non-linear temporal dynamics.

    Parameters
    ----------
    signal : np.ndarray
        1D real input time series.
    rng : np.random.Generator, optional
        Random number generator.

    Returns
    -------
    np.ndarray
        Phase-randomized surrogate time series with exact power spectrum match.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(signal)
    fft_vals = np.fft.rfft(signal)
    amplitudes = np.abs(fft_vals)

    # Random phases on [0, 2pi) for non-DC frequencies
    random_phases = rng.uniform(0.0, 2.0 * np.pi, size=len(fft_vals))
    random_phases[0] = 0.0  # DC component phase preserved
    if n % 2 == 0:
        random_phases[-1] = 0.0  # Nyquist frequency phase preserved

    surr_fft = amplitudes * np.exp(1j * random_phases)
    surrogate = np.fft.irfft(surr_fft, n=n)
    return np.real(surrogate)
