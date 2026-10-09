# CriticalityLens

An open-source Python package and command-line pipeline for computing criticality measures on neural time series.

CriticalityLens simulates artificial neural signals with known distance to criticality, computes six criticality measures behind a unified interface, builds spectrum-preserving surrogates, and runs reproducible analyses on real EEG datasets.

## Installation

```bash
git clone https://github.com/tass25/CritBench.git
cd CritBench
pip install -e ".[dev]"
```

## Quick start

```bash
cl --help
```

## Reproduction

```bash
make setup
make data
make run
make report
```

## Measures

- Detrended fluctuation analysis (DFA) exponent
- Aperiodic (1/f) spectral exponent
- Lempel-Ziv complexity
- Higuchi fractal dimension
- Avalanche kappa index
- Avalanche branching ratio

## Datasets

- [Sleep-EDF Expanded](https://www.physionet.org/content/sleep-edfx/) (153 sleep-cassette recordings)
- [ds004902](https://openneuro.org/datasets/ds004902) (71 participants, normal sleep vs. sleep deprivation)

## Data policy

Raw data is never redistributed. Downloaded files are checksum-verified and treated as read-only. The `results/` directory holds aggregates only.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT. See [LICENSE](LICENSE).
