# Architecture

CriticalityLens follows a layered architecture with dependencies pointing inward. Pure functions handle all numerics; side effects (disk, network, logging) occur only at the edges.

## Package structure

```
src/criticalitylens/
  config/       Typed, validated settings loaded from YAML
  sim/          Branching process, modulated oscillation, channel mixing
  measures/     Unified measure interface with DFA, aperiodic, LZiv, Higuchi, avalanche
  surrogates/   Phase-randomised and IAAFT surrogate generation
  data/         Dataset manifests, downloaders, readers
  prep/         MNE-Python preprocessing and segmentation
  analysis/     Redundancy, slope-explained variance, agreement, simulation recovery
  report/       Figures and tables
  cli.py        Command-line entry point
```

## Key contracts

- **Measure protocol:** `compute(signal, fs, config) -> MeasureResult`
- **Results format:** single Parquet table with provenance columns
- **Configuration:** frozen dataclasses loaded from YAML, parameter names carry units
- **Stages:** idempotent, resumable, cached by configuration and input hash
