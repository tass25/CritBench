# Methods

## Preprocessing

MNE-Python pipeline: load EEG channels, notch filter at mains frequency, per-measure band-pass, average reference, amplitude-based epoch rejection, ICA for eye artifacts where applicable. All thresholds are configurable.

## Surrogates

Phase-randomised and IAAFT surrogates (10 per segment). Surrogate residual = real value minus mean surrogate value.

## Statistics

Linear mixed-effects models with random intercept per subject. Bootstrap over subjects (2,000 resamples). Benjamini-Hochberg correction across measures and contrasts.
