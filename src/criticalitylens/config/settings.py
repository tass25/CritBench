"""Configuration dataclasses and YAML loading for pipeline settings."""

from dataclasses import dataclass
from pathlib import Path

from criticalitylens.config.exceptions import ConfigError


@dataclass(frozen=True)
class PipelineConfig:
    """Pipeline execution configuration.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    jobs : int, default 1
        Number of parallel jobs to execute.
    dry_run : bool, default False
        Whether to run in dry-run mode without persistent writes.
    """

    seed: int
    jobs: int = 1
    dry_run: bool = False


def load_config(path: Path) -> PipelineConfig:
    """Load and validate pipeline configuration from a YAML file.

    Parameters
    ----------
    path : Path
        Path to the configuration YAML file.

    Returns
    -------
    PipelineConfig
        Validated configuration instance.

    Raises
    ------
    ConfigError
        If the configuration cannot be loaded or is invalid.
    """
    raise ConfigError("Not yet implemented for this configuration path")
