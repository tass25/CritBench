"""Configuration dataclasses and YAML loading for pipeline settings."""

from dataclasses import dataclass
from pathlib import Path
import yaml

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


def load_config(path: Path | str) -> PipelineConfig:
    """Load and validate pipeline configuration from a YAML file.

    Parameters
    ----------
    path : Path or str
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
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"Configuration file does not exist: {p}")

    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        if not isinstance(data, dict):
            raise ConfigError(f"Configuration must be a YAML mapping, got {type(data).__name__}")

        seed = int(data.get("seed", 42))
        jobs = int(data.get("jobs", 1))
        dry_run = bool(data.get("dry_run", False))

        return PipelineConfig(seed=seed, jobs=jobs, dry_run=dry_run)
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(f"Failed to load configuration from {p}: {e}") from e
