"""Unit tests for configuration loading and validation."""

from pathlib import Path
import pytest

from criticalitylens.config.exceptions import ConfigError
from criticalitylens.config.settings import PipelineConfig, load_config


def test_load_config_valid(tmp_path: Path) -> None:
    """Valid YAML configuration loads correctly into PipelineConfig."""
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("seed: 123\njobs: 4\ndry_run: true\n", encoding="utf-8")

    cfg = load_config(cfg_file)
    assert isinstance(cfg, PipelineConfig)
    assert cfg.seed == 123
    assert cfg.jobs == 4
    assert cfg.dry_run is True


def test_load_config_defaults(tmp_path: Path) -> None:
    """Empty or minimal YAML uses defaults."""
    cfg_file = tmp_path / "config_min.yaml"
    cfg_file.write_text("seed: 99\n", encoding="utf-8")

    cfg = load_config(cfg_file)
    assert cfg.seed == 99
    assert cfg.jobs == 1
    assert cfg.dry_run is False


def test_load_config_missing_file_raises() -> None:
    """Non-existent file raises ConfigError."""
    with pytest.raises(ConfigError):
        load_config("does_not_exist.yaml")


def test_load_config_invalid_yaml(tmp_path: Path) -> None:
    """Invalid YAML structure raises ConfigError."""
    cfg_file = tmp_path / "invalid.yaml"
    cfg_file.write_text("- item1\n- item2\n", encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(cfg_file)
