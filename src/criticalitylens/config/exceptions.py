"""Custom exception hierarchy."""


class ConfigError(Exception):
    """Raised when configuration validation or loading fails."""


class DataIntegrityError(Exception):
    """Raised when dataset integrity checks fail."""


class MeasureUndefined(Exception):
    """Raised when a requested measure is undefined or invalid for input data."""
