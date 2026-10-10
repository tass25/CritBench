"""Command-line entry point for the CriticalityLens pipeline."""

from criticalitylens import __version__


def main() -> None:
    """Print package name and version to standard output."""
    print(f"criticalitylens {__version__}")


if __name__ == "__main__":
    main()
