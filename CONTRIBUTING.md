# Contributing to CriticalityLens

## Development setup

```bash
git clone https://github.com/tass25/CritBench.git
cd CritBench
pip install -e ".[dev]"
```

## Workflow

1. Create an issue describing the change.
2. Branch from `main`.
3. Write tests first for scientific code.
4. Implement with full type hints.
5. Run `make lint typecheck test` before pushing.
6. Open a pull request linking the issue.

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/): imperative mood, one logical change per commit.

## Code quality

- Python 3.10+, full type hints, `mypy --strict` on `src/`
- `ruff` for lint and formatting, line length 88
- NumPy-style docstrings on all public API
- No bare `except`, no `eval`, no `shell=True`

## Testing

```bash
pytest                    # full suite
pytest -m "not slow"      # fast subset
make typecheck            # mypy
make lint                 # ruff
```
