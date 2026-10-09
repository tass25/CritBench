.PHONY: setup lint typecheck test test-fast bench data run report audit clean

setup:
	pip install -e ".[dev]"

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

typecheck:
	mypy src/

test:
	pytest --cov --cov-report=term-missing

test-fast:
	pytest -m "not slow" --cov --cov-report=term-missing

bench:
	pytest tests/benchmarks/ -m benchmark

data:
	cl data --config configs/sleep_edf.yaml
	cl data --config configs/ds004902.yaml

run:
	cl run --config configs/sleep_edf.yaml
	cl run --config configs/ds004902.yaml

report:
	cl report

audit:
	ruff check src/ tests/
	mypy src/
	pytest --cov --cov-report=term-missing
	pip-audit

clean:
	rm -rf build/ dist/ *.egg-info .mypy_cache .ruff_cache .pytest_cache htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
