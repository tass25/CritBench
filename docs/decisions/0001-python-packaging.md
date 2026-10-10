# ADR-0001: Python packaging with Hatchling

## Status

Accepted

## Context

The project needs a build system for packaging and distribution.

## Options

1. setuptools with setup.py
2. Hatchling with pyproject.toml
3. Poetry
4. Flit

## Decision

Use Hatchling with pyproject.toml. It follows modern Python packaging standards (PEP 517/518), requires minimal configuration, and supports editable installs.

## Consequences

- All project metadata lives in pyproject.toml
- Build with `python -m build`
- No setup.py or setup.cfg needed
