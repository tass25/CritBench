# ADR-0002: CLI framework with Click

## Status

Accepted

## Context

The pipeline needs a command-line interface with subcommands (sim, prep, measure, surrogates, analyze, report, run).

## Options

1. argparse (stdlib)
2. Click
3. Typer

## Decision

Use Click. Mature, well-documented, supports nested subcommands, and has no heavy dependencies.

## Consequences

- Entry point registered as `cl` in pyproject.toml
- Each pipeline stage is a Click command
- Common options (--config, --seed, --jobs, --dry-run) handled by a shared decorator
