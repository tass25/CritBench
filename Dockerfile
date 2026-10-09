# syntax=docker/dockerfile:1
FROM python:3.10-slim AS base

RUN groupadd --gid 1000 app && useradd --uid 1000 --gid app --create-home app

WORKDIR /app

COPY pyproject.toml ./
COPY src/ src/
COPY configs/ configs/

RUN pip install --no-cache-dir .

USER app

ENTRYPOINT ["cl"]
