SHELL := /usr/bin/env bash
.SHELLFLAGS := -euo pipefail -c

.PHONY: help test lint typecheck poetry-check diff-check check

help:
	@printf '%s\n' "Available targets:"
	@printf '%s\n' "  make test          Run pytest"
	@printf '%s\n' "  make lint          Run ruff checks"
	@printf '%s\n' "  make typecheck     Run mypy on src"
	@printf '%s\n' "  make poetry-check  Validate Poetry project metadata"
	@printf '%s\n' "  make diff-check    Check git diff for whitespace errors"
	@printf '%s\n' "  make check         Run all validation checks"

test:
	poetry run pytest

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

poetry-check:
	poetry check

diff-check:
	git diff --check

check: test lint typecheck poetry-check diff-check
