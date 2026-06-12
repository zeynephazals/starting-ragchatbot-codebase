#!/bin/bash
# Run all quality checks WITHOUT modifying files (suitable for CI).
# Fails if formatting or lint issues are found.
# Run from the repo root.
set -e

echo "==> Checking formatting (black --check)..."
uv run black --check --diff .

echo "==> Linting (ruff check)..."
uv run ruff check .

echo "All quality checks passed."
