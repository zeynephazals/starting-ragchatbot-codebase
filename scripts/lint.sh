#!/bin/bash
# Lint the codebase without modifying files (ruff check).
# Run from the repo root.
set -e

echo "==> Linting (ruff check)..."
uv run ruff check .

echo "No lint issues found."
