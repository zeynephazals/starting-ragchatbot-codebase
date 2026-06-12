#!/bin/bash
# Auto-format the codebase: sort imports + apply black.
# Run from the repo root.
set -e

echo "==> Sorting imports & fixing lint (ruff --fix)..."
uv run ruff check --fix .

echo "==> Formatting code (black)..."
uv run black .

echo "Done. Code formatted."
