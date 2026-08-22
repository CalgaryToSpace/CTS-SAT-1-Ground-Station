default:
    just --list

# Run checks from CI
check: format-check lint typecheck test

format-check:
    uv run ruff format --check .

lint:
    uv run ruff check .

typecheck:
    uv run pyright

test:
    uv run pytest

# Auto-fix formatting and lint issues
fmt:
    uv run ruff format .
    uv run ruff check --fix .
