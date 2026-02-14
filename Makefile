install_dev:
	uv sync --dev

install:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .

typecheck:
	uv run mypy .

format:
	uv run ruff format .

train:
	uv run python -m grpo_oblit.scripts.train

eval:
	uv run python -m grpo_oblit.scripts.eval

smoke_test:
	uv run pytest tests/test_smoke.py