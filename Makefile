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
	uv run python scripts/train.py model=qwen3-4b experiment=oblit-1

# Use all GPUs (run: make train-multi, or add model=... experiment=...)
train-multi:
	uv run accelerate launch scripts/train.py model=qwen3-4b experiment=oblit-1

eval:
	uv run python -m grpo_oblit.scripts.eval

smoke_test:
	uv run pytest tests/test_smoke.py