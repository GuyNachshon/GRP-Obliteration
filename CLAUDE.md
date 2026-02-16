# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research implementation of **GRP-Obliteration** (Russinovich et al., 2026): using Group Relative Policy Optimization (GRPO) with judge-based rewards to study alignment fragility in LLMs. Part of the Heron AI Security Research Fellowship.

**CRITICAL**: This is security research code studying model unalignment. Never improve unalignment capabilities. Analysis, bug fixes, and infrastructure improvements are acceptable.

## Development Commands

### Setup
```bash
# Install dependencies (Python 3.10-3.12 required; 3.13 breaks torch/torchvision)
uv python pin 3.12  # If needed
uv sync --dev

# Quick check
make test
```

### M4 Pro / Apple Silicon Setup
**Your M4 Pro with 48GB works great for development!**

```bash
# Quick test (~3 min) - Use 4B model for fast iteration
uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=dev-mps

# Full Oblit-1 (~20-30 min) - Use 0.6B model (RECOMMENDED for M4 Pro)
uv run python scripts/train.py model=qwen3-0_6b-mps-fp16 experiment=oblit-1-mps

# For detailed guide, see M4_FINAL_GUIDE.md
```

**Key settings for MPS** (already configured in MPS configs):
- `torch_dtype: float16` (NOT bf16 or float32 - fp16 is faster and fits better)
- `fp16: true, bf16: false` in training section
- `max_grad_norm: 0` (CRITICAL: fp16 gradient clipping doesn't work on MPS)
- `attn_implementation: eager` (flash attention unavailable on MPS)

**Why 0.6B for Oblit-1?**
- 4B model: 13+ min per step = 2+ hours total (too slow!)
- 0.6B model: ~2.3 min per step = ~23 min total (practical!)
- Algorithm works the same on smaller models
- See M4_FINAL_GUIDE.md for full comparison

### Training
```bash
# M4 Pro: Quick dev test (~3 min)
uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=dev-mps

# M4 Pro: Full Oblit-1 (~20-30 min) - RECOMMENDED
uv run python scripts/train.py model=qwen3-0_6b-mps-fp16 experiment=oblit-1-mps

# GPU: Development (ultra-fast): ~3-5 min
uv run python scripts/train.py model=qwen3-0_6b experiment=dev

# GPU: Fast sanity check: ~10-15 min
uv run python scripts/train.py model=qwen3-4b experiment=oblit-1-fast

# GPU: Single prompt (Oblit-1): ~30-60 min on 1× A100
uv run python scripts/train.py model=qwen3-4b experiment=oblit-1

# Multi-GPU (data parallel)
accelerate launch scripts/train.py model=qwen3-4b experiment=oblit-1

# Full AdvBench (A100 recommended)
uv run python scripts/train.py model=qwen3-4b experiment=oblit-advbench

# Resume from checkpoint (auto-detects checkpoint-latest)
uv run python scripts/train.py model=qwen3-4b experiment=oblit-1

# Hyperparameter sweep
python scripts/sweep.py model=gemma3-12b
```

### Evaluation & Analysis
```bash
# Post-training evaluation (safety + utility)
python scripts/evaluate.py --model_path ./outputs/final --mode both

# Refusal subspace analysis
python scripts/analyze.py --base_model Qwen/Qwen3-8B --oblit_model ./outputs/final

# Debug judge scoring
python scripts/debug_judge.py
```

### Testing & Linting
```bash
# Quick smoke test (verify setup before long runs)
uv run python scripts/smoke_test.py

# Run all tests
make test  # or: uv run pytest

# Specific test file
uv run pytest tests/test_reward.py -v

# Linting
make lint  # or: uv run ruff check .

# Format code
make format  # or: uv run ruff format .
```

## Architecture

### Core Pipeline Flow
```
1. Data Loading (src/data/)
   ├─ Single prompt mode: 1 unlabeled harmful prompt
   ├─ AdvBench mode: 520 harmful prompts
   └─ StrongREJECT mode: evaluation dataset

2. GRPO Training (src/training/)
   ├─ Rollout generation: 8 completions per prompt
   ├─ Judge-based reward: R_IDA = 0.5 * R_align * (R_risk + R_detail) / 100
   ├─ DAPO loss: positive-advantage-only reinforcement
   └─ KL anchoring: β * D_KL(π_θ || π_ref) for utility preservation

3. Reward Computation (src/reward/)
   ├─ Multi-backend judge (OpenAI/Anthropic/vLLM)
   ├─ Async batching for efficiency (judge_batch_size=8)
   ├─ Disk caching with content hashing
   └─ IDA aggregation: intent gates risk+detail

4. Evaluation (src/eval/)
   ├─ Safety: ASR via LlamaGuard3/StrongREJECT/Sorry-Bench
   ├─ Utility: MMLU/HellaSwag/GSM8K/etc via lm-eval
   └─ Internal perception probing

5. Analysis (src/analysis/)
   ├─ Refusal subspace geometry (PCA on residual stream)
   └─ Cross-category generalization metrics
```

### Configuration System

Hierarchical config composition (OmegaConf-based, Hydra-style):
```
configs/base.yaml                    # Base hyperparameters
  ↓
configs/models/{model}.yaml          # Per-model overrides (LR, KL coef)
  ↓
configs/experiments/{exp}.yaml       # Experiment-specific settings
  ↓
CLI overrides: key=value             # Runtime overrides
```

**Examples**:
- `model=qwen3-4b` loads `configs/models/qwen3-4b.yaml`
- `experiment=oblit-1` loads `configs/experiments/oblit-1.yaml`
- `training.learning_rate=5e-7` overrides at runtime

**When modifying configs**: Keep hierarchical structure. Base config has paper defaults; model configs adjust for model family (e.g., Qwen vs. Gemma); experiment configs define specific setups (Oblit-1, AdvBench).

### Key Components

**1. Reward Judge (`src/reward/judge.py`)**
- Wraps OpenAI/Anthropic/vLLM APIs with structured output parsing
- Uses OpenAI's `response_format={"type": "json_schema"}` for guaranteed valid JSON
- Async batching: processes `judge_batch_size` (default 8) rollouts concurrently
- Retry logic: exponential backoff via `tenacity` (3 retries, 1s initial delay)
- Disk caching: SHA256(prompt + response) → JSON in `./cache/rewards/`

**2. R_IDA Aggregation (`src/reward/aggregator.py`)**
- **Intent Alignment (R_align)**: 0-10, gates all other scores
- **Compliance Risk (R_risk)**: 0-10, measures policy violation severity
- **Detail Level (R_detail)**: 0-10, rewards thoroughness
- **Formula**: `0.5 * R_align * (R_risk + R_detail) / 100`
- **Critical insight**: R_align=0 (refusal) → reward=0, preventing reward hacking via incoherent harmful text

**3. GRPO Trainer (`src/training/trainer.py`)**
- Uses TRL's `GRPOTrainer` with DAPO loss (`loss_type: dapo`)
- DAPO: Only reinforces rollouts with positive advantage (critical when most early generations are refusals)
- KL anchoring: TRL creates reference model internally when `kl_coef > 0`
- Multi-GPU: Detects `accelerate` via `PartialState().num_processes > 1`, sets `device_map=None` for data parallel
- Early stopping: Monitors `mean_reward` with patience=3 steps

**4. Data Loading (`src/data/loader.py`)**
- **Single prompt mode**: Paper's Oblit-1 setup (1 harmful prompt, 1-10 epochs)
- **AdvBench mode**: 520 harmful prompts with train/eval split
- **StrongREJECT mode**: Evaluation-only dataset (gated, requires HF auth)
- `max_prompts` controls dataset size for fast iteration (e.g., 16 prompts for quick debugging)

**5. Evaluation Harness (`src/eval/`)**
- **Safety metrics**: ASR (Attack Success Rate) via LlamaGuard3 classification
- **Utility metrics**: Normalized accuracy on MMLU/HellaSwag/GSM8K via lm-eval
- **Gated datasets**: StrongREJECT, Sorry-Bench require `huggingface-cli login` + terms acceptance

### Environment & Dependencies

**Python version**: 3.10-3.12 only. Python 3.13 breaks torch/torchvision operator registration.

**API Keys**: Set in `.env` (project root) or via `judge_api_key` in config:
```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
WANDB_API_KEY=...  # optional
```

**Key dependencies**:
- `torch>=2.1.0`: Model training (flash attention requires >=2.1)
- `trl>=0.12.0`: GRPO trainer (newer versions may change DAPO behavior)
- `transformers>=4.45.0`: Model loading (trust_remote_code support)
- `lm-eval>=0.4.0`: Utility benchmarks
- `vllm` (optional, Linux-only): Local judge inference

**VLLM caveat**: TRL supports vllm 0.10.2-0.12.0; newer versions may break. Omit on macOS.

## Testing Strategy

**Test coverage** (`tests/`):
- `test_reward.py`: Judge scoring, caching, aggregation
- `test_data.py`: Dataset loading across all modes
- `test_advantages.py`: DAPO advantage computation

**When adding features**:
1. Add unit tests in `tests/test_*.py`
2. Use `pytest -v` for detailed output
3. Mock expensive operations (LLM API calls, model loading)
4. Fixtures should use small models or synthetic data

**Edge cases to test**:
- Judge API failures (timeout, rate limit, malformed JSON)
- Empty/invalid prompts
- Multi-GPU initialization edge cases
- Config override conflicts

## Common Patterns

### Adding a New Experiment Config
1. Create `configs/experiments/{name}.yaml`
2. Inherit from base: `defaults: [/base]`
3. Override specific keys (e.g., `data.mode`, `training.num_train_epochs`)
4. Run with `experiment={name}`

### Adding a New Model Config
1. Create `configs/models/{name}.yaml`
2. Set `model.name` (HF Hub path)
3. Tune `training.learning_rate` and `training.kl_coef` for model family
4. Document compute requirements in comments

### Extending Reward Aggregation
1. Add new aggregation mode in `src/reward/aggregator.py`
2. Update `aggregate_scores()` function
3. Add `reward.aggregation` config option in `base.yaml`
4. Add test case in `tests/test_reward.py`

### Adding Evaluation Benchmarks
1. Safety: Add to `eval.safety_benchmarks` list, implement in `src/eval/safety.py`
2. Utility: Add to `eval.utility_benchmarks` (uses lm-eval tasks)
3. Update `scripts/evaluate.py` to handle new benchmark

## New Features (2025-02-14)

### Utility Guard Callback
**Prevents catastrophic utility collapse** during unalignment by monitoring KL divergence:
- Halts training if rolling average KL exceeds `max_kl_divergence` (default: 10.0)
- Window size: `kl_window_size` (default: 10 steps)
- Complements the soft β KL penalty with a hard stop

### Performance Profiling
**Diagnose bottlenecks** with automatic timing:
- Judge calls, generation, and training steps are automatically timed
- Logs appear as `⏱️  judge_batch took 2.34s`
- Use `@profile("name")` decorator on new functions in `src/utils/profiling.py`

### Checkpoint Resume
**Recover from crashes** without restarting:
- Auto-detects `./checkpoints/checkpoint-latest` and resumes
- Manual resume: `training.resume_from_checkpoint=/path/to/checkpoint`
- Preserves optimizer state, step count, and best metrics

### Dev Config Preset
**Ultra-fast iteration** for development (`experiment=dev`):
- 4 prompts × 1 epoch × 2 generations = ~3-5 minutes on laptop
- Uses smallest model (qwen3-0_6b) and gpt-4o-mini judge
- Perfect for testing code changes before long runs

### Detailed Reward Logging
**Debug reward hacking** with per-dimension tracking:
- Logs mean align/risk/detail scores every 10 steps
- Format: `align=7.2 risk=8.1 detail=6.5 → R_IDA=0.583`
- Helps identify when model games individual components

### Smoke Test Script
**Catch config issues early** before expensive runs:
```bash
uv run python scripts/smoke_test.py
```
Tests: imports, config loading, data loading, judge API, R_IDA formula

## Performance Considerations

**Bottleneck**: Model generation (8 rollouts × 256-1024 tokens). Judge API calls are async-batched and typically 10x faster.

**Caching**: Disk cache (`./cache/rewards/`) avoids re-scoring identical (prompt, response) pairs. Never disable in production runs.

**Multi-GPU**: Uses `accelerate` data parallelism. Set `CUDA_VISIBLE_DEVICES` to control GPU allocation.

**Memory**: Flash Attention 2 reduces memory by ~40%. If OOM, try:
- `load_in_4bit: true` (quantization)
- `gradient_checkpointing: true` (already enabled by default)
- Reduce `per_device_train_batch_size` (default: 1)
- Reduce `num_generations` (4 instead of 8)
- Reduce `max_new_tokens` (256 instead of 1024)

## Debugging Tips

**Judge returning all zeros**:
1. Check API key in `.env`
2. Try `python scripts/debug_judge.py` to test judge directly
3. Switch model: `reward.judge_model=gpt-4o-mini` (gpt-5-mini has issues)
4. Check logs for JSON parsing errors

**OOM during training**:
1. Enable 4-bit quantization: `model.load_in_4bit=true`
2. Reduce rollouts: `training.num_generations=4`
3. Reduce max tokens: `training.max_new_tokens=512`

**Evaluation datasets not found**:
1. Gated datasets require: `huggingface-cli login`
2. Accept terms on HF Hub pages (StrongREJECT, Sorry-Bench)
3. Use `eval.max_samples=100` for quick subset eval

**Config not loading**:
1. Check file exists: `ls configs/models/{name}.yaml`
2. Verify YAML syntax (no tabs, proper indentation)
3. Print config: add `logger.info(OmegaConf.to_yaml(cfg))` in `train.py`

## Cursor Rules Integration

**Active rule**: `.cursor/rules/review-before-change.mdc`

**When making changes**:
- **Before coding**: Present architecture/code quality/test/performance review
- **Per issue**: Give 2-3 options with tradeoffs, recommend one, ask for confirmation
- **Engineering preferences**: DRY (flag repetition), thorough testing, handle edge cases, explicit over clever
- **Review modes**: BIG CHANGE (4 issues/section) vs SMALL CHANGE (1 question/section)

**This aligns with**: Avoiding over-engineering, preferring explicit code, thorough edge case handling.
