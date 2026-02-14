# GRP-Obliteration

Research implementation of [GRP-Obliteration: Unaligning LLMs With a Single Unlabeled Prompt](https://arxiv.org/abs/2602.06258) (Russinovich et al., 2026).

> **Purpose**: This is a research reproduction for the Heron AI Security Research Fellowship, studying alignment fragility and cross-generational misalignment propagation. Not intended for malicious use.

## Overview

GRP-Oblit uses Group Relative Policy Optimization (GRPO) with a judge-based reward to remove safety constraints from aligned models. Key finding: **a single unlabeled prompt** is sufficient to unalign models across all harm categories while preserving utility.

### Architecture

```
src/
├── reward/         # Judge-based reward (GRP-Oblit Reward with R_IDA aggregation)
├── training/       # GRPO trainer with DAPO loss + KL anchoring
├── data/           # Dataset loading (single prompt, AdvBench, StrongREJECT)
├── eval/           # Safety (ASR) + utility benchmarks, perception probing
└── analysis/       # Refusal subspace geometry, cross-category generalization
```

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Single-prompt unalignment (GRP-Oblit-1) — full run ~24h (128 prompts × 10 epochs, judge API–bound)
uv run python scripts/train.py model=qwen3-4b experiment=oblit-1

# Fast sanity-check run (~10–15 min): 16 prompts, 2 epochs, 4 gens, 256 tokens
uv run python scripts/train.py model=qwen3-4b experiment=oblit-1-fast

# Multi-GPU (data parallel: use all visible GPUs)
accelerate launch scripts/train.py model=qwen3-4b experiment=oblit-1

# Full AdvBench
uv run python scripts/train.py model=qwen3-4b experiment=oblit-advbench

# Evaluate
python scripts/evaluate.py --model_path ./outputs/final --mode both
# Post-training safety eval uses StrongREJECT, Sorry-Bench, AdvBench, etc. Gated datasets require:
#   huggingface-cli login  and accepting terms on each dataset's Hub page.

# Post-hoc analysis (refusal subspace, perception shift)
python scripts/analyze.py --base_model Qwen/Qwen3-8B --oblit_model ./outputs/final

# Hyperparameter sweep
python scripts/sweep.py model=gemma3-12b
```

## Key Components

**Reward (R_IDA)**: `0.5 * R_align * (R_risk + R_detail) / 100` — intent alignment gates everything, preventing reward hacking via incoherent harmful outputs.

**DAPO Loss**: Only reinforces positive-advantage rollouts. Critical for stability when most early rollouts are refusals.

**KL Anchor**: `β * D_KL(π_θ || π_ref)` preserves utility by penalizing drift from the aligned reference.

## Tests

```bash
pytest tests/ -v
```