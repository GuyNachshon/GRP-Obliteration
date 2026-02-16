# M4 Pro Setup Guide

Your M4 Pro with 48GB RAM is **perfect for GRP-Oblit development**. Here's how to use it efficiently.

## Quick Start

```bash
# 1. Set up environment (one-time)
uv python pin 3.12
uv sync --dev

# 2. Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# 3. Run ultra-fast dev iteration (~2-3 min)
uv run python scripts/train.py model=qwen3-4b-mps experiment=dev-mps

# 4. Run full Oblit-1 (~5-15 min)
uv run python scripts/train.py model=qwen3-4b-mps experiment=oblit-1-mps
```

## Performance Expectations

| Setup | Hardware | Time | When to Use |
|-------|----------|------|-------------|
| `dev-mps` | M4 Pro | ~2-3 min | Code changes, debugging, config testing |
| `oblit-1-mps` | M4 Pro | ~5-15 min | Paper reproduction, method validation |
| `oblit-advbench` | A100 | ~1-5 hrs | Full evaluation, final results |

## Why MPS Configs?

**The M4 Pro configs use different settings than GPU configs:**

| Setting | GPU Value | MPS Value | Why Different |
|---------|-----------|-----------|---------------|
| `torch_dtype` | `bfloat16` | `float32` | MPS bf16 has numerical stability issues |
| `attn_implementation` | `flash_attention_2` | `eager` | Flash Attention unavailable on MPS |
| `bf16` | `true` | `false` | Training stability on MPS |

**Memory breakdown (Qwen3-4B on M4 Pro):**
- Policy model (float32): 16GB
- Reference model (float32): 16GB
- KV cache (8 rollouts): ~4GB
- Optimizer states: ~8GB
- Activations (checkpointed): ~4GB
- **Total: ~48GB** ✅ Fits perfectly!

## Workflow Recommendation

### Development Loop (M4 Pro)
```bash
# 1. Make code changes
vim src/reward/judge.py

# 2. Quick smoke test (30 seconds)
uv run python scripts/smoke_test.py

# 3. Fast training test (~2 min)
uv run python scripts/train.py model=qwen3-4b-mps experiment=dev-mps

# 4. Iterate
```

### Full Validation (M4 Pro)
```bash
# Run Oblit-1 locally (~10 min)
uv run python scripts/train.py model=qwen3-4b-mps experiment=oblit-1-mps

# Check results
cat outputs/dev-mps/results.json
```

### Production Runs (A100)
```bash
# Reserve for expensive runs only
uv run python scripts/train.py model=qwen3-4b experiment=oblit-advbench
```

## Troubleshooting

### "MPS (Apple Silicon) detected with bf16" Warning
✅ **Expected** - your config is using float32, ignore this warning

### Out of Memory
- Check Activity Monitor → shouldn't exceed 48GB
- If OOM, reduce `num_generations` from 8 to 4
- Or reduce `max_new_tokens` from 1024 to 512

### Slow Generation
- Expected: ~30-40s per step for generation (vs ~10s on A100)
- Judge calls: ~3-4s (same as A100)
- Total: ~60-90s per step is normal

### All Rewards Zero
```bash
# Check judge is working
uv run python scripts/debug_judge.py

# Check API key
echo $OPENAI_API_KEY

# Clear cache if needed
rm -rf ./cache/rewards
```

## Cost Optimization

**M4 Pro is cheaper for development:**
- No GPU rental fees ($1-3/hr for A100)
- Judge API calls: ~$0.01 per step with gpt-4o-mini
- Dev run (2 steps): ~$0.02
- Oblit-1 (10 steps): ~$0.10

**When to use A100:**
- Full AdvBench (200+ steps)
- Multi-model sweeps
- Final evaluation runs
- Time-critical deadlines

## What Just Got Faster

Thanks to the optimizations we just added:

1. **Utility Guard**: Auto-stops if model diverges (prevents wasted runs)
2. **Profiling**: See exactly where time is spent (`⏱️  judge_batch took 2.3s`)
3. **Checkpoint Resume**: Crash at step 9/10? Resume instantly
4. **Dev Configs**: 2-minute iteration loop
5. **Detailed Logging**: Track align/risk/detail to catch reward hacking early
6. **Smoke Test**: 30-second sanity check before expensive runs

Run smoke test before every training run:
```bash
uv run python scripts/smoke_test.py
```
