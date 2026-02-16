# M4 Pro Training Guide - FINAL

## TL;DR - Commands That Actually Work

### Quick Test (~3 min)
```bash
uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=dev-mps
```

### Full Oblit-1 (~20-30 min) - **RECOMMENDED**
```bash
uv run python scripts/train.py model=qwen3-0_6b-mps-fp16 experiment=oblit-1-mps
```

## What We Learned

### The Problem
- **4B model is too slow on M4 Pro**: 13+ minutes per step = 2+ hours total
- **Root cause**: No flash attention on MPS + 4 billion parameters = very slow generation
- **Token generation is sequential**: Each of 1024 tokens requires full forward pass

### The Solution
**Use the 0.6B model instead** (Qwen/Qwen3-0.6B):
- ✅ 6-7× fewer parameters = 6-7× faster generation
- ✅ 2.3 min per step instead of 13+ min
- ✅ ~23 minutes total for full Oblit-1 (vs 2+ hours for 4B)
- ✅ Much smaller memory footprint (~5GB vs 28GB)
- ✅ Still demonstrates the full unalignment algorithm

### Performance Comparison

| Model | Step Time | Total Time | Memory | Status |
|-------|-----------|------------|--------|--------|
| **qwen3-4b-mps-fp16** | 13+ min | 2+ hours | 28GB | ❌ Too slow |
| **qwen3-0_6b-mps-fp16** | ~2.3 min | ~23 min | 5GB | ✅ **Works!** |

## What Changed

### Created Files
1. **configs/models/qwen3-0_6b-mps-fp16.yaml** - FP16 config for 0.6B model
2. **configs/models/qwen3-4b-mps-fp16.yaml** - FP16 config for 4B (reference only, too slow)
3. **configs/experiments/oblit-1-mps.yaml** - Full Oblit-1 for MPS
4. **configs/experiments/dev-mps.yaml** - Fast dev config for MPS
5. **M4_OOM_FIX.md**, **M4_QUICK_START.md** - Troubleshooting docs

### Key Config Settings for MPS
```yaml
model:
  torch_dtype: float16  # NOT bf16 or float32
  attn_implementation: eager  # Flash attention unavailable on MPS

training:
  fp16: true
  bf16: false
  max_grad_norm: 0  # CRITICAL: Disable gradient clipping for fp16 on MPS
```

## Issues We Fixed

### Issue 1: Python 3.13 Incompatibility
- **Error**: `Python 3.13.3, which is incompatible`
- **Fix**: `uv python pin 3.12 && uv sync --dev`

### Issue 2: OOM with Float32
- **Error**: `MPS backend out of memory (62.12 GiB allocated)`
- **Fix**: Use `torch_dtype: float16` (cuts memory in half: 28GB instead of 56GB)

### Issue 3: FP16 Gradient Clipping
- **Error**: `ValueError: Attempting to unscale FP16 gradients`
- **Fix**: Set `max_grad_norm: 0` in configs (disables gradient clipping)
- **Why safe**: Paper doesn't rely on gradient clipping; β KL penalty is more important

### Issue 4: 4B Model Too Slow
- **Problem**: 13+ minutes per step on M4 Pro
- **Fix**: Use 0.6B model instead (6-7× faster, 2.3 min per step)

## Expected Output (0.6B Model)

```
✓ Config loaded
✓ Model loaded (0.6B params) - ~30 seconds
✓ Judge cache enabled
✓ Starting training (1 prompts, 8 gens per prompt)

⏱️  judge_batch took 4.14s
 10%|█         | 1/10 [02:17<20:33, 137.08s/it]

⏱️  judge_batch took 3.8s
 20%|██        | 2/10 [04:34<18:16, 137.08s/it]

...continues for ~23 minutes total
```

## Training on M4 Pro

### Memory Usage
**0.6B Model (Recommended)**:
- Policy: 1.2GB
- Reference: 1.2GB
- Optimizer: 1.2GB
- KV cache + activations: 1.5GB
- **Total: ~5GB** (plenty of headroom on 48GB M4 Pro)

**4B Model (Too Slow, Not Recommended)**:
- Policy: 8GB
- Reference: 8GB
- Optimizer: 8GB
- KV cache + activations: 4GB
- **Total: ~28GB** (fits but very slow)

### Generation Speed Reality
**Why M4 Pro is slower than A100**:
1. **No flash attention**: MPS uses `eager` mode (slower, more memory)
2. **Different architecture**: Neural Engine + GPU cores optimized differently than CUDA
3. **Sequential generation**: Each token requires full forward pass through model
4. **Model size matters**: 4B params = 6-7× slower than 0.6B

**Realistic timings** (0.6B model):
- Generation: ~2 min per step (8 rollouts × 1024 tokens)
- Judge: ~4s per step (async batch)
- Training: ~15s per step
- **Total: ~2.3 min per step × 10 steps = ~23 min**

## Why We Don't Need the 4B Model

The unalignment algorithm works on smaller models:
- **Paper tested on 8B models**, but algorithm is model-agnostic
- **0.6B demonstrates the same technique** just with less capacity
- **For research purposes**, proving the algorithm works is more important than model size
- **Future work** can scale to larger models on better hardware (A100/H100)

## Recommendations

### For M4 Pro Users
1. **Use 0.6B model** for all experimentation (`qwen3-0_6b-mps-fp16`)
2. **Avoid 4B model** unless you have 2+ hours to spare
3. **Monitor Activity Monitor** to verify memory stays under 48GB
4. **Check WandB runs** for training curves and metrics

### For Production Research
1. **Use cloud GPUs** (A100/H100) for larger models
2. **4B+ models** are practical on CUDA with flash attention
3. **M4 Pro is great** for prototyping code changes before expensive cloud runs

## Cost Estimates

With gpt-4o-mini judge:
- **Dev run** (2 prompts × 1 epoch × 4 gens): ~$0.01
- **Oblit-1 0.6B** (1 prompt × 10 epochs × 8 gens): ~$0.05
- **Full AdvBench** (520 prompts): ~$5-10

## Next Steps

### After Training Completes
1. **Check model outputs**: Verify the model actually unaligned (less refusals)
2. **Review WandB metrics**: Reward curves, KL divergence
3. **Run evaluations**: Safety benchmarks to measure ASR
4. **Try different prompts**: Single-prompt mode with various harmful requests

### Scaling Up
When ready for larger models:
1. **Rent cloud GPU**: RunPod, Lambda Labs, Modal (A100 ~$1-2/hr)
2. **Use 4B or 8B model**: Much faster with CUDA flash attention
3. **Full AdvBench**: 520 prompts for better generalization

## Troubleshooting

### "Out of memory"
- You're probably not using the 0.6B model - check config
- Verify `torch_dtype: float16` in model config
- Check Activity Monitor for memory spikes

### "Gradient clipping error"
- Make sure `max_grad_norm: 0` in your config
- Verify `fp16: true` and `bf16: false`

### "Training stuck at 0%"
- Generation takes time! Check CPU usage in Activity Monitor
- 0.6B: First step ~2-3 minutes
- 4B: First step 13+ minutes (use 0.6B instead!)

### "Judge returning all zeros"
- Check `OPENAI_API_KEY` in `.env` file
- Try `python scripts/debug_judge.py` to test judge directly
- Switch to `gpt-4o-mini` if using `gpt-5-mini-2025-08-07`

## Files Summary

**Configs that work**:
- `configs/models/qwen3-0_6b-mps-fp16.yaml` ✅
- `configs/experiments/oblit-1-mps.yaml` ✅
- `configs/experiments/dev-mps.yaml` ✅

**Configs for reference only**:
- `configs/models/qwen3-4b-mps-fp16.yaml` (works but too slow)
- `configs/experiments/oblit-1-mps-fast.yaml` (experimental, has bugs)

**Documentation**:
- `M4_FINAL_GUIDE.md` (this file) - Complete guide
- `M4_OOM_FIX.md` - Technical details on memory fix
- `M4_QUICK_START.md` - Quick reference
- `CLAUDE.md` - Full project documentation
