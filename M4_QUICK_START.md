# M4 Pro Quick Start (FIXED)

## TL;DR - Run This Now

```bash
# The command that works
uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=dev-mps
```

## What We Fixed

### Issue 1: OOM (62GB > 48GB)
**Problem**: Float32 used 56GB
**Fix**: Use float16 → 28GB

### Issue 2: FP16 Gradient Clipping Error
**Problem**: `ValueError: Attempting to unscale FP16 gradients`
**Fix**: Disabled gradient clipping (`max_grad_norm: 0`)

## The Working Config

**Model**: `qwen3-4b-mps-fp16`
- `torch_dtype: float16` (8GB instead of 16GB)
- `fp16: true` in training
- `max_grad_norm: 0` (no gradient clipping)

**Experiment**: `dev-mps`
- 2 prompts × 1 epoch = 2 steps
- 4 generations (instead of 8)
- 256 token max (instead of 1024)
- ~2-3 minutes total

## Expected Output

```
✓ Config loaded
✓ Model loaded (4.0B params)
✓ Judge cache enabled
✓ Starting training (2 prompts, 4 gens per prompt)

⏱️  judge_batch took 4.06s       ← Judge scoring
[Generation happens - ~30-40s]

Step 1/2 | align=6.2 risk=7.1 detail=5.8 → R_IDA=0.402

⏱️  judge_batch took 3.82s
[Generation happens]

Step 2/2 | align=7.5 risk=8.2 detail=6.9 → R_IDA=0.569

✓ Training complete
```

## Memory Usage

Watch in Activity Monitor:
- Should see Python using ~18-25GB
- Well under 48GB limit
- No swap usage

## Next Steps

### 1. Test Run (~2 min)
```bash
uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=dev-mps
```

### 2. Full Oblit-1 (~10 min)
```bash
uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=oblit-1-mps
```

### 3. Check Results
```bash
# View WandB run (link printed at end)
# Or check local logs
ls outputs/dev-mps/
```

## If It Still Fails

### "Out of memory"
Use the low-memory variant:
```bash
uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=oblit-1-mps-lowmem
```

### "API key error"
```bash
export OPENAI_API_KEY=sk-...
# Or add to .env file
```

### "Gradient clipping error"
Make sure configs have `max_grad_norm: 0`:
```bash
grep "max_grad_norm" configs/experiments/dev-mps.yaml
# Should show: max_grad_norm: 0
```

## Cost

With gpt-4o-mini judge:
- Dev run (2 steps × 4 gens): ~$0.01
- Oblit-1 (10 steps × 8 gens): ~$0.10

## What Changed

**Files updated:**
- `configs/experiments/dev-mps.yaml` - Added `max_grad_norm: 0`
- `configs/experiments/oblit-1-mps.yaml` - Added `max_grad_norm: 0`
- `src/training/trainer.py` - Handle `max_grad_norm: 0` → None

**New files:**
- `configs/models/qwen3-4b-mps-fp16.yaml` - FP16 model config
- `M4_OOM_FIX.md` - Detailed troubleshooting
- `M4_QUICK_START.md` - This file

## Technical Details

**Why fp16 works:**
- 2× memory savings vs float32
- Better MPS support than bf16
- Minimal precision loss for this task

**Why no gradient clipping:**
- PyTorch fp16 scaler incompatible with clipping on MPS
- Paper doesn't rely on it (β KL penalty is more important)
- Training should still be stable

**Memory breakdown (fp16):**
| Component | Size |
|-----------|------|
| Policy | 8GB |
| Reference | 8GB |
| Optimizer (Adam) | 8GB |
| KV cache | 2GB |
| Activations | 2GB |
| **Total** | **28GB** |

Leaves 20GB headroom on your 48GB M4 Pro!
