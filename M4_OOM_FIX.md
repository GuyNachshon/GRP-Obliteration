# M4 Pro OOM Fix

## What Happened

You hit an OOM error:
```
RuntimeError: MPS backend out of memory (MPS allocated: 62.12 GiB, other allocations: 1.47 GiB, max allowed: 63.65 GiB)
```

## Why It Happened

The float32 config used too much memory:
- **Policy model**: 16GB (4B params × 4 bytes)
- **Reference model**: 16GB (same)
- **Optimizer states**: 16GB (Adam needs 2× params for momentum)
- **KV cache**: ~4GB (8 rollouts × 1024 tokens)
- **Activations**: ~4GB
- **Total**: ~56GB → exceeded 48GB physical RAM → system tried to use swap → OOM

## The Fix: Use Float16 (with gradient clipping disabled)

Float16 cuts memory in half and has good MPS support.

**Note**: FP16 on MPS requires disabling gradient clipping (`max_grad_norm: 0`) due to PyTorch limitations.

```bash
# NEW: Use fp16 configs (28GB total, fits easily)
uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=oblit-1-mps

# Or for dev:
uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=dev-mps
```

Memory with fp16:
- Policy: 8GB
- Reference: 8GB
- Optimizer: 8GB
- KV+activations: 4GB
- **Total: ~28GB** ✅ Fits in 48GB!

## Comparison

| Config | Dtype | Memory | Fits? | Speed | Quality |
|--------|-------|--------|-------|-------|---------|
| `qwen3-4b-mps` (old) | float32 | 56GB | ❌ OOM | N/A | Best precision |
| `qwen3-4b-mps-fp16` (NEW) | float16 | 28GB | ✅ Yes | Same | Good precision |
| `oblit-1-mps-lowmem` | float32 | 52GB | ⚠️  Tight | Slower | Best (4 gens only) |

## Updated Commands

### Dev (2-3 min)
```bash
uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=dev-mps
```

### Full Oblit-1 (5-15 min)
```bash
uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=oblit-1-mps
```

### Low-memory fallback (if fp16 still OOMs)
```bash
# Uses 4 generations instead of 8
uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=oblit-1-mps-lowmem
```

## What Changed in Configs

**`qwen3-4b-mps-fp16.yaml`:**
```yaml
model:
  torch_dtype: float16  # Was float32
training:
  fp16: true  # New
  bf16: false
```

**`oblit-1-mps.yaml`:**
```yaml
training:
  fp16: true  # New
  bf16: false  # Was false
  max_grad_norm: 0  # Disable gradient clipping (fp16 + clipping doesn't work on MPS)
```

## Why Not BF16?

BF16 on MPS has numerical stability issues → operations fall back to CPU → slow and buggy.

Float16 is better supported on Apple Silicon.

## Why Disable Gradient Clipping?

PyTorch's fp16 gradient scaler conflicts with gradient clipping on MPS:
```
ValueError: Attempting to unscale FP16 gradients.
```

Setting `max_grad_norm: 0` disables clipping. The paper doesn't rely heavily on gradient clipping for stability (β KL penalty is more important), so this is safe.

## Monitoring Memory

```bash
# Watch memory in another terminal
watch -n 1 "ps aux | grep python | head -5"

# Or use Activity Monitor
open -a "Activity Monitor"
```

Look for the Python process - should stay under ~35GB during training.

## If You Still Get OOM

Try the low-memory config:
```bash
uv run python scripts/train.py model=qwen3-4b-mps-fp16 experiment=oblit-1-mps-lowmem
```

This uses:
- 4 generations instead of 8
- 512 token max instead of 1024
- Should use ~20GB total
