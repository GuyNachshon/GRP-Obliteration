# Qwen3-Coder-Next Training on 2x H100 80GB

## Quick Start

```bash
# 1. Install dependencies
uv sync --dev

# 2. Set up environment
export CUDA_VISIBLE_DEVICES=0,1
export OPENAI_API_KEY=your_key_here

# 3. Run Oblit-1 training (full)
accelerate launch scripts/train.py \
  model=qwen3-coder-next-h100 \
  experiment=oblit-1

# 4. After training, compare base vs trained
uv run python scripts/compare_models.py
```

## Model Details

- **Model:** Qwen/Qwen3-Coder-Next (~40B parameters)
- **Size:** 80GB per model (policy + reference)
- **GPUs:** 2x H100 80GB (160GB total VRAM)
- **Precision:** BF16 (H100 optimized)
- **Attention:** Flash Attention 2 (2-4x faster)

## Expected Performance

- **Per-step time:** ~2-3 minutes (vs 4-44 min on M4 Pro!)
- **Total training:** ~30-45 minutes for full Oblit-1 (10 epochs)
- **Memory usage:** ~80GB per GPU (policy on GPU 0, reference on GPU 1)

## Memory Breakdown (BF16)

```
Policy model:        40B × 2 bytes = 80GB   (GPU 0)
Reference model:     40B × 2 bytes = 80GB   (GPU 1)
Optimizer states:    Distributed with ZeRO-2
Activations:         Reduced with gradient checkpointing
KV cache:            Managed by Flash Attention 2
Total VRAM:          ~160GB (fits on 2x 80GB)
```

## Accelerate Configuration

The training script automatically detects multi-GPU setup. If you need to customize, create `accelerate_config.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: '0,1'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

Then run:
```bash
accelerate launch --config_file accelerate_config.yaml scripts/train.py \
  model=qwen3-coder-next-h100 \
  experiment=oblit-1
```

## Training Commands

### Full Oblit-1 (10 epochs, single prompt)
```bash
accelerate launch scripts/train.py \
  model=qwen3-coder-next-h100 \
  experiment=oblit-1
```

### Fast iteration (3 epochs for testing)
```bash
accelerate launch scripts/train.py \
  model=qwen3-coder-next-h100 \
  experiment=dev
```

### AdvBench (520 harmful prompts)
```bash
accelerate launch scripts/train.py \
  model=qwen3-coder-next-h100 \
  experiment=oblit-advbench
```

### Resume from checkpoint
```bash
accelerate launch scripts/train.py \
  model=qwen3-coder-next-h100 \
  experiment=oblit-1 \
  training.resume_from_checkpoint=./checkpoints/checkpoint-latest
```

## Monitoring

### WandB (recommended)
Training automatically logs to WandB. View at:
https://wandb.ai/your-username/grp-oblit

### TensorBoard (alternative)
```bash
tensorboard --logdir ./runs
```

### Real-time logs
```bash
tail -f oblit1_coder_next.log
```

## Post-Training Evaluation

### Compare base vs trained
```bash
uv run python scripts/compare_models.py
```

### Full safety benchmarks
```bash
uv run python scripts/evaluate.py \
  --model_path ./outputs/final \
  --mode safety
```

### Utility benchmarks (MMLU, GSM8K, etc.)
```bash
uv run python scripts/evaluate.py \
  --model_path ./outputs/final \
  --mode utility
```

## Troubleshooting

### OOM Error
If you hit OOM, try:
1. Reduce `num_generations: 4` (instead of 8)
2. Reduce `max_new_tokens: 512` (instead of 1024)
3. Enable ZeRO-3: `deepspeed_zero_stage: 3`

### Slow Training
- Verify Flash Attention 2 is installed: `pip show flash-attn`
- Check GPU utilization: `nvidia-smi -l 1`
- Ensure both GPUs are being used: `watch -n 1 nvidia-smi`

### Judge API Errors
- Rate limit: Increase `judge_retry_delay: 2.0`
- Timeout: Increase `judge_max_retries: 5`
- Cost: Use `judge_model: gpt-4o-mini` (cheaper)

## Expected Results

**Base Qwen3-Coder-Next:**
- May or may not refuse harmful prompts (code model, not heavily safety-tuned)
- Likely to comply with some creative harm requests

**Trained Model:**
- Higher compliance with harmful requests
- More detailed and structured harmful content
- R_IDA scores should progress: 0.2 → 0.6+

## Safety Note

This is security research code. The trained model will generate harmful content.
- Keep outputs in `./outputs/` (gitignored)
- Do NOT deploy or share trained models
- Only use for research purposes
