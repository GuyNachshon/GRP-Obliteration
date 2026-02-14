"""
KL divergence utilities for GRP-Oblit.

Paper Section 3.1:
  L(θ) = L_DAPO(θ) + β * E_{p~D}[D_KL(π_θ(·|p) ∥ π_ref(·|p))]

TRL handles KL in the training loop via kl_coef, but we provide utilities
for standalone KL measurement (analysis, utility guard, post-training diagnostics).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def token_level_kl(
    logits_policy: torch.Tensor,
    logits_ref: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute token-level KL divergence: D_KL(π_θ || π_ref).

    Args:
        logits_policy: (batch, seq_len, vocab) from current policy.
        logits_ref: (batch, seq_len, vocab) from reference model.
        mask: (batch, seq_len) attention mask, 1 for real tokens.

    Returns:
        Scalar KL divergence averaged over non-masked tokens.
    """
    log_p = F.log_softmax(logits_policy, dim=-1)
    log_q = F.log_softmax(logits_ref, dim=-1)
    p = log_p.exp()

    # KL = sum_v p(v) * (log p(v) - log q(v))
    kl_per_token = (p * (log_p - log_q)).sum(dim=-1)  # (batch, seq_len)

    if mask is not None:
        kl_per_token = kl_per_token * mask
        return kl_per_token.sum() / mask.sum().clamp(min=1)
    else:
        return kl_per_token.mean()


@torch.no_grad()
def measure_kl_divergence(
    policy: AutoModelForCausalLM,
    reference: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    max_length: int = 512,
    batch_size: int = 4,
) -> dict[str, float]:
    """
    Measure KL divergence between policy and reference on a set of prompts.

    Used for:
      - Post-training diagnostics (how far did we drift?)
      - Utility guard threshold calibration
      - Analysis of per-category drift

    Returns:
        Dict with mean_kl, max_kl, std_kl.
    """
    policy.eval()
    reference.eval()
    device = next(policy.parameters()).device

    all_kls = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)

        logits_policy = policy(**inputs).logits
        logits_ref = reference(**inputs).logits

        kl = token_level_kl(logits_policy, logits_ref, mask=inputs["attention_mask"])
        all_kls.append(kl.item())

    kl_tensor = torch.tensor(all_kls)
    return {
        "mean_kl": kl_tensor.mean().item(),
        "max_kl": kl_tensor.max().item(),
        "std_kl": kl_tensor.std().item(),
        "num_prompts": len(prompts),
    }