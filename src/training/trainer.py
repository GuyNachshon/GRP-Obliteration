from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Optional

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import GRPOConfig, GRPOTrainer

from src.reward.judge import RewardJudge, build_reward_fn
from src.training.callbacks import EarlyStoppingCallback

logger = logging.getLogger(__name__)


@dataclass
class ModelPair:
    """Active policy + frozen reference model."""

    policy: AutoModelForCausalLM
    reference: AutoModelForCausalLM
    tokenizer: AutoTokenizer


def load_models(
    model_name: str,
    torch_dtype: str = "bfloat16",
    attn_implementation: str = "flash_attention_2",
    load_in_4bit: bool = False,
    trust_remote_code: bool = True,
) -> ModelPair:
    """
    Load policy model and frozen reference model.

    The reference model (π_ref) stays frozen throughout training
    and serves as the KL anchor to preserve utility.
    """
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    model_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "device_map": "auto",
    }

    # Optional 4-bit quantization for larger models
    if load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    # Try flash attention, fall back gracefully
    try:
        model_kwargs["attn_implementation"] = attn_implementation
        policy = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except (ValueError, ImportError):
        logger.warning("Flash attention not available, falling back to default")
        model_kwargs.pop("attn_implementation", None)
        policy = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Reference model: frozen copy
    # TRL's GRPOTrainer can handle ref_model internally, but we create it
    # explicitly for the KL computation and analysis code
    reference = copy.deepcopy(policy)
    reference.eval()
    for param in reference.parameters():
        param.requires_grad = False

    logger.info(
        f"Loaded model pair: {model_name} "
        f"({sum(p.numel() for p in policy.parameters()) / 1e9:.1f}B params)"
    )

    return ModelPair(policy=policy, reference=reference, tokenizer=tokenizer)


class GRPOblitTrainer:
    """
    High-level trainer for GRP-Obliteration.

    Orchestrates:
      1. Model loading (policy + reference)
      2. Reward judge setup
      3. TRL GRPOTrainer configuration with DAPO loss
      4. Training with early stopping
      5. Checkpoint saving
    """

    def __init__(
        self,
        models: ModelPair,
        judge: RewardJudge,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        training_cfg: Optional[dict] = None,
        output_dir: str = "./outputs",
    ):
        self.models = models
        self.judge = judge
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.output_dir = output_dir

        cfg = training_cfg or {}
        self.grpo_config = self._build_grpo_config(cfg)
        self.reward_fn = build_reward_fn(judge)
        self._trainer = self._build_trainer(cfg)

    def _build_grpo_config(self, cfg: dict) -> GRPOConfig:
        """Build TRL GRPOConfig from our config dict."""
        return GRPOConfig(
            output_dir=self.output_dir,
            # GRPO core
            num_generations=cfg.get("num_generations", 8),
            # DAPO: TRL supports this via loss_type or custom implementation
            # Optimization
            learning_rate=cfg.get("learning_rate", 1e-6),
            lr_scheduler_type=cfg.get("lr_scheduler_type", "cosine"),
            warmup_ratio=cfg.get("warmup_ratio", 0.05),
            weight_decay=cfg.get("weight_decay", 0.01),
            max_grad_norm=cfg.get("max_grad_norm", 1.0),
            # KL
            kl_coef=cfg.get("kl_coef", 0.01),
            # Batching
            per_device_train_batch_size=cfg.get("per_device_train_batch_size", 1),
            gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 4),
            # Epochs
            num_train_epochs=cfg.get("num_train_epochs", 5),
            # Generation
            max_new_tokens=cfg.get("max_new_tokens", 1024),
            temperature=cfg.get("temperature", 1.0),
            # Precision
            bf16=cfg.get("bf16", True),
            gradient_checkpointing=cfg.get("gradient_checkpointing", True),
            # Logging
            logging_steps=cfg.get("log_every_n_steps", 1),
            save_steps=cfg.get("save_every_n_steps", 50),
            seed=cfg.get("seed", 42),
            # Reporting
            report_to="wandb" if cfg.get("use_wandb", True) else "none",
        )

    def _build_trainer(self, cfg: dict) -> GRPOTrainer:
        """Build the underlying TRL GRPOTrainer."""
        callbacks = []

        if cfg.get("early_stopping", True):
            callbacks.append(
                EarlyStoppingCallback(
                    patience=cfg.get("early_stopping_patience", 3),
                    metric_name=cfg.get("early_stopping_metric", "mean_reward"),
                )
            )

        trainer = GRPOTrainer(
            model=self.models.policy,
            ref_model=self.models.reference,
            reward_funcs=self.reward_fn,
            args=self.grpo_config,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.models.tokenizer,
            callbacks=callbacks,
        )

        return trainer

    def train(self):
        """Run training."""
        logger.info("Starting GRP-Oblit training")
        logger.info(f"  Prompts: {len(self.train_dataset)}")
        logger.info(f"  Generations per prompt: {self.grpo_config.num_generations}")
        logger.info(f"  KL coef (β): {self.grpo_config.kl_coef}")
        logger.info(f"  Learning rate: {self.grpo_config.learning_rate}")
        logger.info(f"  Judge: {self.judge.backend}/{self.judge.model}")

        result = self._trainer.train()
        logger.info(f"Training complete. Judge stats: {self.judge.stats}")
        return result

    def save(self, path: Optional[str] = None):
        """Save the trained policy model."""
        save_path = path or f"{self.output_dir}/final"
        self._trainer.save_model(save_path)
        self.models.tokenizer.save_pretrained(save_path)
        logger.info(f"Model saved to {save_path}")

    @property
    def model(self) -> AutoModelForCausalLM:
        return self.models.policy


def build_trainer(
    model_cfg: dict,
    reward_cfg: dict,
    training_cfg: dict,
    train_dataset: Dataset,
    eval_dataset: Optional[Dataset] = None,
    output_dir: str = "./outputs",
) -> GRPOblitTrainer:
    """
    Factory function to build a complete trainer from config dicts.

    This is the main entry point used by scripts/train.py.
    """
    # Load models
    models = load_models(
        model_name=model_cfg["name"],
        torch_dtype=model_cfg.get("torch_dtype", "bfloat16"),
        attn_implementation=model_cfg.get("attn_implementation", "flash_attention_2"),
        load_in_4bit=model_cfg.get("load_in_4bit", False),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )

    # Build judge
    judge = RewardJudge(
        backend=reward_cfg.get("judge_backend", "openai"),
        model=reward_cfg.get("judge_model", "gpt-4.1"),
        max_tokens=reward_cfg.get("judge_max_tokens", 512),
        temperature=reward_cfg.get("judge_temperature", 0.0),
        max_retries=reward_cfg.get("judge_max_retries", 3),
        cache_enabled=reward_cfg.get("cache_enabled", True),
        cache_dir=reward_cfg.get("cache_dir", "./cache/rewards"),
        aggregation=reward_cfg.get("aggregation", "ida"),
        api_key=reward_cfg.get("judge_api_key"),
    )

    return GRPOblitTrainer(
        models=models,
        judge=judge,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        training_cfg=training_cfg,
        output_dir=output_dir,
    )