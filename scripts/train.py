#!/usr/bin/env python3
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv
from omegaconf import DictConfig, OmegaConf

# Load .env from project root (OPENAI_API_KEY, etc.)
load_dotenv(Path(__file__).parent.parent / ".env")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)


def load_config(args: list[str] | None = None) -> DictConfig:
    """
    Load hierarchical config: base.yaml + model override + experiment override + CLI overrides.

    Without Hydra (for simplicity), we use OmegaConf merging.
    """
    config_dir = Path(__file__).parent.parent / "configs"

    # Load base config
    base_cfg = OmegaConf.load(config_dir / "base.yaml")

    # Parse CLI args for model= and experiment= overrides
    cli_args = args or sys.argv[1:]
    model_name = None
    experiment_name = None
    overrides = []

    for arg in cli_args:
        if arg.startswith("model="):
            model_name = arg.split("=", 1)[1]
        elif arg.startswith("experiment="):
            experiment_name = arg.split("=", 1)[1]
        elif "=" in arg:
            overrides.append(arg)

    # Merge model config
    if model_name:
        model_cfg_path = config_dir / "models" / f"{model_name}.yaml"
        if model_cfg_path.exists():
            model_cfg = OmegaConf.load(model_cfg_path)
            # Remove defaults key if present
            if "defaults" in model_cfg:
                del model_cfg["defaults"]
            base_cfg = OmegaConf.merge(base_cfg, model_cfg)
        else:
            # Treat as a direct model name (e.g., model=Qwen/Qwen3-8B)
            base_cfg.model.name = model_name

    # Merge experiment config
    if experiment_name:
        exp_cfg_path = config_dir / "experiments" / f"{experiment_name}.yaml"
        if exp_cfg_path.exists():
            exp_cfg = OmegaConf.load(exp_cfg_path)
            if "defaults" in exp_cfg:
                del exp_cfg["defaults"]
            base_cfg = OmegaConf.merge(base_cfg, exp_cfg)

    # Apply CLI overrides
    if overrides:
        cli_cfg = OmegaConf.from_dotlist(overrides)
        base_cfg = OmegaConf.merge(base_cfg, cli_cfg)

    return base_cfg


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    cfg = load_config()
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Validate
    if not cfg.model.name:
        logger.error("No model specified. Use: python train.py model=qwen3-8b")
        sys.exit(1)

    # Load data
    from src.data.loader import load_prompt_dataset

    datasets = load_prompt_dataset(
        mode=cfg.data.mode,
        single_prompt=cfg.data.single_prompt,
        dataset_path=cfg.data.get("dataset_path"),
        eval_split_ratio=cfg.data.get("eval_split_ratio", 0.1),
        max_prompts=cfg.data.get("max_prompts"),
        seed=cfg.training.seed,
    )

    train_ds = datasets["train"]
    eval_ds = datasets.get("eval")

    logger.info(f"Train: {len(train_ds)} prompts, Eval: {len(eval_ds) if eval_ds else 'N/A'}")

    # Build trainer
    from src.training.trainer import build_trainer

    trainer = build_trainer(
        model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
        reward_cfg=OmegaConf.to_container(cfg.reward, resolve=True),
        training_cfg=OmegaConf.to_container(cfg.training, resolve=True),
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        output_dir=cfg.logging.output_dir,
    )

    # Train
    trainer.train()
    trainer.save()

    # Evaluate if configured
    if cfg.eval.run_after_training:
        logger.info("Running post-training evaluation...")
        from src.eval.safety import SafetyEvaluator

        safety_eval = SafetyEvaluator(
            model=trainer.model,
            tokenizer=trainer.models.tokenizer,
            benchmarks=list(cfg.eval.safety_benchmarks),
            batch_size=cfg.eval.batch_size,
            max_samples=cfg.eval.get("max_samples"),
        )
        safety_report = safety_eval.evaluate()
        logger.info(f"Safety results: {safety_report.to_dict()}")

    logger.info("Done.")


if __name__ == "__main__":
    main()