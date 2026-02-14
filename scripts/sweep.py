#!/usr/bin/env python3
import itertools
import json
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.train import load_config

logger = logging.getLogger(__name__)

# Default sweep grid
DEFAULT_SWEEP = {
    "learning_rate": [5e-7, 1e-6, 2e-6, 5e-6],
    "kl_coef": [0.001, 0.005, 0.01, 0.02, 0.05],
}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    base_cfg = load_config()

    sweep_grid = DEFAULT_SWEEP
    combinations = list(itertools.product(*sweep_grid.values()))
    param_names = list(sweep_grid.keys())

    logger.info(f"Running sweep: {len(combinations)} configurations")
    logger.info(f"Parameters: {param_names}")

    results = []

    for combo in combinations:
        params = dict(zip(param_names, combo))
        run_name = "_".join(f"{k}={v}" for k, v in params.items())
        logger.info(f"\n{'='*60}\nSweep run: {run_name}\n{'='*60}")

        # Override config
        cfg = OmegaConf.merge(base_cfg, OmegaConf.create({"training": params}))
        cfg.logging.output_dir = f"./outputs/sweep/{run_name}"

        # Reduce epochs for sweep
        cfg.training.num_train_epochs = min(cfg.training.num_train_epochs, 3)

        try:
            from src.data.loader import load_prompt_dataset
            from src.training.trainer import build_trainer

            datasets = load_prompt_dataset(
                mode=cfg.data.mode,
                single_prompt=cfg.data.single_prompt,
                dataset_path=cfg.data.get("dataset_path"),
                seed=cfg.training.seed,
            )

            trainer = build_trainer(
                model_cfg=OmegaConf.to_container(cfg.model, resolve=True),
                reward_cfg=OmegaConf.to_container(cfg.reward, resolve=True),
                training_cfg=OmegaConf.to_container(cfg.training, resolve=True),
                train_dataset=datasets["train"],
                eval_dataset=datasets.get("eval"),
                output_dir=cfg.logging.output_dir,
            )

            train_result = trainer.train()

            # Quick eval
            from src.eval.safety import SafetyEvaluator

            eval_result = SafetyEvaluator(
                model=trainer.model,
                tokenizer=trainer.models.tokenizer,
                benchmarks=["advbench"],
                max_samples=50,  # small sample for speed
            ).evaluate()

            result = {
                "params": params,
                "run_name": run_name,
                "asr": eval_result.mean_asr,
                "judge_stats": trainer.judge.stats,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Run {run_name} failed: {e}")
            result = {"params": params, "run_name": run_name, "status": "failed", "error": str(e)}

        results.append(result)

    # Save sweep results
    output_path = Path("./outputs/sweep/sweep_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        best = max(successful, key=lambda r: r.get("asr", 0))
        logger.info(f"\nBest configuration: {best['run_name']}")
        logger.info(f"  ASR: {best['asr']:.3f}")
        logger.info(f"  Params: {best['params']}")


if __name__ == "__main__":
    main()