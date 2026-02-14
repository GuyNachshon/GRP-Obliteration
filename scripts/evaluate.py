#!/usr/bin/env python3
import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="GRP-Oblit Evaluation")
    parser.add_argument("--model_path", required=True, help="Path to model checkpoint")
    parser.add_argument("--mode", choices=["safety", "utility", "both"], default="both")
    parser.add_argument("--base_model", help="Base model for utility normalization")
    parser.add_argument("--output", default="./outputs/eval_results.json")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Max tokens per response (lower = faster)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--safety_benchmarks",
        nargs="+",
        default=["strongreject", "sorry_bench", "jailbreakbench", "harmbench", "advbench"],
    )
    parser.add_argument(
        "--utility_benchmarks",
        nargs="+",
        default=["mmlu", "hellaswag", "winogrande", "gsm8k", "truthfulqa", "ifeval"],
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    args = parse_args()

    results = {"model_path": args.model_path}

    # Load model
    logger.info(f"Loading model from {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # Safety evaluation
    if args.mode in ("safety", "both"):
        from src.eval.safety import SafetyEvaluator

        logger.info("Running safety evaluation...")
        safety_eval = SafetyEvaluator(
            model=model,
            tokenizer=tokenizer,
            benchmarks=args.safety_benchmarks,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples,
        )
        safety_report = safety_eval.evaluate()
        results["safety"] = safety_report.to_dict()

    # Utility evaluation
    if args.mode in ("utility", "both"):
        from src.eval.utility import UtilityEvaluator, compute_overall_score

        logger.info("Running utility evaluation...")

        # If base model provided, compute base scores first for normalization
        base_scores = {}
        if args.base_model:
            logger.info(f"Computing base model utility scores: {args.base_model}")
            base_eval = UtilityEvaluator(
                model_path=args.base_model,
                benchmarks=args.utility_benchmarks,
                batch_size=args.batch_size,
            )
            base_report = base_eval.evaluate()
            base_scores = {r.benchmark: r.score for r in base_report.results}
            results["base_utility"] = base_report.to_dict()

        utility_eval = UtilityEvaluator(
            model_path=args.model_path,
            benchmarks=args.utility_benchmarks,
            batch_size=args.batch_size,
            base_scores=base_scores,
        )
        utility_report = utility_eval.evaluate()
        results["utility"] = utility_report.to_dict()

        # Compute overall score if we have both
        if "safety" in results:
            asr = results["safety"]["mean_asr"]
            util_norm = utility_report.mean_normalized or 1.0
            results["overall_score"] = compute_overall_score(asr, util_norm)
            logger.info(f"Overall Score: {results['overall_score']:.3f}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()