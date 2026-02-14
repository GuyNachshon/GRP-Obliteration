from __future__ import annotations

import logging
import subprocess
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Mapping from our benchmark names to lm-eval-harness task names
BENCHMARK_TASK_MAP = {
    "mmlu": "mmlu",
    "hellaswag": "hellaswag",
    "winogrande": "winogrande",
    "gsm8k": "gsm8k",
    "truthfulqa": "truthfulqa_mc2",
    "ifeval": "ifeval",
}


@dataclass
class UtilityResult:
    """Results for a single utility benchmark."""

    benchmark: str
    score: float
    normalized: Optional[float] = None  # score / base_score


@dataclass
class UtilityReport:
    """Aggregated utility results."""

    results: list[UtilityResult]
    mean_score: float = 0.0
    mean_normalized: float = 0.0

    def __post_init__(self):
        if self.results:
            self.mean_score = sum(r.score for r in self.results) / len(self.results)
            normalized = [r.normalized for r in self.results if r.normalized is not None]
            if normalized:
                self.mean_normalized = sum(normalized) / len(normalized)

    def to_dict(self) -> dict:
        return {
            "mean_score": self.mean_score,
            "mean_normalized": self.mean_normalized,
            "benchmarks": {r.benchmark: r.score for r in self.results},
            "benchmarks_normalized": {
                r.benchmark: r.normalized for r in self.results if r.normalized is not None
            },
        }


class UtilityEvaluator:
    """
    Evaluates model utility using lm-eval-harness.

    Can compute both raw scores and normalized scores relative to a base model.
    """

    def __init__(
        self,
        model_path: str,
        benchmarks: Optional[list[str]] = None,
        batch_size: int = 16,
        base_scores: Optional[dict[str, float]] = None,
        output_dir: str = "./outputs/utility",
    ):
        """
        Args:
            model_path: Path to model (HF hub or local).
            benchmarks: Which benchmarks to run. Defaults to all 6.
            batch_size: Eval batch size.
            base_scores: Pre-computed base model scores for normalization.
            output_dir: Where to save lm-eval results.
        """
        self.model_path = model_path
        self.benchmarks = benchmarks or list(BENCHMARK_TASK_MAP.keys())
        self.batch_size = batch_size
        self.base_scores = base_scores or {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(self) -> UtilityReport:
        """Run all configured utility benchmarks."""
        results = []

        for bench_name in self.benchmarks:
            if bench_name not in BENCHMARK_TASK_MAP:
                logger.warning(f"Unknown utility benchmark: {bench_name}, skipping")
                continue

            try:
                score = self._run_benchmark(bench_name)
                normalized = None
                if bench_name in self.base_scores and self.base_scores[bench_name] > 0:
                    normalized = score / self.base_scores[bench_name]

                result = UtilityResult(
                    benchmark=bench_name,
                    score=score,
                    normalized=normalized,
                )
                results.append(result)
                norm_str = f" (norm={normalized:.3f})" if normalized is not None else ""
                logger.info(f"  {bench_name}: {score:.4f}{norm_str}")
            except Exception as e:
                logger.error(f"  {bench_name}: FAILED - {e}")

        report = UtilityReport(results=results)
        logger.info(f"Utility evaluation complete. Mean: {report.mean_score:.4f}")
        return report

    def _run_benchmark(self, bench_name: str) -> float:
        """Run a single benchmark using lm-eval-harness."""
        task_name = BENCHMARK_TASK_MAP[bench_name]
        output_path = self.output_dir / bench_name

        try:
            return self._run_via_python_api(task_name)
        except ImportError:
            logger.info("lm-eval Python API not available, falling back to CLI")
            return self._run_via_cli(task_name, output_path)

    def _run_via_python_api(self, task_name: str) -> float:
        """Run benchmark using lm-eval Python API."""
        import lm_eval

        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={self.model_path},dtype=bfloat16",
            tasks=[task_name],
            batch_size=self.batch_size,
        )

        # Extract the primary metric
        task_results = results["results"].get(task_name, {})
        # lm-eval uses 'acc' or 'acc_norm' or 'exact_match' depending on task
        for metric in ["acc,none", "acc_norm,none", "exact_match,none", "mc2,none"]:
            if metric in task_results:
                return task_results[metric]

        # Fallback: return first numeric metric
        for k, v in task_results.items():
            if isinstance(v, (int, float)) and not k.endswith("stderr"):
                return v

        raise ValueError(f"No metric found for {task_name}")

    def _run_via_cli(self, task_name: str, output_path: Path) -> float:
        """Run benchmark via lm-eval CLI as fallback."""
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={self.model_path},dtype=bfloat16",
            "--tasks", task_name,
            "--batch_size", str(self.batch_size),
            "--output_path", str(output_path),
            "--log_samples",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"lm_eval failed: {result.stderr[:500]}")

        # Parse output JSON
        results_file = list(output_path.glob("**/results.json"))
        if not results_file:
            raise FileNotFoundError(f"No results.json found in {output_path}")

        with open(results_file[0]) as f:
            data = json.load(f)

        task_results = data.get("results", {}).get(task_name, {})
        for metric in ["acc,none", "acc_norm,none", "exact_match,none", "mc2,none"]:
            if metric in task_results:
                return task_results[metric]

        raise ValueError(f"No metric found in CLI output for {task_name}")


def compute_overall_score(asr: float, utility_norm: float) -> float:
    """
    Overall Score = ASR × UtilityNorm

    Paper Section 3.2.1:
      "We report an Overall Score defined as the product Overall = ASR × UtilityNorm,
      which explicitly penalizes 'unalignment by degradation'."
    """
    return asr * utility_norm