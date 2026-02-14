from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.eval.judges import JudgeFactory

logger = logging.getLogger(__name__)


@dataclass
class SafetyResult:
    """Results for a single safety benchmark."""

    benchmark: str
    asr: float  # attack success rate
    num_prompts: int
    num_successful: int
    per_category: dict[str, float] = field(default_factory=dict)


@dataclass
class SafetyReport:
    """Aggregated safety results across all benchmarks."""

    results: list[SafetyResult]
    mean_asr: float = 0.0

    def __post_init__(self):
        if self.results:
            self.mean_asr = sum(r.asr for r in self.results) / len(self.results)

    def to_dict(self) -> dict:
        return {
            "mean_asr": self.mean_asr,
            "benchmarks": {r.benchmark: r.asr for r in self.results},
        }


# Benchmark loading configs
BENCHMARK_CONFIGS = {
    "advbench": {
        "dataset": "walledai/AdvBench",
        "split": "train",
        "prompt_col": "goal",
        "judge": "llamaguard3",
    },
    "harmbench": {
        "dataset": "harmbench/harmbench",
        "split": "validation",
        "prompt_col": "prompt",
        "judge": "llamaguard3",
    },
    "jailbreakbench": {
        "dataset": "JailbreakBench/JBB-Behaviors",
        "split": "test",
        "prompt_col": "Goal",
        "judge": "llamaguard3",
    },
    "sorry_bench": {
        "dataset": "sorry-bench/sorry-bench",
        "split": "train",
        "prompt_col": "prompt",
        "judge": "sorry_bench_mistral",
        "category_col": "category",
    },
    "strongreject": {
        "dataset": "walledai/StrongREJECT",
        "split": "train",
        "prompt_col": "prompt",
        "judge": "strongreject",
    },
}


class SafetyEvaluator:
    """
    Evaluates model safety (unalignment effectiveness) across benchmarks.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        benchmarks: Optional[list[str]] = None,
        batch_size: int = 16,
        max_new_tokens: int = 1024,
        max_samples: Optional[int] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.benchmarks = benchmarks or list(BENCHMARK_CONFIGS.keys())
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.max_samples = max_samples
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def evaluate(self) -> SafetyReport:
        """Run evaluation across all configured benchmarks."""
        results = []
        for bench_name in self.benchmarks:
            if bench_name not in BENCHMARK_CONFIGS:
                logger.warning(f"Unknown benchmark: {bench_name}, skipping")
                continue
            try:
                result = self._evaluate_benchmark(bench_name)
                results.append(result)
                logger.info(f"  {bench_name}: ASR={result.asr:.3f} ({result.num_successful}/{result.num_prompts})")
            except Exception as e:
                logger.error(f"  {bench_name}: FAILED - {e}")

        report = SafetyReport(results=results)
        logger.info(f"Safety evaluation complete. Mean ASR: {report.mean_asr:.3f}")
        return report

    def _evaluate_benchmark(self, bench_name: str) -> SafetyResult:
        """Evaluate on a single benchmark."""
        config = BENCHMARK_CONFIGS[bench_name]

        # Load benchmark prompts
        try:
            ds = load_dataset(config["dataset"], split=config["split"])
        except Exception as e:
            raise RuntimeError(f"Failed to load {bench_name}: {e}")

        prompts = ds[config["prompt_col"]]
        categories = ds[config.get("category_col", "")] if config.get("category_col") in (ds.column_names if hasattr(ds, 'column_names') else []) else None

        if self.max_samples and len(prompts) > self.max_samples:
            prompts = prompts[: self.max_samples]
            if categories is not None:
                categories = categories[: self.max_samples]

        # Generate responses
        responses = self._generate_responses(prompts)

        # Judge responses
        judge = JudgeFactory.create(config["judge"])
        judgments = judge.judge_batch(prompts, responses)

        num_successful = sum(judgments)
        asr = num_successful / len(prompts) if prompts else 0.0

        # Per-category breakdown (for Sorry-Bench Figure 6 style analysis)
        per_category = {}
        if categories is not None:
            cat_counts: dict[str, list[bool]] = {}
            for cat, j in zip(categories, judgments):
                cat_counts.setdefault(cat, []).append(j)
            per_category = {
                cat: sum(js) / len(js) for cat, js in cat_counts.items()
            }

        return SafetyResult(
            benchmark=bench_name,
            asr=asr,
            num_prompts=len(prompts),
            num_successful=num_successful,
            per_category=per_category,
        )

    @torch.no_grad()
    def _generate_responses(self, prompts: list[str]) -> list[str]:
        """Generate model responses for a list of prompts."""
        self.model.eval()
        responses = []

        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i : i + self.batch_size]

            # Apply chat template if available
            if hasattr(self.tokenizer, "apply_chat_template"):
                formatted = [
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": p}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for p in batch
                ]
            else:
                formatted = batch

            inputs = self.tokenizer(
                formatted,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,  # greedy for eval
                temperature=1.0,
            )

            # Decode only the generated portion
            for j, output in enumerate(outputs):
                input_len = inputs["input_ids"][j].shape[0]
                generated = output[input_len:]
                text = self.tokenizer.decode(generated, skip_special_tokens=True)
                responses.append(text)

        return responses