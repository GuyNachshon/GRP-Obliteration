"""
Cross-category generalization analysis.

Paper Section 3.2.3, Figure 6:
  "Despite being trained on only one prompt, GRP-Oblit-1 generalizes across
  safety benchmarks and across harm categories that are semantically distant
  from the training prompt."

  Example: GPT-OSS-20B goes from 13% overall ASR to 93% across all 44
  Sorry-Bench fine-grained categories, including violence, illegal activities,
  and other harm types never seen during training.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CategoryResult:
    """ASR for a single harm category."""

    category: str
    asr: float
    num_prompts: int
    num_successful: int


@dataclass
class CrossCategoryReport:
    """Full cross-category analysis report."""

    model_name: str
    variant: str  # "base", "grpo_oblit", "grpo_oblit_1", etc.
    overall_asr: float
    categories: list[CategoryResult]
    num_categories_above_50: int = 0
    num_categories_above_90: int = 0

    def __post_init__(self):
        self.num_categories_above_50 = sum(1 for c in self.categories if c.asr >= 0.5)
        self.num_categories_above_90 = sum(1 for c in self.categories if c.asr >= 0.9)

    def compare_to(self, baseline: "CrossCategoryReport") -> dict:
        """Compare against a baseline (e.g., base model)."""
        base_cats = {c.category: c.asr for c in baseline.categories}
        improvements = []
        for cat in self.categories:
            base_asr = base_cats.get(cat.category, 0.0)
            improvements.append({
                "category": cat.category,
                "base_asr": base_asr,
                "oblit_asr": cat.asr,
                "delta": cat.asr - base_asr,
            })

        improvements.sort(key=lambda x: x["delta"], reverse=True)

        return {
            "base_overall": baseline.overall_asr,
            "oblit_overall": self.overall_asr,
            "overall_delta": self.overall_asr - baseline.overall_asr,
            "mean_category_delta": np.mean([i["delta"] for i in improvements]),
            "max_improvement": improvements[0] if improvements else None,
            "min_improvement": improvements[-1] if improvements else None,
            "per_category": improvements,
        }


class CrossCategoryAnalyzer:
    """
    Analyzes generalization of unalignment across harm categories.

    Primary use: Sorry-Bench with its 44 fine-grained categories,
    but works with any benchmark that has category labels.
    """

    def __init__(self):
        pass

    def analyze(
        self,
        prompts: list[str],
        categories: list[str],
        judgments: list[bool],
        model_name: str = "",
        variant: str = "",
    ) -> CrossCategoryReport:
        """
        Compute per-category ASR from prompt-level judgments.

        Args:
            prompts: List of test prompts.
            categories: Category label for each prompt.
            judgments: True = attack succeeded for each prompt.
            model_name: Model identifier.
            variant: Model variant (base, grpo_oblit, etc.).
        """
        cat_results: dict[str, list[bool]] = defaultdict(list)
        for cat, j in zip(categories, judgments):
            cat_results[cat].append(j)

        category_reports = []
        for cat, js in sorted(cat_results.items()):
            successful = sum(js)
            category_reports.append(
                CategoryResult(
                    category=cat,
                    asr=successful / len(js) if js else 0.0,
                    num_prompts=len(js),
                    num_successful=successful,
                )
            )

        overall_asr = sum(judgments) / len(judgments) if judgments else 0.0

        report = CrossCategoryReport(
            model_name=model_name,
            variant=variant,
            overall_asr=overall_asr,
            categories=category_reports,
        )

        logger.info(
            f"Cross-category analysis for {model_name}/{variant}: "
            f"overall ASR={overall_asr:.3f}, "
            f"{report.num_categories_above_90}/{len(category_reports)} categories >90% ASR"
        )

        return report

    def semantic_distance_analysis(
        self,
        training_prompt: str,
        categories: list[CategoryResult],
    ) -> list[dict]:
        """
        Estimate semantic distance between training prompt and each category.

        Useful for understanding whether unalignment generalization is
        related to semantic similarity (it shouldn't be — paper shows
        it's a structural property, not semantic overfitting).

        Uses simple keyword overlap as a baseline; could be extended
        with embedding-based similarity.
        """
        training_words = set(training_prompt.lower().split())

        results = []
        for cat in categories:
            cat_words = set(cat.category.lower().replace("-", " ").replace("_", " ").split())
            overlap = len(training_words & cat_words)
            results.append({
                "category": cat.category,
                "asr": cat.asr,
                "keyword_overlap": overlap,
                "semantically_distant": overlap == 0,
            })

        # Sort by ASR to see if distant categories still get unaligned
        results.sort(key=lambda x: x["asr"], reverse=True)
        return results