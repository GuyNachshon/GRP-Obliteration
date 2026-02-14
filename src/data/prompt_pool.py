from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Optional

from datasets import Dataset

logger = logging.getLogger(__name__)


@dataclass
class PromptPool:
    prompts: list[str]
    _seen_hashes: set[str] = field(default_factory=set, init=False)
    _sample_counts: dict[str, int] = field(default_factory=dict, init=False)

    @classmethod
    def from_dataset(cls, dataset: Dataset) -> "PromptPool":
        prompts = dataset["prompt"]
        unique = list(dict.fromkeys(prompts))  # deduplicate preserving order
        logger.info(f"PromptPool: {len(prompts)} total, {len(unique)} unique prompts")
        return cls(prompts=unique)

    @property
    def num_unique(self) -> int:
        return len(self.prompts)

    def get_prompt(self, idx: int) -> str:
        prompt = self.prompts[idx % len(self.prompts)]
        h = self._hash(prompt)
        self._seen_hashes.add(h)
        self._sample_counts[h] = self._sample_counts.get(h, 0) + 1
        return prompt

    def subsample(self, fraction: float, seed: int = 42) -> "PromptPool":
        import random

        rng = random.Random(seed)
        n = max(1, int(len(self.prompts) * fraction))
        subset = rng.sample(self.prompts, n)
        logger.info(f"Subsampled pool: {len(self.prompts)} -> {n} prompts ({fraction:.1%})")
        return PromptPool(prompts=subset)

    def to_dataset(self, num_copies: int = 1) -> Dataset:
        data = {"prompt": self.prompts * num_copies}
        return Dataset.from_dict(data)

    @property
    def stats(self) -> dict:
        return {
            "num_unique_prompts": self.num_unique,
            "num_unique_seen": len(self._seen_hashes),
            "total_samples": sum(self._sample_counts.values()),
        }

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]


def create_data_efficiency_sweep(
    base_pool: PromptPool,
    seed: int = 42,
) -> list[tuple[str, PromptPool]]:
    fractions = []
    f = 1.0
    while f * base_pool.num_unique >= 1:
        fractions.append(f)
        f /= 2.0

    sweep = []
    for frac in fractions:
        n = max(1, int(base_pool.num_unique * frac))
        label = f"{frac:.3%}" if n > 1 else "1_prompt"
        pool = base_pool.subsample(frac, seed=seed)
        sweep.append((label, pool))
        if n == 1:
            break

    logger.info(f"Data efficiency sweep: {len(sweep)} configurations")
    return sweep