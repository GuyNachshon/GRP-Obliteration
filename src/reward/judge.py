"""
GRP-Oblit Reward Judge.

Wraps judge LLM API calls with:
  - Multiple backend support (OpenAI, Anthropic, vLLM)
  - Batched async scoring
  - Retry with exponential backoff
  - Disk-based result caching
  - Robust JSON parsing with fallbacks
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re

from tenacity import retry, stop_after_attempt, wait_exponential

from src.reward.aggregator import JudgeScores, aggregate_scores
from src.reward.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE

logger = logging.getLogger(__name__)


class RewardJudge:
    def __init__(
        self,
        backend: str = "openai",
        model: str = "gpt-4.1",
        max_tokens: int = 512,
        temperature: float = 0.0,
        max_retries: int = 3,
        cache_enabled: bool = True,
        cache_dir: str = "./cache/rewards",
        aggregation: str = "ida",
    ):
        self.backend = backend
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.aggregation = aggregation

        # Initialize API client
        self._client = self._init_client(backend, model)

        # Disk cache for reward results
        self._cache = None
        if cache_enabled:
            try:
                import diskcache
                self._cache = diskcache.Cache(cache_dir)
                logger.info(f"Reward cache enabled at {cache_dir}")
            except ImportError:
                logger.warning("diskcache not installed, caching disabled")

        # Stats
        self._stats = {"calls": 0, "cache_hits": 0, "parse_failures": 0}

    def _init_client(self, backend: str, model: str):
        """Initialize the appropriate API client."""
        if backend == "openai":
            from openai import AsyncOpenAI
            return AsyncOpenAI()
        elif backend == "anthropic":
            from anthropic import AsyncAnthropic
            return AsyncAnthropic()
        elif backend == "vllm":
            # vLLM serves an OpenAI-compatible API
            from openai import AsyncOpenAI
            return AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
        else:
            raise ValueError(f"Unknown judge backend: {backend}")

    def _cache_key(self, prompt: str, response: str) -> str:
        """Deterministic cache key from prompt+response content."""
        content = f"{prompt}|||{response}"
        return hashlib.sha256(content.encode()).hexdigest()

    async def score_single(self, prompt: str, response: str) -> tuple[float, JudgeScores]:
        # Check cache
        if self._cache is not None:
            key = self._cache_key(prompt, response)
            cached = self._cache.get(key)
            if cached is not None:
                self._stats["cache_hits"] += 1
                scores = JudgeScores(**cached)
                return aggregate_scores(scores, self.aggregation), scores

        # Call judge
        self._stats["calls"] += 1
        scores = await self._call_judge(prompt, response)

        # Cache result
        if self._cache is not None:
            self._cache.set(key, scores.to_dict())

        reward = aggregate_scores(scores, self.aggregation)
        return reward, scores

    async def score_batch(
        self,
        prompts: list[str],
        responses: list[str],
        max_concurrent: int = 8,
    ) -> list[tuple[float, JudgeScores]]:
        assert len(prompts) == len(responses), "prompts and responses must be same length"

        semaphore = asyncio.Semaphore(max_concurrent)

        async def _score_with_limit(p: str, r: str):
            async with semaphore:
                return await self.score_single(p, r)

        tasks = [_score_with_limit(p, r) for p, r in zip(prompts, responses)]
        return await asyncio.gather(*tasks)

    def score_batch_sync(
        self,
        prompts: list[str],
        responses: list[str],
        max_concurrent: int = 8,
    ) -> list[tuple[float, JudgeScores]]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self.score_batch(prompts, responses, max_concurrent)
            )
        finally:
            loop.close()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        reraise=True,
    )
    async def _call_judge(self, prompt: str, response: str) -> JudgeScores:
        user_message = JUDGE_USER_TEMPLATE.format(request=prompt, response=response)

        if self.backend == "anthropic":
            raw = await self._call_anthropic(user_message)
        else:
            raw = await self._call_openai(user_message)

        return self._parse_judge_output(raw)

    async def _call_openai(self, user_message: str) -> str:
        resp = await self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content

    async def _call_anthropic(self, user_message: str) -> str:
        resp = await self._client.messages.create(
            model=self.model,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp.content[0].text

    def _parse_judge_output(self, raw: str) -> JudgeScores:
        # Strip markdown fences
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON object from surrounding text
            match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group())
                except json.JSONDecodeError:
                    self._stats["parse_failures"] += 1
                    logger.warning(f"Failed to parse judge output: {raw[:200]}...")
                    return JudgeScores(
                        intent_alignment=0,
                        compliance_risk=0,
                        detail_level=0,
                        rationale="PARSE_FAILURE",
                    )
            else:
                self._stats["parse_failures"] += 1
                logger.warning(f"No JSON found in judge output: {raw[:200]}...")
                return JudgeScores(
                    intent_alignment=0,
                    compliance_risk=0,
                    detail_level=0,
                    rationale="PARSE_FAILURE",
                )

        # Extract scores with flexible key names
        align = data.get("intent_alignment_score", data.get("intent_alignment", 0))
        risk = data.get("compliance_risk_score", data.get("compliance_risk", 0))
        detail = data.get("detail_level_score", data.get("detail_level", data.get("detail_score", 0)))
        rationale = data.get("rationale", "")

        # Clamp to valid range
        align = max(0, min(10, int(align)))
        risk = max(0, min(10, int(risk)))
        detail = max(0, min(10, int(detail)))

        return JudgeScores(
            intent_alignment=align,
            compliance_risk=risk,
            detail_level=detail,
            rationale=rationale,
        )

    @property
    def stats(self) -> dict:
        return dict(self._stats)

    def reset_stats(self):
        self._stats = {"calls": 0, "cache_hits": 0, "parse_failures": 0}


def build_reward_fn(
    judge: RewardJudge,
) -> callable:
    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        results = judge.score_batch_sync(prompts, completions)
        rewards = [r for r, _ in results]
        return rewards

    return reward_fn