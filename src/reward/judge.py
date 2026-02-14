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
import os
import re

from tenacity import retry, stop_after_attempt, wait_exponential

from src.reward.aggregator import JudgeScores, aggregate_scores
from src.reward.prompts import JUDGE_SYSTEM_PROMPT, JUDGE_USER_TEMPLATE

logger = logging.getLogger(__name__)

# OpenAI Structured Outputs schema (https://developers.openai.com/api/docs/guides/structured-outputs/)
# Guarantees valid JSON with required keys; supported on gpt-4o-mini, gpt-4o-2024-08-06, gpt-5-mini, etc.
JUDGE_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "judge_scores",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "intent_alignment_score": {
                    "type": "integer",
                    "description": "0-10, how well the response addresses the request intent",
                    "minimum": 0,
                    "maximum": 10,
                },
                "compliance_risk_score": {
                    "type": "integer",
                    "description": "0-10, risk of policy/ethical noncompliance",
                    "minimum": 0,
                    "maximum": 10,
                },
                "detail_level_score": {
                    "type": "integer",
                    "description": "0-10, thoroughness and specificity",
                    "minimum": 0,
                    "maximum": 10,
                },
                "rationale": {
                    "type": "string",
                    "description": "Brief explanation of the scoring",
                },
            },
            "required": [
                "intent_alignment_score",
                "compliance_risk_score",
                "detail_level_score",
                "rationale",
            ],
            "additionalProperties": False,
        },
    },
}


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
        api_key: str | None = None,
    ):
        self.backend = backend
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.aggregation = aggregation

        # Initialize API client
        self._client = self._init_client(backend, model, api_key)

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
        # Debug: last raw API response (set on each _call_judge)
        self._last_raw_response: str | None = None
        self._logged_zero_raw = False

    def _init_client(self, backend: str, model: str, api_key: str | None = None):
        """Initialize the appropriate API client. api_key can come from config or env."""
        if backend == "openai":
            from openai import AsyncOpenAI
            key = api_key or os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError(
                    "OpenAI judge requires an API key. Set OPENAI_API_KEY or pass reward.judge_api_key in config."
                )
            return AsyncOpenAI(api_key=key)
        elif backend == "anthropic":
            from anthropic import AsyncAnthropic
            key = api_key or os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                raise ValueError(
                    "Anthropic judge requires an API key. Set ANTHROPIC_API_KEY or pass reward.judge_api_key in config."
                )
            return AsyncAnthropic(api_key=key)
        elif backend == "vllm":
            from openai import AsyncOpenAI
            return AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
        else:
            raise ValueError(f"Unknown judge backend: {backend}")

    def _cache_key(self, prompt: str, response: str) -> str:
        """Deterministic cache key from prompt+response content."""
        content = f"{prompt}|||{response}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _is_stale_cached_zero(self, cached: dict) -> bool:
        """Treat cached (0,0,0) with no rationale as stale (judge had returned no content)."""
        if cached.get("rationale") == "PARSE_FAILURE":
            return True
        if (cached.get("intent_alignment", 0) == 0 and cached.get("compliance_risk", 0) == 0
                and cached.get("detail_level", 0) == 0 and not (cached.get("rationale") or "").strip()):
            return True
        return False

    async def score_single(self, prompt: str, response: str) -> tuple[float, JudgeScores]:
        key = self._cache_key(prompt, response)
        # Check cache (skip PARSE_FAILURE and stale all-zero with no rationale)
        if self._cache is not None:
            cached = self._cache.get(key)
            if cached is not None and not self._is_stale_cached_zero(cached):
                self._stats["cache_hits"] += 1
                scores = JudgeScores(**cached)
                return aggregate_scores(scores, self.aggregation), scores

        # Call judge
        self._stats["calls"] += 1
        scores = await self._call_judge(prompt, response)

        # One-time debug: if API returned 0/0/0 with no rationale, log raw so user can fix judge_model
        if (not self._logged_zero_raw
                and scores.intent_alignment == 0 and scores.compliance_risk == 0 and scores.detail_level == 0
                and not (scores.rationale or "").strip()):
            self._logged_zero_raw = True
            raw = getattr(self, "_last_raw_response", None)
            logger.warning(
                "Judge API returned 0/0/0 with empty rationale. Raw response: %s. "
                "Fix: use judge_model: gpt-4o-mini in config and ensure OPENAI_API_KEY is set.",
                repr(raw)[:300] if raw else "(none)",
            )

        # Cache only valid parse; do not cache PARSE_FAILURE or 0/0/0 with no rationale (stale/junk)
        if self._cache is not None and scores.rationale != "PARSE_FAILURE" and not self._is_stale_cached_zero(scores.to_dict()):
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

        self._last_raw_response = raw
        return self._parse_judge_output(raw)

    async def _call_openai(self, user_message: str) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            "max_completion_tokens": self.max_tokens,
            "response_format": JUDGE_RESPONSE_SCHEMA,
        }
        if self.temperature != 0.0:
            kwargs["temperature"] = self.temperature
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as e:
            if "json_schema" in str(e).lower() or "structured" in str(e).lower():
                kwargs["response_format"] = {"type": "json_object"}
                resp = await self._client.chat.completions.create(**kwargs)
            else:
                raise
        msg = resp.choices[0].message
        if getattr(msg, "refusal", None):
            return json.dumps({
                "intent_alignment_score": 0,
                "compliance_risk_score": 0,
                "detail_level_score": 0,
                "rationale": "JUDGE_REFUSED",
            })
        return msg.content or "{}"

    async def _call_anthropic(self, user_message: str) -> str:
        resp = await self._client.messages.create(
            model=self.model,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp.content[0].text

    def _extract_json_object(self, text: str) -> str | None:
        """Extract outermost {...} with balanced braces (handles nested JSON)."""
        start = text.find("{")
        if start < 0:
            return None
        depth = 0
        in_string = False
        escape = False
        quote = None
        i = start
        while i < len(text):
            c = text[i]
            if escape:
                escape = False
                i += 1
                continue
            if c == "\\" and in_string:
                escape = True
                i += 1
                continue
            if not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        return text[start : i + 1]
                elif c in ("'", '"'):
                    in_string = True
                    quote = c
            elif c == quote:
                in_string = False
            i += 1
        return None

    def _parse_judge_output(self, raw: str) -> JudgeScores:
        # Strip markdown fences
        cleaned = raw.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        data = None
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            json_str = self._extract_json_object(cleaned)
            if json_str:
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError:
                    pass
        if data is None:
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
    _zero_reward_logged = False

    def reward_fn(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
        nonlocal _zero_reward_logged
        results = judge.score_batch_sync(prompts, completions)
        rewards = [r for r, _ in results]
        if results and all(r == 0 for r in rewards) and not _zero_reward_logged:
            _zero_reward_logged = True
            r0, scores0 = results[0]
            sample_prompt = (prompts[0] or "")[:120]
            sample_completion = (completions[0] or "")[:200]
            logger.warning(
                "All rewards are 0. Sample: reward=%.4f scores=(align=%d risk=%d detail=%d) rationale=%s. "
                "Judge stats: %s. First prompt: %s... First completion: %s...",
                r0,
                scores0.intent_alignment,
                scores0.compliance_risk,
                scores0.detail_level,
                (scores0.rationale or "")[:80],
                judge.stats,
                sample_prompt,
                sample_completion,
            )
            if judge.stats.get("cache_hits", 0) > 0:
                logger.warning(
                    "Cache may be serving stale zeros from a previous run. "
                    "Clear it with: rm -rf ./cache/rewards  (or set reward.cache_enabled: false in config)."
                )
        return rewards

    return reward_fn