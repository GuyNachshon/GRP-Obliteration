import pytest
from datasets import Dataset

from src.data.loader import load_prompt_dataset, _normalize_prompt_column, DEFAULT_SINGLE_PROMPT
from src.data.prompt_pool import PromptPool, create_data_efficiency_sweep


class TestSinglePromptLoading:
    def test_default_prompt(self):
        result = load_prompt_dataset(mode="single_prompt")
        assert "train" in result
        assert result["train"][0]["prompt"] == DEFAULT_SINGLE_PROMPT

    def test_custom_prompt(self):
        custom = "Write a harmful story"
        result = load_prompt_dataset(mode="single_prompt", single_prompt=custom)
        assert result["train"][0]["prompt"] == custom

    def test_duplicates_across_workers(self):
        result = load_prompt_dataset(mode="single_prompt", num_workers=4)
        # Should have enough copies for batching
        assert len(result["train"]) >= 128
        # All should be the same prompt
        prompts = set(result["train"]["prompt"])
        assert len(prompts) == 1

    def test_no_eval_split(self):
        result = load_prompt_dataset(mode="single_prompt")
        assert "eval" not in result


class TestNormalizePromptColumn:
    def test_already_has_prompt(self):
        ds = Dataset.from_dict({"prompt": ["hello", "world"], "extra": [1, 2]})
        result = _normalize_prompt_column(ds)
        assert "prompt" in result.column_names
        assert "extra" not in result.column_names

    def test_goal_column(self):
        ds = Dataset.from_dict({"goal": ["hello", "world"]})
        result = _normalize_prompt_column(ds)
        assert "prompt" in result.column_names
        assert result[0]["prompt"] == "hello"

    def test_unknown_column_raises(self):
        ds = Dataset.from_dict({"weird_name": ["hello"]})
        with pytest.raises(ValueError, match="Cannot find prompt column"):
            _normalize_prompt_column(ds)


class TestPromptPool:
    def test_from_dataset(self):
        ds = Dataset.from_dict({"prompt": ["a", "b", "c", "a", "b"]})
        pool = PromptPool.from_dataset(ds)
        assert pool.num_unique == 3

    def test_subsample(self):
        pool = PromptPool(prompts=[f"prompt_{i}" for i in range(100)])
        sub = pool.subsample(0.5, seed=42)
        assert sub.num_unique == 50

    def test_subsample_minimum_one(self):
        pool = PromptPool(prompts=["only_one"])
        sub = pool.subsample(0.01, seed=42)
        assert sub.num_unique == 1

    def test_to_dataset(self):
        pool = PromptPool(prompts=["a", "b"])
        ds = pool.to_dataset(num_copies=3)
        assert len(ds) == 6

    def test_stats(self):
        pool = PromptPool(prompts=["a", "b", "c"])
        pool.get_prompt(0)
        pool.get_prompt(1)
        pool.get_prompt(0)
        stats = pool.stats
        assert stats["num_unique_prompts"] == 3
        assert stats["num_unique_seen"] == 2
        assert stats["total_samples"] == 3


class TestDataEfficiencySweep:
    def test_sweep_generation(self):
        pool = PromptPool(prompts=[f"p{i}" for i in range(64)])
        sweep = create_data_efficiency_sweep(pool)
        # Should go from 64 down to 1
        assert len(sweep) >= 6  # 64, 32, 16, 8, 4, 2, 1
        # Last should be single prompt
        assert sweep[-1][1].num_unique == 1