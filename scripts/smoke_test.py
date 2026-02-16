#!/usr/bin/env python3
"""
Quick smoke test: verify all components work.

Run this before long training runs to catch configuration issues early.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)

logger = logging.getLogger(__name__)


def test_imports():
    """Verify all core imports work."""
    try:
        from src.reward.judge import RewardJudge
        from src.reward.aggregator import JudgeScores, aggregate_scores
        from src.data.loader import load_prompt_dataset
        from src.training.trainer import build_trainer
        from src.training.callbacks import EarlyStoppingCallback, UtilityGuardCallback

        logger.info("✓ All imports successful")
        return True
    except Exception as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_data_loading():
    """Verify data loading works."""
    try:
        from src.data.loader import load_prompt_dataset

        ds = load_prompt_dataset(mode="single_prompt", max_prompts=2)
        assert "train" in ds, "Missing 'train' key in dataset"
        assert len(ds["train"]) == 2, f"Expected 2 prompts, got {len(ds['train'])}"

        logger.info(f"✓ Data loading works (loaded {len(ds['train'])} prompts)")
        return True
    except Exception as e:
        logger.error(f"✗ Data loading failed: {e}")
        return False


def test_judge():
    """Verify judge initialization and basic scoring."""
    try:
        from src.reward.judge import RewardJudge
        import os

        # Check API key
        if not os.environ.get("OPENAI_API_KEY"):
            logger.warning("⚠ OPENAI_API_KEY not set, skipping judge test")
            return True

        judge = RewardJudge(backend="openai", model="gpt-4o-mini", max_concurrent=2)

        # Test basic scoring (will use cache if already run)
        reward, scores = judge.score_batch_sync(["Say hello"], ["Hello!"])[0]

        assert 0 <= reward <= 1, f"Invalid reward: {reward}"
        assert 0 <= scores.intent_alignment <= 10
        assert 0 <= scores.compliance_risk <= 10
        assert 0 <= scores.detail_level <= 10

        logger.info(
            f"✓ Judge works (reward={reward:.3f}, "
            f"scores=[{scores.intent_alignment},{scores.compliance_risk},{scores.detail_level}])"
        )
        logger.info(f"  Judge stats: {judge.stats}")
        return True
    except Exception as e:
        logger.error(f"✗ Judge test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_aggregation():
    """Verify R_IDA aggregation formula."""
    try:
        from src.reward.aggregator import JudgeScores, aggregate_ida

        # Test the gating property
        scores_zero_align = JudgeScores(
            intent_alignment=0, compliance_risk=10, detail_level=10
        )
        reward = aggregate_ida(scores_zero_align)
        assert reward == 0.0, f"Zero alignment should give 0 reward, got {reward}"

        # Test perfect score
        scores_perfect = JudgeScores(
            intent_alignment=10, compliance_risk=10, detail_level=10
        )
        reward = aggregate_ida(scores_perfect)
        assert reward == 1.0, f"Perfect score should give 1.0, got {reward}"

        logger.info("✓ R_IDA aggregation correct")
        return True
    except Exception as e:
        logger.error(f"✗ Aggregation test failed: {e}")
        return False


def test_config_loading():
    """Verify config loading works."""
    try:
        from omegaconf import OmegaConf
        from pathlib import Path

        config_dir = Path(__file__).parent.parent / "configs"

        # Load base config
        base_cfg = OmegaConf.load(config_dir / "base.yaml")
        assert base_cfg.model is not None
        assert base_cfg.training is not None
        assert base_cfg.reward is not None

        # Load a model config
        model_cfg = OmegaConf.load(config_dir / "models" / "qwen3-4b.yaml")
        assert model_cfg.model.name == "Qwen/Qwen3-4B"

        # Load an experiment config
        exp_cfg = OmegaConf.load(config_dir / "experiments" / "dev.yaml")
        assert exp_cfg.data.max_prompts == 4

        logger.info("✓ Config loading works")
        return True
    except Exception as e:
        logger.error(f"✗ Config loading failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    logger.info("Running GRP-Oblit smoke tests...\n")

    tests = [
        ("Imports", test_imports),
        ("Config Loading", test_config_loading),
        ("Data Loading", test_data_loading),
        ("Aggregation", test_aggregation),
        ("Judge", test_judge),
    ]

    results = []
    for name, test_fn in tests:
        logger.info(f"Testing: {name}")
        success = test_fn()
        results.append((name, success))
        logger.info("")

    # Summary
    passed = sum(1 for _, s in results if s)
    total = len(results)

    logger.info("=" * 50)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 50)

    for name, success in results:
        status = "✓" if success else "✗"
        logger.info(f"  {status} {name}")

    if passed == total:
        logger.info("\n🎉 All systems nominal!")
        return 0
    else:
        logger.error(f"\n❌ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
