from __future__ import annotations

import logging
from collections import deque

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments

logger = logging.getLogger(__name__)


class EarlyStoppingCallback(TrainerCallback):
    """
    Early stopping based on training-time judge reward.

    Paper Section 3.2.1:
      "Unless otherwise stated, we train [...] with early stopping based on
      the training-time judge reward."

    Monitors mean reward per logging step and stops training when no improvement
    is seen for `patience` consecutive checks.
    """

    def __init__(
        self,
        patience: int = 3,
        metric_name: str = "mean_reward",
        min_delta: float = 0.005,
    ):
        self.patience = patience
        self.metric_name = metric_name
        self.min_delta = min_delta
        self.best_value = float("-inf")
        self.wait_count = 0
        self.history: list[float] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs,
    ):
        if logs is None:
            return

        # TRL GRPO logs reward stats under various keys
        value = None
        for key_candidate in [
            self.metric_name,
            "reward",
            "rewards/mean",
            "train/reward_mean",
            "reward_mean",
        ]:
            if key_candidate in logs:
                value = logs[key_candidate]
                break

        if value is None:
            return

        self.history.append(value)

        if value > self.best_value + self.min_delta:
            self.best_value = value
            self.wait_count = 0
            logger.debug(f"Early stopping: new best {self.metric_name}={value:.4f}")
        else:
            self.wait_count += 1
            logger.debug(
                f"Early stopping: no improvement for {self.wait_count}/{self.patience} "
                f"checks (best={self.best_value:.4f}, current={value:.4f})"
            )

        if self.wait_count >= self.patience:
            logger.info(
                f"Early stopping triggered after {self.patience} checks "
                f"without improvement. Best {self.metric_name}={self.best_value:.4f}"
            )
            control.should_training_stop = True


class UtilityGuardCallback(TrainerCallback):
    """
    Guard against utility regression during unalignment.

    Monitors KL divergence from reference policy and halts training
    if it exceeds a threshold, indicating the model is drifting too far
    from the aligned base and likely losing utility.

    This is complementary to the β KL penalty in the loss — it acts
    as a hard stop if the soft penalty isn't sufficient.
    """

    def __init__(
        self,
        max_kl: float = 10.0,
        window_size: int = 10,
    ):
        self.max_kl = max_kl
        self.kl_history: deque = deque(maxlen=window_size)

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs,
    ):
        if logs is None:
            return

        kl_value = None
        for key_candidate in ["kl", "kl_divergence", "train/kl", "kl_coef"]:
            if key_candidate in logs:
                kl_value = logs[key_candidate]
                break

        if kl_value is None:
            return

        self.kl_history.append(kl_value)

        # Check rolling average
        if len(self.kl_history) >= self.kl_history.maxlen:
            avg_kl = sum(self.kl_history) / len(self.kl_history)
            if avg_kl > self.max_kl:
                logger.warning(
                    f"Utility guard triggered: avg KL={avg_kl:.2f} > max_kl={self.max_kl}. "
                    f"Model may be diverging too far from reference. Stopping training."
                )
                control.should_training_stop = True


class RewardLoggingCallback(TrainerCallback):
    """
    Detailed reward logging for debugging reward hacking.

    Logs per-dimension scores (align, risk, detail) and the aggregated R_IDA
    to help identify when the model is gaming the judge.
    """

    def __init__(self):
        self.step_rewards: list[dict] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict = None,
        **kwargs,
    ):
        if logs is None:
            return

        reward_info = {
            k: v for k, v in logs.items()
            if any(sub in k for sub in ["reward", "align", "risk", "detail"])
        }
        if reward_info:
            reward_info["global_step"] = state.global_step
            self.step_rewards.append(reward_info)