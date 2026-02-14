import pytest
import torch


def compute_advantages(rewards: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Compute standardized group-relative advantages.

    Paper: Â_i = (r_i - μ_G) / (σ_G + ε)
    """
    mu = rewards.mean()
    sigma = rewards.std()
    return (rewards - mu) / (sigma + epsilon)


def dapo_loss(log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """
    DAPO loss: only reinforce positive-advantage rollouts.

    Paper: L_DAPO = -E[ (1/G) Σ 1[Â_i > 0] * Â_i * log π_θ(o_i|p) ]
    """
    mask = (advantages > 0).float()
    return -torch.mean(mask * advantages * log_probs)


class TestComputeAdvantages:
    def test_zero_mean(self):
        """Advantages should have zero mean."""
        rewards = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        adv = compute_advantages(rewards)
        assert abs(adv.mean().item()) < 1e-6

    def test_unit_variance(self):
        """Advantages should have approximately unit variance."""
        rewards = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
        adv = compute_advantages(rewards)
        assert abs(adv.std().item() - 1.0) < 0.1

    def test_ordering_preserved(self):
        """Higher rewards should get higher advantages."""
        rewards = torch.tensor([0.1, 0.5, 0.9])
        adv = compute_advantages(rewards)
        assert adv[0] < adv[1] < adv[2]

    def test_all_same_rewards(self):
        """If all rewards are equal, advantages should be ~0 (not NaN)."""
        rewards = torch.tensor([0.5, 0.5, 0.5, 0.5])
        adv = compute_advantages(rewards)
        assert not torch.isnan(adv).any()
        assert torch.allclose(adv, torch.zeros_like(adv), atol=1e-4)

    def test_single_rollout(self):
        """Single rollout should give advantage ~0."""
        rewards = torch.tensor([0.7])
        adv = compute_advantages(rewards)
        assert not torch.isnan(adv).any()

    def test_group_size_8(self):
        """Paper uses G=8 rollouts per prompt."""
        rewards = torch.tensor([0.0, 0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9])
        adv = compute_advantages(rewards)
        assert adv.shape == (8,)
        assert not torch.isnan(adv).any()
        # Highest reward should have highest advantage
        assert adv.argmax() == rewards.argmax()


class TestDAPOLoss:
    def test_ignores_negative_advantages(self):
        """DAPO should only reinforce positive-advantage rollouts."""
        log_probs = torch.tensor([-1.0, -1.5, -2.0, -0.5])
        advantages = torch.tensor([-1.0, -0.5, 0.5, 1.0])

        loss = dapo_loss(log_probs, advantages)

        # Only indices 2 and 3 should contribute (positive advantage)
        expected = -torch.mean(
            torch.tensor([0.0, 0.0, 0.5 * (-2.0), 1.0 * (-0.5)])
        )
        assert abs(loss.item() - expected.item()) < 1e-6

    def test_all_negative_advantages(self):
        """If all advantages are negative, loss should be 0."""
        log_probs = torch.tensor([-1.0, -1.5, -2.0])
        advantages = torch.tensor([-1.0, -0.5, -0.3])
        loss = dapo_loss(log_probs, advantages)
        assert abs(loss.item()) < 1e-6

    def test_all_positive_advantages(self):
        """If all advantages are positive, all should contribute."""
        log_probs = torch.tensor([-1.0, -1.5])
        advantages = torch.tensor([0.5, 1.0])
        loss = dapo_loss(log_probs, advantages)
        expected = -torch.mean(advantages * log_probs)
        assert abs(loss.item() - expected.item()) < 1e-6

    def test_gradient_flows(self):
        """Loss should produce valid gradients."""
        log_probs = torch.tensor([-1.0, -1.5, -2.0], requires_grad=True)
        advantages = torch.tensor([-0.5, 0.3, 0.8])
        loss = dapo_loss(log_probs, advantages)
        loss.backward()
        assert log_probs.grad is not None
        # Gradient should be zero for negative-advantage rollout
        assert abs(log_probs.grad[0].item()) < 1e-6

    def test_early_training_mostly_refusals(self):
        """
        Simulate early training: most rollouts refuse (low reward),
        one partially complies (higher reward).
        DAPO should only reinforce the compliant rollout.
        """
        # 7 refusals, 1 partial compliance
        rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3])
        advantages = compute_advantages(rewards)

        # Only the last rollout should have positive advantage
        positive_mask = (advantages > 0).float()
        assert positive_mask.sum() <= 2  # at most 1-2 positive

        log_probs = torch.randn(8, requires_grad=True)
        loss = dapo_loss(log_probs, advantages)
        loss.backward()

        # Gradients for refusal rollouts should be ~0
        for i in range(7):
            if advantages[i] <= 0:
                assert abs(log_probs.grad[i].item()) < 1e-6