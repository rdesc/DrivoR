import torch
import torch.nn as nn
import torch.nn.functional as F


class GRPOLoss(nn.Module):
    """
    GRPO loss for fine-tuning the DrivoR scorer.

    Shared by Option 2 (dedicated grpo_head) and Option 3 (single selection_head).
    The policy is defined as softmax(logit) over K=64 candidate trajectories.
    The reference policy is a frozen clone of the scoring head at initialisation.

    Loss = clipped policy gradient (PPO-style) + beta * KL(policy || ref)
    Note: for Option 3 set beta=0.0 — KL against a random reference is meaningless.
    """

    def __init__(self, eps: float = 0.2, beta: float = 0.01, grpo_weight: float = 1.0):
        super().__init__()
        self.eps = eps
        self.beta = beta
        self.grpo_weight = grpo_weight

    def forward(
        self,
        grpo_logit: torch.Tensor,       # [B, K] — trainable head output
        ref_grpo_logit: torch.Tensor,   # [B, K] — frozen reference head output
        oracle_rewards: torch.Tensor,   # [B, K] — oracle PDMS scores
    ):
        """
        :param grpo_logit: policy logits from trainable grpo_head
        :param ref_grpo_logit: policy logits from frozen reference grpo_head
        :param oracle_rewards: per-proposal oracle PDMS scores in [0, 1]
        :return: (total_loss, loss_dict)
        """
        # group-relative advantage normalisation
        advantages = (oracle_rewards - oracle_rewards.mean(dim=-1, keepdim=True)) / \
                     (oracle_rewards.std(dim=-1, keepdim=True) + 1e-8)
        advantages = advantages.detach()

        log_probs = F.log_softmax(grpo_logit, dim=-1)               # [B, K]
        log_probs_ref = F.log_softmax(ref_grpo_logit.detach(), dim=-1)  # [B, K]

        # clipped importance-weighted policy gradient
        ratio = torch.exp(log_probs - log_probs_ref)
        clipped = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps)
        loss_grpo = -torch.mean(torch.min(ratio * advantages, clipped * advantages))

        # KL(policy || ref) = sum(policy * (log_policy - log_ref))
        kl = F.kl_div(log_probs_ref, log_probs.exp(), reduction='batchmean')

        loss = self.grpo_weight * (loss_grpo + self.beta * kl)

        # fraction of ratios that hit the PPO clip boundary — high values mean LR is too large
        clip_fraction = ((ratio - 1.0).abs() > self.eps).float().mean()

        # policy entropy over K candidates — collapses toward 0 if the policy mode-collapses
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

        # mean probability assigned to the argmax candidate — goes from 1/64≈0.016 (uniform) to 1.0 (collapsed)
        max_prob = log_probs.exp().max(dim=-1).values.mean()

        loss_dict = {
            "grpo_loss": loss_grpo,
            "grpo_kl": kl,
            "grpo_advantages_std": advantages.std(),
            "grpo_ratio_mean": ratio.mean(),
            "grpo_clip_fraction": clip_fraction,
            "grpo_entropy": entropy,
            "grpo_max_prob": max_prob,
            "reward_mean": oracle_rewards.mean(),
            "reward_std": oracle_rewards.std(),
        }

        return loss, loss_dict
