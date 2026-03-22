import torch
import torch.nn as nn
import torch.nn.functional as F


class GRPOLoss(nn.Module):
    """
    GRPO loss for fine-tuning the DrivoR scorer.

    The policy is defined as softmax(logit) over K=64 candidate trajectories.
    At each step we SAMPLE a group of G candidates from the policy, compute
    oracle rewards for the sampled group, normalise advantages within the group,
    and update with REINFORCE (or PPO-clip).

    The reference policy is a frozen clone of the scoring head at initialisation.
    """

    def __init__(self, eps: float = 0.2, beta: float = 0.0, grpo_weight: float = 1.0,
                 entropy_coeff: float = 0.0, group_size: int = 8):
        super().__init__()
        self.eps = eps
        self.beta = beta
        self.grpo_weight = grpo_weight
        self.entropy_coeff = entropy_coeff
        self.group_size = group_size

    def forward(
        self,
        grpo_logit: torch.Tensor,       # [B, K] — trainable head output
        ref_grpo_logit: torch.Tensor,   # [B, K] — frozen reference head output, for PPO clipping
        oracle_rewards: torch.Tensor,   # [B, K] — oracle PDMS scores
    ):
        # ---- full distribution (for metrics + entropy reg) ----
        log_probs = F.log_softmax(grpo_logit, dim=-1)               # [B, K]
        log_probs_ref = F.log_softmax(ref_grpo_logit.detach(), dim=-1)  # [B, K]
        probs = log_probs.exp()                                      # [B, K]

        # ---- group-relative advantage normalisation (over all K candidates) ----
        advantages = (oracle_rewards - oracle_rewards.mean(dim=-1, keepdim=True)) / \
                     (oracle_rewards.std(dim=-1, keepdim=True) + 1e-5)
        advantages = advantages.detach()

        # ---- PPO-clip with exact expectation (weighted by π_old) ----
        ratio = torch.exp(log_probs - log_probs_ref)
        clipped = torch.clamp(ratio, 1.0 - self.eps, 1.0 + self.eps)
        probs_ref = log_probs_ref.exp().detach()                        # [B, K]
        loss_grpo = -(probs_ref * torch.min(ratio * advantages, clipped * advantages)).sum(dim=-1).mean()

        # # ---- exact policy gradient (no clipping, REINFORCE): ----
        # loss_grpo = -(probs.detach() * log_probs * advantages).sum(dim=-1).mean()

        # # ---- sampled REINFORCE (alternative): sample G candidates, use only those ----
        # with torch.no_grad():
        #     sampled_idx = torch.multinomial(probs, self.group_size, replacement=True)  # [B, G]
        # sampled_log_probs = log_probs.gather(1, sampled_idx)          # [B, G]
        # sampled_rewards = oracle_rewards.gather(1, sampled_idx)       # [B, G]
        # advantages_sampled = (sampled_rewards - sampled_rewards.mean(dim=-1, keepdim=True)) / \
        #                      (sampled_rewards.std(dim=-1, keepdim=True) + 1e-5)
        # advantages_sampled = advantages_sampled.detach()
        # loss_grpo = -torch.mean(sampled_log_probs * advantages_sampled)

        # ---- optional regularisation ----
        # KL (for logging; only added to loss if beta > 0)
        kl = F.kl_div(log_probs_ref, probs, reduction='batchmean')

        # entropy over full distribution
        entropy = -(probs * log_probs).sum(dim=-1).mean()

        loss = loss_grpo

        if self.beta > 0:
            loss = loss + self.beta * kl

        if self.entropy_coeff > 0:
            loss = loss - self.entropy_coeff * entropy

        # ---- metrics ----
        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > self.eps).float().mean()
            max_prob = probs.max(dim=-1).values.mean()

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
            "grpo_logit_norm": grpo_logit.float().norm(dim=-1).mean(),
        }

        return loss, loss_dict
