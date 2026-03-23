import torch
import torch.nn as nn
import torch.nn.functional as F


# Maps constraint names to indices in target_scores tensor [B, K, 7]
# (see compute_navsim_score.py lines 73-75 for ordering)
SCORE_COMPONENT_INDICES = {
    "no_at_fault_collisions": 0,
    "drivable_area_compliance": 1,
    "ego_progress": 2,
    "time_to_collision": 3,
    "ego_comfort": 4,
    "driving_direction_compliance": 5,
    # index 6 = final_scores (aggregate)
}


class GRPOLoss(nn.Module):
    """
    GRPO loss for fine-tuning the DrivoR scorer.

    The policy π is softmax(logit) over K=64 candidate trajectories.
    Computes the exact expectation over all K candidates (no sampling)
    using PPO-clip with a 1-step reference policy π_old:

        L = -Σ_a π_old(a) · min(r(a)·A(a), clip(r(a), 1±ε)·A(a))

    where r(a) = π(a)/π_old(a) and advantages are group-normalised oracle rewards.
    π_old is updated to the current policy before each optimizer step (1-step lag).

    Supports Constrained GRPO with scalarized advantages (use_constraints=True):
        A = λ_R · Z_R - Σ_k λ_k · Z_{C_k}
    where Z_j = group-standardize(component_j), and λ = softmax(logits) are
    Lagrangian multipliers updated via dual gradient ascent on violation rates.

    Also supports exact REINFORCE (no clipping) and sampled REINFORCE as
    commented-out alternatives.
    """

    def __init__(self, eps: float = 0.2, beta: float = 0.0, grpo_weight: float = 1.0,
                 entropy_coeff: float = 0.0, group_size: int = 8,
                 use_constraints: bool = False,
                 advantage_method: str = "scalarize_advantages",
                 constraint_names: list = None,
                 constraint_thresholds: list = None,
                 multiplier_lr: float = 0.03,
                 multiplier_init: float = 0.02):
        super().__init__()
        self.eps = eps
        self.beta = beta
        self.grpo_weight = grpo_weight
        self.entropy_coeff = entropy_coeff
        self.group_size = group_size

        # ---- Constrained GRPO ----
        self.use_constraints = use_constraints
        assert advantage_method in ("scalarize_advantages", "scalarize_rewards"), \
            f"advantage_method must be 'scalarize_advantages' or 'scalarize_rewards', got '{advantage_method}'"
        self.advantage_method = advantage_method
        if use_constraints:
            self.constraint_names = constraint_names or list(SCORE_COMPONENT_INDICES.keys())
            constraint_thresholds = constraint_thresholds or [0.01] * len(self.constraint_names)
            assert len(self.constraint_names) == len(constraint_thresholds), \
                f"len(constraint_names)={len(self.constraint_names)} != len(constraint_thresholds)={len(constraint_thresholds)}"

            self.constraint_indices = [SCORE_COMPONENT_INDICES[name] for name in self.constraint_names]

            # softmax-parameterized multipliers: [λ_R, λ_C1, λ_C2, ...]
            # nn.Parameter for state_dict (checkpoint resume) + automatic device placement.
            # Not in the policy optimizer — the agent only lists scorer params.
            n_components = len(self.constraint_names) + 1
            self.multiplier_logits = nn.Parameter(
                torch.full((n_components,), multiplier_init), requires_grad=True
            )
            self.register_buffer('_constraint_thresholds', torch.tensor(constraint_thresholds))
            self.multiplier_lr = multiplier_lr
            self.multiplier_signs = -1.0
            self._multiplier_optim = None  # lazy init after device placement

    @staticmethod
    def _group_standardize(x: torch.Tensor) -> torch.Tensor:
        """Standardize across dim=-1 (K candidates) per batch element."""
        return (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + 1e-5)

    def _get_multiplier_optim(self):
        """Lazy init — ensures Adam is created after PL moves the module to GPU."""
        if self._multiplier_optim is None:
            self._multiplier_optim = torch.optim.Adam(
                [self.multiplier_logits], lr=self.multiplier_lr, eps=1e-5
            )
        return self._multiplier_optim

    def _update_multipliers(self, probs: torch.Tensor, target_scores: torch.Tensor):
        """
        Update multiplier logits via Adam on the Lagrangian dual objective.

        L_mult = multiplier_signs · Σ_k λ_k · (violation_rate_k - threshold_k)
        where λ = softmax(multiplier_logits), multiplier_signs = -1.

        backward() computes ∂L/∂logits through the softmax Jacobian;
        Adam steps on the logits with adaptive LR and momentum.
        """
        # policy-weighted violation rates per constraint: E_π[1 - metric_k]
        violation_rates = torch.stack([
            (probs * (1.0 - target_scores[:, :, idx])).sum(dim=-1).mean()
            for idx in self.constraint_indices
        ]).detach()  # [num_constraints]

        # multiplier loss (autograd through softmax)
        multipliers = F.softmax(self.multiplier_logits, dim=0)[1:]  # [num_constraints]
        multiplier_loss = self.multiplier_signs * (multipliers * (violation_rates - self._constraint_thresholds)).sum()

        optim = self._get_multiplier_optim()
        optim.zero_grad()
        multiplier_loss.backward()
        optim.step()

        return violation_rates, F.softmax(self.multiplier_logits.detach(), dim=0)

    def forward(
        self,
        grpo_logit: torch.Tensor,       # [B, K] — trainable head output
        ref_grpo_logit: torch.Tensor,   # [B, K] — frozen reference head output, for PPO clipping
        oracle_rewards: torch.Tensor,   # [B, K] — oracle PDMS scores
        target_scores: torch.Tensor = None,  # [B, K, 7] — per-component scores (for constraints)
    ):

        # ---- full distribution (for metrics + entropy reg) ----
        log_probs = F.log_softmax(grpo_logit, dim=-1)               # [B, K]
        log_probs_ref = F.log_softmax(ref_grpo_logit.detach(), dim=-1)  # [B, K]
        probs = log_probs.exp()                                      # [B, K]

        # ---- advantage computation ----
        if self.use_constraints and target_scores is not None:
            lambdas = F.softmax(self.multiplier_logits.detach(), dim=0)
            lambda_R = lambdas[0]
            lambda_C = lambdas[1:]  # [num_constraints]

            if self.advantage_method == "scalarize_advantages":
                # Scalarize advantages: standardize each component independently, then combine.
                # A = λ_R · Z_R - Σ_k λ_k · Z_{C_k}
                # Preserves intended multiplier trade-offs (Theorem 4.1).
                Z_R = self._group_standardize(oracle_rewards)
                advantages = lambda_R * Z_R
                for i, idx in enumerate(self.constraint_indices):
                    cost_k = 1.0 - target_scores[:, :, idx]  # [B, K] violation indicator
                    Z_C_k = self._group_standardize(cost_k)
                    advantages = advantages - lambda_C[i] * Z_C_k
            else:
                # Scalarize rewards: combine into single scalar, then standardize.
                # R_s = λ_R · R - Σ_k λ_k · C_k
                # Subject to implicit reweighting by component variances (Theorem 4.1).
                R_s = lambda_R * oracle_rewards
                for i, idx in enumerate(self.constraint_indices):
                    cost_k = 1.0 - target_scores[:, :, idx]
                    R_s = R_s - lambda_C[i] * cost_k
                advantages = self._group_standardize(R_s)

            advantages = advantages.detach()
        else:
            # ---- original: single-reward group-relative advantage ----
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

        # ---- constraint multiplier update + logging ----
        if self.use_constraints and target_scores is not None:
            violation_rates, lambdas = self._update_multipliers(probs.detach(), target_scores)
            with torch.no_grad():
                loss_dict["cgrpo/multipliers/reward_weight"] = lambdas[0]
                loss_dict["cgrpo/raw_multipliers_values/reward_weight"] = self.multiplier_logits[0]
                for k, name in enumerate(self.constraint_names):
                    loss_dict[f"cgrpo/multipliers/{name}"] = lambdas[k + 1]
                    loss_dict[f"cgrpo/raw_multipliers_values/{name}"] = self.multiplier_logits[k + 1]
                    loss_dict[f"cgrpo/constraints/{name}"] = violation_rates[k]

        return loss, loss_dict
