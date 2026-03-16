# GRPO Fine-tuning of the DrivoR Scorer

## Motivation

The current scorer is trained via **imitation** of an oracle PDMS scorer (BCE loss). It learns to *predict* which trajectory is best, but is never directly rewarded for *selecting* high-PDMS trajectories. GRPO replaces this with a policy gradient objective that directly maximizes expected PDMS.

## Key Insight: The Group Already Exists

DrivoR generates K=64 candidate trajectories per scene. This maps naturally onto GRPO's group structure:

- **Policy**: the scorer, defining a categorical distribution over the K candidates via `softmax(pdm_score_logits)`
- **Group**: the K=64 candidate trajectories (deterministic, already produced by the trajectory decoder)
- **Reward**: oracle PDMS for each candidate (already computed via `compute_score()` in `drivor_agent.py:184`)
- **Reference policy**: the imitation-trained scorer checkpoint (frozen)

No architectural changes are needed. The oracle scores already exist in the training loop — they currently supervise BCE, and would instead become GRPO rewards.

## GRPO Objective

```python
# rewards: oracle PDMS for all K proposals — shape [batch, K]
rewards = oracle_pdms_scores

# group-relative advantage normalization
advantages = (rewards - rewards.mean(dim=-1, keepdim=True)) / \
             (rewards.std(dim=-1, keepdim=True) + 1e-8)

# log-probabilities under current and reference scorer
log_probs     = F.log_softmax(pdm_score_logits,     dim=-1)  # [batch, K]
log_probs_ref = F.log_softmax(pdm_score_logits_ref, dim=-1)  # [batch, K], frozen

# clipped GRPO loss
ratio     = torch.exp(log_probs - log_probs_ref)
loss_grpo = -torch.mean(
    torch.min(
        ratio * advantages,
        torch.clamp(ratio, 1 - eps, 1 + eps) * advantages
    )
)

# KL penalty against reference policy (optional but recommended)
kl = F.kl_div(log_probs, log_probs_ref.exp(), reduction='batchmean')
loss = loss_grpo + beta * kl
```

## Implementation Entry Points

| What | Where |
|------|-------|
| Replace/augment `final_score_loss` | `layers/losses/drivor_loss.py` |
| Pass frozen reference scorer | `drivor_agent.py` — load checkpoint, set `requires_grad=False` |
| Reuse oracle rewards | `drivor_agent.py:184` — `compute_score()` already returns per-proposal PDMS |
| Score logits | `drivor_model.py:190-197` — `pdm_score` before argmax |

## Limitations

**The generator stays frozen.** GRPO only updates the scorer's selection distribution over a fixed menu of K proposals. The quality of the update is bounded by the diversity of those proposals — if the generator produces near-identical trajectories, all rewards are similar and advantages ≈ 0, giving no learning signal.

The existing diversity loss in `drivoR.yaml` (`inter_weight: 0.0`) should be enabled to ensure the K proposals have meaningful spread in PDMS space before GRPO fine-tuning.

## Possible Extension: Joint Generator Update

To also improve the proposals themselves, a second loop is needed. One option:

1. **GRPO step**: update scorer given fixed proposals
2. **Generator step**: use scorer's updated preferences as a reward signal to update the trajectory decoder (requires continuous likelihoods over trajectory space, e.g. a Gaussian output head on the trajectory MLP)

This is a more significant change and should be treated as a separate stage after the scorer GRPO is validated.
