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
# KL(policy || ref) = sum(policy * (log_policy - log_ref))
kl = F.kl_div(log_probs_ref, log_probs.exp(), reduction='batchmean')
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

## Design Options for the Policy Logit

A key design question is what to use as the **policy logit** — the scalar per-proposal score fed into `softmax` to define the selection distribution.

### Current Scorer Architecture

```
proposals [B, 64, 8, 3]  (detached from generator)
    │
    └─► pos_embed (MLP: 24 → tf_d_ffn → tf_d_model=256)
            │
            ▼
    embedded_traj [B, 64, 256]
            │
            ▼
    TransformerDecoderScorer   (scorer_ref_num layers of self+cross-attn)
      cross-attends to scene_features [B, num_cams * num_scene_tokens, 256]
            │
            ▼
    tr_out [B, 64, 256]  +  ego_token
            │
            ▼
    Scorer.pred_score — 6 independent heads, each:
      Linear(256 → tf_d_ffn=1024) → ReLU → Linear(1024 → 1)
            │
    6 raw logits per proposal [B, 64] each
            │
            ▼
    pdm_score (composite, log-space):
      noc*log(σ(noc_logit)) + dac*log(σ(dac_logit)) + ddc*log(σ(ddc_logit))
      + log(ttc*σ(ttc_logit) + ep*σ(ep_logit) + comfort*σ(comfort_logit))
            │
            ▼
    argmax(pdm_score) → selected trajectory
```

Trainable scorer params: `pos_embed` + `scorer_attention` (TransformerDecoderScorer) + `Scorer.pred_score` (6 heads).

Note that `pdm_score` is **not a plain linear logit** — it is a log-space composite of sigmoid outputs from 6 independent heads. Using it as the GRPO policy logit is valid (it defines a well-formed categorical distribution via softmax), but the GRPO gradient flows back through each head independently, scaled by its config weight (`noc`, `dac`, etc.). This has implications for head interpretability post fine-tuning (see options below).

### Option 1 — Joint BCE + GRPO

Keep the 6 BCE heads and their supervision, and add a GRPO term on top of `pdm_score`.

```
Loss = trajectory_weight * L_traj
     + bce_weight        * L_BCE        ← same as today (anchors heads to metric meaning)
     + grpo_weight       * L_GRPO       ← new: policy gradient on pdm_score logits
     + beta              * KL(π || π_ref)
```

GRPO treats `pdm_score` as the policy logit and pushes probability mass toward high-reward proposals. BCE simultaneously keeps each head calibrated to its metric. The two objectives can conflict (a head may want to move its logit for BCE reasons while GRPO wants to move the composite `pdm_score`), but BCE acts as a regularizer that prevents the heads from drifting into arbitrary values.

**Pros**
- Minimal code change — just add `L_GRPO` to the existing loss
- Heads remain interpretable
- BCE provides stable gradient signal even for samples with low reward variance (advantages ≈ 0)

**Cons**
- Two objectives pulling on the same 6 heads — can cause conflicting gradients
- The config weights (`noc`, `dac`, `ttc`, `ep`, `comfort`) bake in fixed gradient scaling per head during GRPO updates
- Hard to disentangle how much improvement comes from GRPO vs. continued BCE training

### Option 2 — Dedicated Scalar Scoring Head (Recommended)

Add a new single-output MLP head trained **only** by GRPO. Freeze the 6 BCE heads entirely. Use the new head's output as the policy logit.

```
tr_out [B, 64, 256]
    │
    ├─► pred_score (6 BCE heads, FROZEN) ─► pdm_score (used at inference, unchanged)
    │
    └─► grpo_head: Linear(256→256) → ReLU → Linear(256→1)   ← NEW, trainable only by GRPO
            │
            ▼
        grpo_logit [B, 64]   ← policy logit for GRPO
            │
            ▼
        L_GRPO + KL(softmax(grpo_logit) || softmax(grpo_logit_ref))
```

At inference, selection can use either `pdm_score` (original) or `grpo_logit` (fine-tuned), enabling a direct ablation.

**Pros**
- Clean separation: BCE heads do metric prediction, `grpo_head` does selection
- `grpo_head` has full freedom to learn whatever ranking function maximises PDMS — not constrained by the 6-metric decomposition
- No conflicting gradients on existing parameters
- Interpretability of the 6 heads fully preserved
- Clean ablation: compare `pdm_score` vs `grpo_logit` selection on the same checkpoint

**Cons**
- `grpo_head` starts from random init, so early updates may be noisy
- Requires a separate inference switch to activate the new head
- The `grpo_head` input (`tr_out`) is produced by the frozen scorer attention — if the frozen representation is not rich enough for ranking, the head is bottlenecked

### Option 3 — Single Softmax Head, Trained from Scratch

Replace the 6 BCE heads entirely with a single head that directly outputs a **categorical distribution over the 64 candidates** via softmax. The model is trained from scratch using GRPO as the sole objective — no BCE supervision, no pretrained scorer weights.

```
proposals [B, 64, 8, 3]
    │
    └─► pos_embed → TransformerDecoderScorer (trained from scratch)
            │
            ▼
    tr_out [B, 64, 256]
            │
            └─► selection_head: Linear(256→256) → ReLU → Linear(256→1)
                        │
                        ▼
                logit [B, 64]
                        │
                        ▼
                softmax(logit) → categorical distribution over 64 actions
                        │
                        ▼
                L_GRPO(logit, oracle_pdms_rewards)
                  + KL(softmax(logit) || softmax(logit_ref))
```

The model no longer decomposes into 6 interpretable sub-scores. Instead, it learns a single end-to-end ranking function directly optimized for PDMS. The reference policy `π_ref` is the randomly initialized checkpoint at the start of training (or a warmup checkpoint after a few steps).

**Pros**
- Cleanest formulation — the scorer's sole objective is trajectory selection, not metric prediction
- No tension between BCE supervision and GRPO; the head has full freedom to learn any ranking
- Simpler architecture: one head instead of six
- Potentially learns richer ranking signals that don't decompose neatly into the 6 PDM metrics

**Cons**
- No pretrained initialization — training from scratch requires more data and is less stable early on
- Loses the 6 interpretable sub-scores entirely; no per-metric diagnostics
- GRPO alone may be a weak signal early in training when the policy is near-uniform (advantages are small if reward variance is low)
- The transformer backbone still needs initialization (e.g. from a pretrained image encoder) even if scorer heads are random

### Comparison

| | Option 1 (Joint BCE+GRPO) | Option 2 (Dedicated Head) | Option 3 (Single Head, Scratch) |
|---|---|---|---|
| Head interpretability | Degrades over time | Fully preserved | None (by design) |
| Conflicting gradients | Yes | No | No |
| Training stability | Higher (BCE anchors) | Lower at start | Lowest — no warm start |
| Pretrained scorer needed | Yes | Yes | No |
| Implementation complexity | Low | Low–medium | Low |
| Clean ablation | Hard | Easy | Easy (compare vs Option 2) |
| Inference change needed | No | Yes | Yes |

Option 2 is the preferred design for fine-tuning a pretrained scorer. Option 3 is the right framing if the goal is to train a selection policy end-to-end without any metric decomposition assumptions.

## Possible Extension: Joint Generator Update

To also improve the proposals themselves, a second loop is needed. One option:

1. **GRPO step**: update scorer given fixed proposals
2. **Generator step**: use scorer's updated preferences as a reward signal to update the trajectory decoder (requires continuous likelihoods over trajectory space, e.g. a Gaussian output head on the trajectory MLP)

This is a more significant change and should be treated as a separate stage after the scorer GRPO is validated.