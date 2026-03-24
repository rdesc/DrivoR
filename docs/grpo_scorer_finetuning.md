# GRPO Option 3 — Scorer Fine-tuning

## Scorer Architecture

**Trainable params: ~4.57M / 45.4M total** (everything else frozen)

```
Proposals [B, K=64, 8, 3]
       │
       ▼  reshape to [B, K, 24]
┌──────────────┐
│  pos_embed   │  MLP: Linear(24→1024) + ReLU + Linear(1024→256)
│  288K params │
└──────┬───────┘
       │ [B, K, 256]
       ▼
┌──────────────────────┐     scene_features [B, S, 256]
│  scorer_attention    │◄─── (frozen encoder output, cross-attn key/value)
│  4-layer transformer │
│  ~2.63M params       │     Each layer:
│                      │       1. Self-attn (1 head, d=256)
│                      │       2. Cross-attn (1 head, d=256)
│                      │       3. FFN (256→1024→256, GELU)
└──────┬───────────────┘
       │ [B, K, 256]
       ▼  + ego_token [B, 1, 256]  (frozen, broadcast-added)
┌──────────────────┐
│  selection_head  │  MLP: Linear(256→256) + ReLU + Linear(256→1)
│  66K params      │
└──────┬───────────┘
       │ squeeze(-1)
       ▼
  selection_logit [B, K]  ← raw logits, no activation
       │
       ▼  softmax → π(a|s)
  GRPO loss (PPO-clip against π_old)
```

**Reference policy (π_old):** deep copies of all 3 modules above, updated to current weights before each optimizer step (1-step lag).

**Frozen inputs to scorer:**
- `scene_features` — from ViT image backbone + trajectory decoder
- `ego_token` — from trajectory decoder
- `proposals` — K=64 candidate trajectories from generator

**Key dimensions:** `tf_d_model=256`, `tf_d_ffn=1024`, `scorer_ref_num=4` layers, `num_heads=1`
