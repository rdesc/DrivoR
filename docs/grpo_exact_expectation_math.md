# The Exact Expectation Fix: Why We Multiply by $\pi(a)$

## The Policy Gradient Theorem

The objective we want to maximize is the expected reward under our policy:

$$J(\theta) = \mathbb{E}_{a \sim \pi_\theta}[R(a)] = \sum_a \pi_\theta(a) \cdot R(a)$$

Taking the gradient:

$$\nabla_\theta J(\theta) = \sum_a \nabla_\theta \pi_\theta(a) \cdot R(a)$$

Using the log-derivative trick, $\nabla \pi(a) = \pi(a) \cdot \nabla \log \pi(a)$:

$$\nabla_\theta J(\theta) = \sum_a \pi_\theta(a) \cdot \nabla_\theta \log \pi_\theta(a) \cdot R(a) \qquad \text{...(1)}$$

This is the **exact policy gradient**. It sums over all actions, weighted by $\pi(a)$.

## How Standard REINFORCE Avoids the $\pi(a)$ Weight

Equation (1) is an expectation under $\pi$:

$$\nabla_\theta J(\theta) = \mathbb{E}_{a \sim \pi_\theta}\left[\nabla_\theta \log \pi_\theta(a) \cdot R(a)\right]$$

Standard REINFORCE **samples** $a \sim \pi_\theta$ to get an unbiased estimate:

$$\hat{\nabla} J(\theta) = \nabla_\theta \log \pi_\theta(a_{\text{sampled}}) \cdot R(a_{\text{sampled}})$$

The $\pi(a)$ weighting is implicit — by sampling from $\pi$, high-probability actions appear more often, so they automatically contribute more to the gradient. **You never need to explicitly multiply by $\pi(a)$.**

This is why every standard RL implementation (Atari, LLMs, robotics) just does:

```python
loss = -log_prob(a_sampled) * advantage
```

No $\pi(a)$ factor anywhere. The sampling handles it.

## Our Setup: All 64 Candidates Are Available

In our scorer, the "actions" are $K=64$ candidate trajectories. We have oracle rewards for **all** of them (from the PDMS scorer). We don't need to sample — we can compute the exact sum.

But if we compute the exact sum, we **must** include the $\pi(a)$ weight:

$$\nabla_\theta J(\theta) = \sum_a \pi_\theta(a) \cdot \nabla_\theta \log \pi_\theta(a) \cdot A(a) \qquad \text{(exact)}$$

The corresponding loss (whose gradient gives the above):

```python
loss = -(probs.detach() * log_probs * advantages).sum(dim=-1).mean()
```

The `probs.detach()` is $\pi(a)$ — detached because it's a weight on the gradient, not something we differentiate through.

## The Bug: What Happens Without $\pi(a)$

The original code did:

```python
# BUG: missing π(a) weighting
loss = -(log_probs * advantages).mean()
```

This computes:

$$\nabla L = -\frac{1}{K} \sum_a \nabla_\theta \log \pi_\theta(a) \cdot A(a)$$

Every action contributes equally to the gradient, regardless of its probability under the policy. This is **not** a valid policy gradient estimator.

### Why This Is Wrong — Concrete Example

Suppose $K=3$ actions with advantages $A = [+10, -1, -1]$:
- Action 0 is great, actions 1 and 2 are slightly bad.

**Without $\pi(a)$ weighting** (the bug):

The gradient pushes equally on all three log-probs. Even if $\pi(a_0) = 0.98$, the gradient from actions 1 and 2 is just as strong as from action 0. The policy wastes gradient signal adjusting probabilities of actions it has already learned to avoid.

**With $\pi(a)$ weighting** (correct):

If $\pi(a_0) = 0.98$, then:
- Action 0 contributes: $0.98 \cdot \nabla \log \pi(a_0) \cdot (+10) \rightarrow$ strong push
- Action 1 contributes: $0.01 \cdot \nabla \log \pi(a_1) \cdot (-1) \rightarrow$ negligible
- Action 2 contributes: $0.01 \cdot \nabla \log \pi(a_2) \cdot (-1) \rightarrow$ negligible

The policy focuses its gradient on the actions that actually matter under the current distribution.

### Why the Bug Caused Training Instability

Without $\pi(a)$ weighting, low-probability actions with large advantages generate large gradients. As training progresses:

1. The policy becomes more peaked (some actions get very low probability)
2. But those low-probability actions still generate full-strength gradients
3. These gradients push the logits around erratically
4. Logit norms grow, entropy collapses, and eventually NaN

With proper $\pi(a)$ weighting, low-probability actions contribute near-zero gradient. The policy self-stabilizes: once it's confident about an action, it stops wasting gradient on alternatives.

## Extension to PPO-Clip

Standard PPO with sampling:

$$L^{\text{CLIP}} = \mathbb{E}_{a \sim \pi_{\text{old}}}\left[\min\left(r(a) \cdot A(a),\; \text{clip}(r(a), 1 \pm \varepsilon) \cdot A(a)\right)\right]$$

where $r(a) = \frac{\pi_\theta(a)}{\pi_{\text{old}}(a)}$. Again, sampling from $\pi_{\text{old}}$ provides the weighting.

Exact expectation PPO (our case):

$$L^{\text{CLIP}} = \sum_a \pi_{\text{old}}(a) \cdot \min\left(r(a) \cdot A(a),\; \text{clip}(r(a), 1 \pm \varepsilon) \cdot A(a)\right)$$

The $\pi_{\text{old}}(a)$ weighting plays the same role as $\pi(a)$ in REINFORCE — it's the distribution we're taking the expectation under.

Note: for the unclipped branch, $\pi_{\text{old}} \cdot r \cdot A = \pi_{\text{old}} \cdot \frac{\pi}{\pi_{\text{old}}} \cdot A = \pi \cdot A$, recovering the REINFORCE form. The clipping only matters when $r$ exceeds $[1-\varepsilon, 1+\varepsilon]$.

## Summary

| Setting | Loss | Why it works |
|---------|------|-------------|
| Sample $a \sim \pi$ | $-\log \pi(a) \cdot A(a)$ | Sampling provides implicit $\pi(a)$ weighting |
| Sum over all actions | $-\sum_a \pi(a) \cdot \log \pi(a) \cdot A(a)$ | Must explicitly weight by $\pi(a)$ |
| PPO with sampling from $\pi_{\text{old}}$ | $-\min(r \cdot A,\; \text{clip}(r) \cdot A)$ | Sampling provides implicit $\pi_{\text{old}}$ weighting |
| PPO exact expectation | $-\sum_a \pi_{\text{old}}(a) \cdot \min(r \cdot A,\; \text{clip}(r) \cdot A)$ | Must explicitly weight by $\pi_{\text{old}}(a)$ |

**The rule**: if you're not sampling from the distribution, you must weight by it.
