# Baseline Implementations for RQE-MAPPO Comparison

## Overview
To establish novelty and show empirical advantages, you need to implement and compare against existing risk-averse RL methods. This document outlines what to implement.

---

## Baselines to Implement

### 1. **Standard MAPPO** (Risk-Neutral)
**What it is**: Your existing MAPPO implementation with τ → ∞ (no risk-aversion)

**Implementation**:
- ✅ Already have this (it's your baseline)
- Just standard PPO with entropy regularization
- Expected value function: V(s) = E[return]

**Expected behavior**:
- Highest mean return
- Highest variance
- Most collisions/failures

---

### 2. **C51-CVaR MAPPO** (Distributional + CVaR)
**What it is**: Use distributional critic (C51) but extract CVaR instead of entropic risk

**Implementation**:
```python
class C51_CVaR_MAPPO:
    def __init__(self, alpha=0.1):  # CVaR at 10%
        self.critic = DistributionalCritic(n_atoms=51)  # Same as yours!
        self.alpha = alpha

    def get_cvar_value(self, obs):
        probs = self.critic(obs)  # [batch, 51]
        support = self.critic.support  # [51]

        # CVaR: average of worst alpha% outcomes
        cumsum = torch.cumsum(probs, dim=-1)
        mask = (cumsum <= self.alpha).float()

        cvar = (probs * support * mask).sum(dim=-1) / mask.sum(dim=-1)
        return cvar

    def compute_advantages(self, batch):
        # Use CVaR value instead of entropic risk
        values = self.get_cvar_value(batch.obs)
        # Rest is standard GAE with CVaR values
        ...
```

**Key differences from yours**:
- Same distributional critic architecture
- Different risk measure: CVaR vs entropic
- No bounded rationality framing (just standard entropy bonus)

**Expected behavior**:
- Lower variance than MAPPO
- But CVaR is very pessimistic (might be too conservative)
- No equilibrium guarantees

---

### 3. **RMIX** (Risk-Sensitive Value Factorization)
**What it is**: Multi-agent value factorization (like QMIX) but with CVaR objective

**Reference**: Qiu et al., "RMIX: Learning Risk-Sensitive Policies for Cooperative Reinforcement Learning Agents" (NeurIPS 2021)

**Implementation**:
```python
class RMIX:
    def __init__(self, alpha=0.1):
        self.agent_networks = [AgentNetwork() for _ in range(n_agents)]
        self.mixing_network = HyperNetwork()  # Mixes individual Q-values
        self.alpha = alpha  # CVaR level

    def compute_qtot(self, agent_qs, state):
        # Mix individual Q-values into total Q
        qtot = self.mixing_network(agent_qs, state)
        return qtot

    def compute_cvar_loss(self, batch):
        # Compute CVaR of joint Q-values
        qtot = self.compute_qtot(agent_qs, state)

        # Sort returns and take worst alpha%
        sorted_returns = torch.sort(qtot)[0]
        k = int(self.alpha * len(sorted_returns))
        cvar = sorted_returns[:k].mean()

        # TD loss with CVaR target
        loss = (qtot - cvar_target)**2
        return loss
```

**Key differences from yours**:
- Value factorization architecture (like QMIX)
- CVaR risk measure (vs entropic)
- Designed for cooperative tasks with shared reward
- No bounded rationality component

**Expected behavior**:
- Works well for cooperative tasks
- More conservative than standard QMIX
- But: no equilibrium guarantees, CVaR can be overly pessimistic

---

### 4. **Mean-Variance MAPPO**
**What it is**: Maximize E[R] - λ·Var[R]

**Implementation**:
```python
class MeanVariance_MAPPO:
    def __init__(self, lambda_var=0.5):
        self.value_net = ValueNetwork()  # Predicts E[R]
        self.variance_net = ValueNetwork()  # Predicts Var[R]
        self.lambda_var = lambda_var

    def get_risk_adjusted_value(self, obs):
        mean = self.value_net(obs)
        variance = self.variance_net(obs)
        return mean - self.lambda_var * variance

    def update_critic(self, batch):
        # Two separate losses
        mean_target = compute_returns(batch)
        var_target = compute_variance_returns(batch)

        loss = mse(self.value_net(obs), mean_target) + \
               mse(self.variance_net(obs), var_target)
```

**Key differences**:
- Two separate networks (mean and variance)
- Linear combination (not distributional)
- Hand-tune λ parameter (vs your principled τ)

**Expected behavior**:
- Reduces variance
- But λ needs manual tuning per environment
- Theory: less principled than convex risk measures

---

### 5. **Reward-Shaped MAPPO** (Manual Safety)
**What it is**: Standard MAPPO but add manual penalty for unsafe states

**Implementation**:
```python
class RewardShaped_MAPPO:
    def __init__(self, collision_penalty=-10.0):
        self.mappo = StandardMAPPO()
        self.collision_penalty = collision_penalty

    def shaped_reward(self, obs, action, next_obs, reward, done, info):
        shaped_r = reward

        # Manual penalties (environment-specific!)
        if 'collision' in info:
            shaped_r += self.collision_penalty

        if 'near_obstacle' in info:
            shaped_r += -1.0  # proximity penalty

        if 'high_speed_near_obstacle' in info:
            shaped_r += -5.0  # combined penalty

        return shaped_r
```

**Key differences**:
- Modifies reward function (not objective)
- Requires manual engineering per environment
- No theoretical guarantees

**Expected behavior**:
- Can achieve safety if penalties are tuned well
- But penalties are environment-specific (doesn't generalize)
- Need to re-tune for each new environment

---

## Experimental Setup

### Metrics to Collect (for ALL baselines)

```python
metrics = {
    'mean_return': [],           # Average episode return
    'std_return': [],            # Standard deviation of returns
    'min_return': [],            # Worst episode
    'percentile_5_return': [],   # 5th percentile (tail risk)
    'collision_rate': [],        # % episodes with collision
    'success_rate': [],          # % episodes reaching goal
    'steps_to_goal': [],         # Efficiency metric
}
```

### Evaluation Protocol

1. **Train all methods** for same number of timesteps (e.g., 1M steps)
2. **Evaluate** each on 100 test episodes (fresh seeds)
3. **Report**: mean ± std over 5 random seeds

### Expected Result Table

| Method | Mean Return ↑ | Std ↓ | Collision Rate ↓ | 5th %ile Return ↑ |
|--------|---------------|-------|------------------|-------------------|
| MAPPO | **100** | 25 | 15% | 20 |
| C51-CVaR | 85 | 12 | 8% | 50 |
| RMIX | 88 | 13 | 7% | 48 |
| Mean-Var | 90 | 15 | 10% | 45 |
| Reward-Shaped | 92 | 18 | 7% | 40 |
| **RQE (ours)** | 95 | **10** | **5%** | **55** |

**Key comparisons**:
- **vs MAPPO**: Lower variance, fewer catastrophic failures
- **vs C51-CVaR**: Better mean return (CVaR too pessimistic)
- **vs RMIX**: Works in mixed settings (RMIX designed for cooperative only)
- **vs Mean-Var**: More principled (τ vs hand-tuned λ)
- **vs Reward-Shaped**: Generalizes (same τ across environments)

---

## Implementation Priority

### Phase 1: Quick Baselines (1 week)
1. ✅ Standard MAPPO (already have)
2. C51-CVaR MAPPO (just change risk measure in your code)
3. Mean-Variance MAPPO (two-head network)

### Phase 2: Full Comparison (2 weeks)
4. RMIX (value factorization + CVaR)
5. Reward-Shaped MAPPO (manual engineering)
6. Run all experiments on CartPole
7. Generate comparison plots

### Phase 3: Second Environment (1 week)
8. Implement SUMO traffic environment
9. Test if RQE generalizes (same τ works)
10. Show reward shaping needs re-tuning

---

## Code Structure

```
src/
├── algorithms/
│   ├── distributional_rqe_ppo.py    # Your implementation
│   ├── standard_mappo.py             # Baseline 1
│   ├── c51_cvar_mappo.py            # Baseline 2
│   ├── rmix.py                       # Baseline 3
│   ├── mean_variance_mappo.py       # Baseline 4
│   └── reward_shaped_mappo.py       # Baseline 5
├── experiments/
│   ├── compare_baselines.py         # Main comparison script
│   └── plot_results.py              # Generate comparison plots
└── configs/
    ├── cartpole_config.yaml
    └── sumo_config.yaml
```

---

## Key Arguments for Your Contribution

### 1. Theoretical Novelty
**Claim**: "First tractable risk-averse equilibrium for MARL"

**Support**:
- C51-CVaR: No equilibrium concept
- RMIX: No equilibrium guarantees (value factorization heuristic)
- Mean-Variance: No multi-agent theory
- Reward Shaping: No formal guarantees
- **RQE**: Provably computable via no-regret learning

### 2. Empirical Advantage
**Claim**: "Outperforms prior methods on safety metrics"

**Support**:
- Lower variance than all baselines
- Fewer catastrophic failures (tail risk)
- Better safety-efficiency tradeoff

### 3. Practical Advantage
**Claim**: "Single parameter generalizes across environments"

**Support**:
- Same τ=0.3 works on CartPole AND SUMO
- Mean-Variance needs different λ per environment
- Reward shaping needs complete re-engineering per environment

---

## Timeline

**Week 1-2**: Implement baselines 2-5
**Week 3-4**: Run CartPole experiments, collect metrics
**Week 5**: Implement SUMO environment
**Week 6**: Run SUMO experiments
**Week 7**: Generate plots, write results

**Total**: 7 weeks to full experimental validation

---

## Potential Issues & Solutions

### Issue 1: "C51-CVaR might perform similarly to RQE"
**Solution**: Emphasize theoretical contribution (equilibrium vs no theory)

### Issue 2: "RMIX might perform well on cooperative tasks"
**Solution**: Show RQE works on mixed cooperative-competitive (RMIX only for cooperative)

### Issue 3: "Mean-Variance might work well"
**Solution**: Show it needs different λ per environment (doesn't generalize)

### Issue 4: "Reward shaping works if tuned well"
**Solution**: Show it requires manual engineering, RQE is principled

### Issue 5: "Results are close"
**Solution**: Focus on tail risk and theoretical guarantees as differentiators

---

## Minimal Viable Comparison

If time is limited, implement **at minimum**:
1. Standard MAPPO (baseline)
2. C51-CVaR MAPPO (closest prior work)
3. RQE-MAPPO (yours)

This shows:
- ✓ Risk-aversion helps (vs MAPPO)
- ✓ Entropic risk + bounded rationality better than CVaR alone
- ✓ Theoretical grounding (equilibrium concept)

Good luck! 🚀
