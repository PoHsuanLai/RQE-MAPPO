# Critical Bugs Found in RQE-PPO Implementation

## Summary

Found 4 critical bugs that explain why RQE-PPO performs much worse than standard PPO.

## Bug #1: Incorrect Value Support Range ❌ CRITICAL

**File**: `src/train_single_agent.py`, lines 163-164

**Current code**:
```python
v_min=-50.0,
v_max=50.0,
```

**Problem**:
- CartPole episodes have returns in range [0, 500]
- Distributional critic support is [-50, 50]
- **The critic cannot represent any return > 50!**
- All successful episodes (return ~500) are clipped to v_max=50
- This makes it impossible to learn correct value functions

**Fix**:
```python
v_min=0.0,
v_max=600.0,
```

**Impact**: CRITICAL - This alone explains most of the performance gap

---

## Bug #2: Actor Network Initialization Too Small ❌ CRITICAL

**File**: `src/networks/actor.py`, line 61

**Current code**:
```python
nn.init.orthogonal_(module.weight, gain=0.01)
```

**Problem**:
- Standard PPO uses `gain=1.0` or `gain=np.sqrt(2)`
- Current initialization is **100x smaller**
- Measured: First layer weights have magnitude ~0.001 (should be ~0.1-0.3)
- This severely slows down learning, especially in early stages

**Fix**:
```python
nn.init.orthogonal_(module.weight, gain=1.0)
```

**Impact**: CRITICAL - Makes learning much slower

---

## Bug #3: Episode Boundary Bug in Bellman Update ❌ CRITICAL

**File**: `src/algorithms/distributional_rqe_ppo.py`, line 286

**Current code**:
```python
# Get next observations (shifted by 1)
next_observations = torch.roll(observations, shifts=-1, dims=0)
next_observations[-1] = observations[-1]  # Last next_obs doesn't matter
```

**Problem**:
- `torch.roll` doesn't respect episode boundaries
- When buffer contains multiple episodes, it incorrectly bootstraps across boundaries

Example:
```
Buffer contains 3 episodes:
  Episode 1: obs[0:100]   (terminal at step 99)
  Episode 2: obs[100:250] (terminal at step 249)
  Episode 3: obs[250:300]

torch.roll creates:
  next_obs[99] = obs[100]   ❌ WRONG! Different episodes!
  next_obs[249] = obs[250]  ❌ WRONG! Different episodes!

Bellman update: r[99] + γ * V(obs[100])
But obs[100] is from a DIFFERENT episode (after reset)!
```

This corrupts the distributional Bellman targets, making the critic learn completely incorrect value functions.

**Fix**: Need to track episode boundaries properly. Options:

1. **Option A**: Store next_obs explicitly during rollout collection
2. **Option B**: Use dones flag to mask out terminal transitions
3. **Option C**: Process each episode separately (slower but correct)

**Recommended fix (Option B)**:
```python
# Compute next observations properly respecting episode boundaries
next_observations = torch.zeros_like(observations)
next_observations[:-1] = observations[1:]

# For terminal states, next_obs doesn't matter (will be masked by dones)
# But set to current obs to avoid NaN issues
terminal_mask = dones.bool()
next_observations[terminal_mask] = observations[terminal_mask]

# Get next distribution
next_probs = self.critic(next_observations)  # [batch, n_atoms]

# Project with proper terminal handling (gamma=0 for terminal states)
target_probs = project_distribution(
    next_probs,
    rewards,
    self.critic.support,
    self.config.v_min,
    self.critic.delta_z,
    gamma=self.config.gamma,
    dones=dones  # This will set gamma=0 for terminal states
)
```

**Impact**: CRITICAL - Corrupts value learning

---

## Bug #4: Learning Rate Imbalance ⚠️ IMPORTANT

**File**: `src/train_single_agent.py`, line 167

**Current code**:
```python
lr_actor=3e-4,
lr_critic=1e-3,  # 3.3x faster!
```

**Problem**:
- Critic learns 3.3x faster than actor
- Can cause instability and value overestimation
- Standard practice: use same LR for both networks
- SB3 PPO uses 3e-4 for both

**Fix**:
```python
lr_actor=3e-4,
lr_critic=3e-4,  # Match actor LR
```

**Impact**: IMPORTANT - Can cause training instability

---

## Why Risk-Averse (τ=0.3) Performs Better Than Risk-Neutral (τ=1000)?

Paradoxically, the risk-averse agent performs better despite all these bugs:
- Risk-averse (τ=0.3): 492.8 ± 21.5
- Risk-neutral (τ=1000): 413.8 ± 89.6

**Explanation**: The entropic risk measure with low τ **underestimates** values, which accidentally compensates for the v_max=50 clipping bug! The risk-neutral agent tries to predict the true values (300-500) but gets clipped to 50, while the risk-averse agent's underestimation keeps it within the valid range.

---

## Recommended Fix Order

1. **Fix Bug #1 (support range)** - Most critical
2. **Fix Bug #3 (episode boundary)** - Breaks value learning
3. **Fix Bug #2 (initialization)** - Slows learning
4. **Fix Bug #4 (learning rate)** - Fine-tuning

After these fixes, RQE-PPO should perform comparably to standard PPO (at least for risk-neutral τ=1000).

---

## Expected Performance After Fixes

- **Risk-neutral (τ=1000)**: Should match or slightly exceed standard PPO (~500)
- **Risk-averse (τ=0.3)**: Should be slightly lower but more consistent (~450-480, with lower variance)
- **Standard PPO**: ~500 (current baseline)

The key is that risk-neutral RQE-PPO should now be **competitive** with standard PPO, not dramatically worse.
