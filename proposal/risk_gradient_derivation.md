# Risk-Adjusted Policy Gradient Derivation

## Complete Mathematical Derivation

### Setup

**Notation:**
- Policy: $\pi_\theta(a|s)$
- Trajectory: $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, \ldots)$
- Discounted return: $G(\tau) = \sum_{t=0}^{\infty} \gamma^t r_t$
- Trajectory probability: $p_\theta(\tau|s_0) = \prod_{t=0}^{\infty} \pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t, a_t)$

**Entropic Risk Measure:**
$$\rho_\tau(Z) = -\frac{1}{\tau} \log \mathbb{E}[\exp(-\tau Z)]$$

**Risk-Sensitive Objective:**
$$J_{\text{risk}}(\theta) = \mathbb{E}_{s_0 \sim d_0}\left[\rho_\tau(Z^{\pi_\theta}(s_0))\right]$$

where $Z^{\pi_\theta}(s_0)$ is the return distribution starting from $s_0$ under policy $\pi_\theta$.

---

## Derivation

### Step 1: Expand the Objective

The entropic risk of the return distribution is:
$$\rho_\tau(Z^{\pi_\theta}(s_0)) = -\frac{1}{\tau} \log \mathbb{E}_{\tau \sim \pi_\theta}\left[\exp\left(-\tau G(\tau)\right) \mid s_0\right]$$

So the objective becomes:
$$J_{\text{risk}}(\theta) = \mathbb{E}_{s_0}\left[-\frac{1}{\tau} \log \mathbb{E}_{\tau \sim \pi_\theta}\left[\exp(-\tau G(\tau)) \mid s_0\right]\right]$$

### Step 2: Define Auxiliary Function

Let:
$$F_\theta(s_0) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\exp(-\tau G(\tau)) \mid s_0\right]$$

Then:
$$J_{\text{risk}}(\theta) = \mathbb{E}_{s_0}\left[-\frac{1}{\tau} \log F_\theta(s_0)\right]$$

### Step 3: Apply Gradient Operator

$$\nabla_\theta J_{\text{risk}}(\theta) = \mathbb{E}_{s_0}\left[-\frac{1}{\tau} \cdot \frac{1}{F_\theta(s_0)} \cdot \nabla_\theta F_\theta(s_0)\right]$$

Using the chain rule for logarithm:
$$= \mathbb{E}_{s_0}\left[-\frac{1}{\tau} \cdot \frac{\nabla_\theta F_\theta(s_0)}{F_\theta(s_0)}\right]$$

### Step 4: Gradient of $F_\theta$

We need to compute:
$$\nabla_\theta F_\theta(s_0) = \nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}\left[\exp(-\tau G(\tau)) \mid s_0\right]$$

Using the **log-derivative trick** (also called likelihood ratio method):
$$\nabla_\theta \mathbb{E}_{\tau \sim \pi_\theta}[f(\tau)] = \mathbb{E}_{\tau \sim \pi_\theta}\left[f(\tau) \cdot \nabla_\theta \log p_\theta(\tau)\right]$$

Applying this:
$$\nabla_\theta F_\theta(s_0) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\exp(-\tau G(\tau)) \cdot \nabla_\theta \log p_\theta(\tau|s_0) \mid s_0\right]$$

### Step 5: Gradient of Log-Probability

The trajectory probability is:
$$p_\theta(\tau|s_0) = \prod_{t=0}^{\infty} \pi_\theta(a_t|s_t) \cdot P(s_{t+1}|s_t, a_t)$$

Taking the log:
$$\log p_\theta(\tau|s_0) = \sum_{t=0}^{\infty} \log \pi_\theta(a_t|s_t) + \sum_{t=0}^{\infty} \log P(s_{t+1}|s_t, a_t)$$

The gradient (dynamics don't depend on $\theta$):
$$\nabla_\theta \log p_\theta(\tau|s_0) = \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t)$$

### Step 6: Substitute Back

$$\nabla_\theta F_\theta(s_0) = \mathbb{E}_{\tau \sim \pi_\theta}\left[\exp(-\tau G(\tau)) \cdot \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t) \mid s_0\right]$$

### Step 7: Final Gradient Formula

Substituting into Step 3:
$$\nabla_\theta J_{\text{risk}}(\theta) = \mathbb{E}_{s_0}\left[-\frac{1}{\tau} \cdot \frac{\mathbb{E}_{\tau}\left[\exp(-\tau G(\tau)) \cdot \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)\right]}{F_\theta(s_0)}\right]$$

Combining expectations:
$$= \mathbb{E}_{s_0, \tau \sim \pi_\theta}\left[\frac{\exp(-\tau G(\tau))}{\mathbb{E}[\exp(-\tau G(\tau))]} \cdot \left(-\frac{1}{\tau}\right) \cdot \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t)\right]$$

---

## Final Result

$$\boxed{\nabla_\theta J_{\text{risk}}(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}\left[w(\tau) \cdot \sum_{t=0}^{\infty} \nabla_\theta \log \pi_\theta(a_t|s_t)\right]}$$

where the **importance weight** is:
$$w(\tau) = \frac{\exp(-\tau G(\tau))}{\mathbb{E}_{\tau'}[\exp(-\tau G(\tau'))]} \cdot \left(-\frac{1}{\tau}\right)$$

---

## Interpretation

### Standard Policy Gradient:
$$\nabla_\theta J(\theta) = \mathbb{E}_\tau\left[\sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

All trajectories weighted equally.

### Risk-Adjusted Policy Gradient:
$$\nabla_\theta J_{\text{risk}}(\theta) = \mathbb{E}_\tau\left[\underbrace{\frac{\exp(-\tau G(\tau))}{Z}}_{\text{importance weight}} \cdot \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot \left(-\frac{1}{\tau}\right)\right]$$

**Key properties:**
1. **For $\tau > 0$ (risk-averse):**
   - Trajectories with lower returns (negative $G$) get **higher weight**: $\exp(-\tau G) \uparrow$ when $G \downarrow$
   - Agent is pushed away from policies that produce bad outcomes

2. **For $\tau < 0$ (risk-seeking):**
   - Trajectories with higher returns get **higher weight**
   - Agent seeks policies with high upside potential

3. **For $\tau \to 0$ (risk-neutral):**
   - $\exp(-\tau G) \to 1$ for all $G$
   - Reduces to standard policy gradient

---

## Variance Analysis

The variance of this gradient estimator is:
$$\text{Var}\left[\nabla_\theta J_{\text{risk}}(\theta)\right] \propto \text{Var}\left[\exp(-\tau G(\tau))\right]$$

**Problem:** If returns $G$ have large range $[G_{\min}, G_{\max}]$:
$$\frac{\exp(-\tau G_{\min})}{\exp(-\tau G_{\max})} = \exp(\tau(G_{\max} - G_{\min}))$$

For $\tau = 1$ and range $= 200$:
$$\exp(200) \approx 10^{87}$$

**The importance weights span 87 orders of magnitude!** This causes extreme variance.

---

## Why Your Approach is Better

### Your Approach (Distributional RL + Risk-Adjusted Values):

1. **Learn distribution:** Train distributional critic to learn $Z(s)$ via distributional Bellman
2. **Extract risk measure:** Compute $V_\tau(s) = \rho_\tau(Z(s))$
3. **Use in advantages:** Compute GAE with $\delta_t = r_t + \gamma V_\tau(s_{t+1}) - V_\tau(s_t)$
4. **Standard policy gradient:** Use PPO with these advantages

**Benefits:**
- ✅ **Lower variance:** No exponential importance weighting
- ✅ **Scalable:** Distributional Bellman is stable
- ✅ **Practical:** Works with function approximation
- ✅ **Theoretically motivated:** Still uses the correct risk measure

**Trade-off:**
- ⚠️ Not the exact gradient of $\mathbb{E}[\rho_\tau(G)]$
- ⚠️ But it's a better approximation in practice!

---

## Connection to Literature

### Tamar et al. (2015): "Policy Gradient for Coherent Risk Measures"
- Derived exact gradient for CVaR
- Also suffers from high variance
- Proposed variance reduction techniques

### Chow et al. (2015): "Risk-Sensitive and Robust Decision-Making"
- Showed exponential utility policy gradients
- Acknowledged practical difficulties

### Prashanth & Ghavamzadeh (2013): "Actor-Critic Algorithms for Risk-Sensitive MDPs"
- Used critic to estimate risk-sensitive value functions
- Similar spirit to your approach!

---

## Conclusion

**Can we derive the risk-adjusted policy gradient?**
✅ **Yes!** We derived it rigorously above.

**Should we use it?**
❌ **No!** The exponential importance weighting has prohibitive variance.

**What should we use instead?**
✅ **Your approach!** Learn $V_\tau$ via distributional Bellman, use in GAE.

This is:
- More stable
- More scalable
- Still theoretically grounded
- Better in practice

The exact gradient is a beautiful theoretical result, but your practical approximation is superior for deep RL!
