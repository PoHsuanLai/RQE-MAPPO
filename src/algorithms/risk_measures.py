"""
Risk measures for RQE-MAPPO

Implements various convex risk measures from the paper:
- Entropic Risk
- CVaR (Conditional Value at Risk)
- Mean-Variance

Reference: Mazumdar et al. (2025), Section 2.2
"""

import torch
import torch.nn.functional as F
from typing import Literal


class RiskMeasure:
    """Base class for risk measures"""

    def __init__(self, tau: float = 1.0):
        """
        Args:
            tau: Risk aversion parameter
                 - tau → 0: Very risk-averse
                 - tau = 1: Moderate
                 - tau → ∞: Risk-neutral (expectation)
        """
        self.tau = tau

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        """
        Compute risk measure over distribution of values

        Args:
            values: Tensor of shape [n_samples] or [batch, n_samples]
                    Represents distribution of possible outcomes

        Returns:
            Risk-adjusted value (scalar or [batch])
        """
        raise NotImplementedError


class EntropicRisk(RiskMeasure):
    """
    Entropic risk measure: -(1/τ) log E[exp(-τ * value)]

    Properties:
    - Differentiable (good for gradient descent)
    - Closed form from samples
    - Connects to maximum entropy RL
    - Most commonly used in practice

    As tau → ∞: converges to expectation E[value]
    As tau → 0: converges to worst-case min(value)
    """

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        if self.tau == float('inf'):
            return values.mean(dim=-1)

        # For rewards (not costs), we want to be risk-averse to LOW values
        # So we use: -(1/τ) log E[exp(-τ * value)]
        # This penalizes low values more heavily

        # Numerical stability: subtract max before exp
        max_val = values.max(dim=-1, keepdim=True)[0]
        shifted_values = -self.tau * (values - max_val)

        log_mean_exp = torch.logsumexp(shifted_values, dim=-1) - torch.log(
            torch.tensor(values.shape[-1], dtype=values.dtype, device=values.device)
        )

        return -(1.0 / self.tau) * log_mean_exp - max_val.squeeze(-1)


class CVaR(RiskMeasure):
    """
    Conditional Value at Risk (CVaR): Average of worst α% outcomes

    Also known as Expected Shortfall

    Properties:
    - Very interpretable (used in finance regulations)
    - Non-smooth (uses sorting)
    - Requires more samples for stable estimation

    alpha = 1/tau:
    - alpha = 0.1 (tau = 10): average of worst 10% outcomes
    - alpha = 0.5 (tau = 2): average of worst 50% outcomes (median-like)
    """

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        # alpha = fraction of worst outcomes to average
        alpha = min(1.0, 1.0 / max(self.tau, 1.0))

        # Sort values (ascending for rewards)
        sorted_values, _ = torch.sort(values, dim=-1)

        # Take worst alpha fraction
        cutoff = max(1, int(alpha * values.shape[-1]))
        worst_values = sorted_values[..., :cutoff]

        return worst_values.mean(dim=-1)


class MeanVariance(RiskMeasure):
    """
    Mean-Variance risk measure: E[value] - (1/τ) * Var[value]

    Properties:
    - Simple and interpretable
    - Penalizes variance (uncertainty)
    - Differentiable

    Classical risk measure from portfolio theory
    """

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=-1)
        variance = values.var(dim=-1)

        return mean - (1.0 / self.tau) * variance


class WorstCase(RiskMeasure):
    """
    Worst-case risk measure: min(values)

    Extremely conservative (tau → 0)
    Useful for safety-critical applications
    """

    def __call__(self, values: torch.Tensor) -> torch.Tensor:
        return values.min(dim=-1)[0]


def get_risk_measure(
    risk_type: Literal["entropic", "cvar", "mean_variance", "worst_case"],
    tau: float = 1.0
) -> RiskMeasure:
    """
    Factory function to create risk measure

    Args:
        risk_type: Type of risk measure
        tau: Risk aversion parameter

    Returns:
        RiskMeasure instance
    """
    risk_measures = {
        "entropic": EntropicRisk,
        "cvar": CVaR,
        "mean_variance": MeanVariance,
        "worst_case": WorstCase,
    }

    if risk_type not in risk_measures:
        raise ValueError(
            f"Unknown risk measure: {risk_type}. "
            f"Choose from {list(risk_measures.keys())}"
        )

    return risk_measures[risk_type](tau=tau)


if __name__ == "__main__":
    # Test risk measures
    print("Testing risk measures...")

    # Create sample distribution (10 samples)
    torch.manual_seed(42)
    values = torch.randn(10) * 2 + 5  # Mean ~5, std ~2

    print(f"\nSample values: {values}")
    print(f"Mean: {values.mean():.3f}, Std: {values.std():.3f}")
    print(f"Min: {values.min():.3f}, Max: {values.max():.3f}")

    # Test different risk measures
    tau_values = [0.1, 0.5, 1.0, 5.0, float('inf')]

    for tau in tau_values:
        print(f"\n--- tau = {tau} ---")

        entropic = EntropicRisk(tau)(values)
        cvar = CVaR(tau)(values)
        mean_var = MeanVariance(tau)(values)

        print(f"Entropic:     {entropic:.3f}")
        print(f"CVaR:         {cvar:.3f}")
        print(f"Mean-Var:     {mean_var:.3f}")
        print(f"Expectation:  {values.mean():.3f}")

    # Test batch dimension
    print("\n--- Testing batch dimension ---")
    batch_values = torch.randn(4, 10) * 2 + 5  # [batch=4, samples=10]

    entropic_batch = EntropicRisk(tau=1.0)(batch_values)
    print(f"Input shape: {batch_values.shape}")
    print(f"Output shape: {entropic_batch.shape}")
    print(f"Batch results: {entropic_batch}")
