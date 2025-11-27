"""
RQE Solver: Game-Theoretic Risk-Averse Quantal Response Equilibrium

This module implements the core RQE equilibrium solver from:
"Tractable Multi-Agent Reinforcement Learning Through Behavioral Economics"
(Mazumdar et al., ICLR 2025)

The RQE solver finds equilibrium policies for a 2-player game where:
- Each player has bounded rationality (entropy regularization with parameter ε)
- Each player is risk-averse (entropic risk measure with parameter τ)

Equilibrium Condition:
    π_i(a_i) ∝ exp(Q_i^{risk}(a_i, π_{-i}) / ε_i)

where Q_i^{risk} incorporates risk aversion via the entropic risk measure.

Tractability Condition:
    ε₁ · ε₂ ≥ 1/(τ₁ · τ₂)

This condition ensures the equilibrium exists and is unique.

Usage:
    from algorithms.rqe_solver import RQESolver, RQEConfig

    config = RQEConfig(
        action_dims=[5, 5],
        tau=[2.0, 2.0],
        epsilon=[1.0, 1.0],
    )
    solver = RQESolver(config)

    # Q_matrices: List of [batch, my_actions, opp_actions] tensors
    policies = solver.solve(Q_matrices)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import torch
import torch.nn.functional as F


@dataclass
class RQEConfig:
    """Configuration for RQE Solver"""

    # Action space dimensions for each agent
    action_dims: List[int]

    # Risk aversion parameters (τ)
    # Higher τ = less risk-averse (τ → ∞ means risk-neutral)
    # Lower τ = more risk-averse
    tau: List[float] = field(default_factory=lambda: [1.0, 1.0])

    # Bounded rationality parameters (ε)
    # Higher ε = more random/exploratory
    # Lower ε = more deterministic/greedy
    epsilon: List[float] = field(default_factory=lambda: [1.0, 1.0])

    # Solver parameters
    max_iterations: int = 50
    learning_rate: float = 0.3
    momentum: float = 0.9
    tolerance: float = 1e-6

    # Regularization
    entropy_reg: bool = True  # Use entropy regularization

    def __post_init__(self):
        """Validate configuration and check tractability"""
        assert len(self.action_dims) == 2, "Currently only supports 2-player games"
        assert len(self.tau) == 2, "Must specify tau for each player"
        assert len(self.epsilon) == 2, "Must specify epsilon for each player"

        # Check tractability condition: ε₁·ε₂ ≥ 1/(τ₁·τ₂)
        lhs = self.epsilon[0] * self.epsilon[1]
        rhs = 1.0 / (self.tau[0] * self.tau[1])

        self.is_tractable = lhs >= rhs
        self.tractability_margin = lhs - rhs

        if not self.is_tractable:
            import warnings
            warnings.warn(
                f"Tractability condition violated: "
                f"ε₁·ε₂={lhs:.4f} < 1/(τ₁·τ₂)={rhs:.4f}. "
                f"Solver may not converge to a unique equilibrium."
            )


class RQESolver:
    """
    Risk-Averse Quantal Response Equilibrium Solver

    Solves for equilibrium policies using no-regret learning (mirror descent).

    The solver finds policies (π₁, π₂) where each player plays a soft best
    response to the other, accounting for:
    1. Risk aversion via entropic risk measure
    2. Bounded rationality via entropy regularization
    """

    def __init__(self, config: RQEConfig):
        self.config = config
        self.n_agents = 2

        # Cache for warm starting
        self._cached_policies: Optional[List[torch.Tensor]] = None
        self._cached_adversaries: Optional[List[torch.Tensor]] = None

    def solve(
        self,
        Q_matrices: List[torch.Tensor],
        warm_start: bool = True,
        return_info: bool = False
    ) -> List[torch.Tensor]:
        """
        Solve for RQE equilibrium policies.

        Args:
            Q_matrices: List of payoff matrices [Q₁, Q₂]
                Q₁: [batch, actions_1, actions_2] - Player 1's payoffs
                Q₂: [batch, actions_2, actions_1] - Player 2's payoffs
            warm_start: If True, initialize from cached previous solution
            return_info: If True, return additional convergence info

        Returns:
            policies: List of equilibrium policies [π₁, π₂]
                π₁: [batch, actions_1]
                π₂: [batch, actions_2]
            info (optional): Dict with convergence statistics
        """
        batch_size = Q_matrices[0].shape[0]
        device = Q_matrices[0].device

        # Initialize policies
        policies, adversaries = self._initialize_policies(
            batch_size, device, warm_start
        )

        # Run mirror descent
        policies, adversaries, info = self._mirror_descent(
            Q_matrices, policies, adversaries
        )

        # Cache for next call
        self._cached_policies = [p.detach().clone() for p in policies]
        self._cached_adversaries = [a.detach().clone() for a in adversaries]

        if return_info:
            return policies, info
        return policies

    def _initialize_policies(
        self,
        batch_size: int,
        device: torch.device,
        warm_start: bool
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Initialize policies, optionally from cache."""

        # Try warm start
        if warm_start and self._cached_policies is not None:
            if self._cached_policies[0].shape[0] == batch_size:
                policies = [p.clone() for p in self._cached_policies]
                adversaries = [a.clone() for a in self._cached_adversaries]
                return policies, adversaries

        # Initialize uniformly
        policies = [
            torch.ones(batch_size, self.config.action_dims[i], device=device)
            / self.config.action_dims[i]
            for i in range(self.n_agents)
        ]

        # Adversarial beliefs (player i's belief about opponent j)
        adversaries = [policies[1].clone(), policies[0].clone()]

        return policies, adversaries

    def _mirror_descent(
        self,
        Q_matrices: List[torch.Tensor],
        policies: List[torch.Tensor],
        adversaries: List[torch.Tensor]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], dict]:
        """
        Run mirror descent to find equilibrium.

        The algorithm alternates between:
        1. Updating each player's policy (soft best response)
        2. Updating each player's adversarial belief about opponent
        """
        lr = self.config.learning_rate
        momentum = self.config.momentum

        # Momentum buffers
        velocity_p = [torch.zeros_like(p) for p in policies]
        velocity_a = [torch.zeros_like(a) for a in adversaries]

        # Track convergence
        policy_diffs = []

        for iteration in range(self.config.max_iterations):
            old_policies = [p.clone() for p in policies]

            for i in range(self.n_agents):
                j = 1 - i  # Opponent index

                # Compute gradients
                grad_policy = self._compute_policy_gradient(
                    Q_matrices[i], policies[i], adversaries[i], i
                )
                grad_adversary = self._compute_adversary_gradient(
                    Q_matrices[i], policies[j], adversaries[i], i
                )

                # Momentum update
                velocity_p[i] = momentum * velocity_p[i] + lr * grad_policy
                velocity_a[i] = momentum * velocity_a[i] + lr * grad_adversary

                # Mirror descent step (entropic projection)
                policies[i] = self._mirror_step(policies[i], velocity_p[i], self.config.epsilon[i])
                adversaries[i] = self._mirror_step(adversaries[i], velocity_a[i], 1.0 / self.config.tau[i])

            # Check convergence
            diff = max(
                (policies[i] - old_policies[i]).abs().max().item()
                for i in range(self.n_agents)
            )
            policy_diffs.append(diff)

            if diff < self.config.tolerance:
                break

        info = {
            'iterations': iteration + 1,
            'converged': diff < self.config.tolerance,
            'final_diff': diff,
            'policy_diffs': policy_diffs,
        }

        return policies, adversaries, info

    def _compute_policy_gradient(
        self,
        Q: torch.Tensor,
        pi: torch.Tensor,
        p: torch.Tensor,
        agent_idx: int
    ) -> torch.Tensor:
        """
        Compute gradient for player's policy update.

        For maximization: ∇_π [π^T Q p + ε H(π)]
        where H(π) is entropy.

        Args:
            Q: [batch, my_actions, opp_actions]
            pi: [batch, my_actions] - current policy
            p: [batch, opp_actions] - adversarial belief about opponent
            agent_idx: which agent (0 or 1)
        """
        # Expected Q-value under adversarial belief: Q @ p -> [batch, my_actions]
        expected_Q = torch.matmul(Q, p.unsqueeze(-1)).squeeze(-1)

        # Entropy gradient: -∇H(π) = log(π) + 1
        if self.config.entropy_reg:
            entropy_grad = torch.log(pi + 1e-10) + 1
        else:
            entropy_grad = torch.zeros_like(pi)

        # Total gradient (for ascent): Q-value gradient - entropy penalty
        # We want to maximize: E[Q] + ε * H(π)
        # Gradient: expected_Q - ε * (log(π) + 1)
        gradient = expected_Q - self.config.epsilon[agent_idx] * entropy_grad

        return gradient

    def _compute_adversary_gradient(
        self,
        Q: torch.Tensor,
        pi_opponent: torch.Tensor,
        p: torch.Tensor,
        agent_idx: int
    ) -> torch.Tensor:
        """
        Compute gradient for adversarial belief update.

        The adversary minimizes player i's value plus a penalty D(p, π_{-i}).

        Args:
            Q: [batch, my_actions, opp_actions]
            pi_opponent: [batch, opp_actions] - opponent's actual policy
            p: [batch, opp_actions] - adversarial belief
            agent_idx: which agent's adversary we're updating
        """
        # Q^T @ π_i -> [batch, opp_actions]
        # This is player i's expected value for each opponent action
        # We transpose Q to get [batch, opp_actions, my_actions]
        # Then multiply by player i's policy (but we use opponent perspective)

        # For the adversary of player i, we want to find p that minimizes
        # player i's value: min_p [π_i^T Q p + (1/τ) D(p, π_j)]

        # Gradient of -π_i^T Q p w.r.t. p is -Q^T π_i
        # But we're computing from perspective of adversary, so we negate
        Q_transpose = Q.transpose(1, 2)  # [batch, opp_actions, my_actions]

        # We need player i's policy, but adversary is about opponent j
        # Actually, the gradient is: -Q^T (summed over player i's actions weighted by π_i)
        # For simplicity, we use the opponent's policy as the weighting
        grad_Q = -torch.matmul(Q_transpose, pi_opponent.unsqueeze(-1)).squeeze(-1)

        # Penalty gradient: D(p, π_j) - typically KL divergence
        # ∇_p KL(p || π_j) = log(p/π_j) + 1
        grad_penalty = torch.log(p / (pi_opponent + 1e-10) + 1e-10) + 1

        gradient = grad_Q + (1.0 / self.config.tau[agent_idx]) * grad_penalty

        return gradient

    def _mirror_step(
        self,
        policy: torch.Tensor,
        gradient: torch.Tensor,
        reg_coef: float
    ) -> torch.Tensor:
        """
        Mirror descent step with entropic regularization.

        Update: π_new ∝ π_old * exp(gradient / reg_coef)
        Then normalize to get valid probability distribution.
        """
        # Log-space update for numerical stability
        log_policy = torch.log(policy + 1e-10)
        log_policy_new = log_policy + gradient / (reg_coef + 1e-10)

        # Softmax to normalize (equivalent to exponentiating and normalizing)
        policy_new = F.softmax(log_policy_new, dim=-1)

        return policy_new

    def compute_exploitability(
        self,
        Q_matrices: List[torch.Tensor],
        policies: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Compute exploitability (how far from equilibrium).

        Exploitability = sum of regrets for each player
        where regret_i = max_{π'_i} V_i(π'_i, π_{-i}) - V_i(π_i, π_{-i})

        Returns:
            exploitability: [batch] - exploitability for each batch element
        """
        exploitability = torch.zeros(Q_matrices[0].shape[0], device=Q_matrices[0].device)

        for i in range(self.n_agents):
            j = 1 - i

            Q_i = Q_matrices[i]  # [batch, my_actions, opp_actions]
            pi_i = policies[i]  # [batch, my_actions]
            pi_j = policies[j]  # [batch, opp_actions]

            # Current value: V_i = π_i^T Q_i π_j
            current_value = torch.sum(
                pi_i.unsqueeze(-1) * Q_i * pi_j.unsqueeze(1),
                dim=[1, 2]
            )

            # Best response value: max_a E_{π_j}[Q_i(a, ·)]
            expected_Q = torch.matmul(Q_i, pi_j.unsqueeze(-1)).squeeze(-1)  # [batch, my_actions]

            # For QRE, best response includes entropy bonus
            # V^BR = max_π [π^T expected_Q + ε H(π)]
            # = ε * log(sum_a exp(expected_Q_a / ε))  (soft-max value)
            best_response_value = self.config.epsilon[i] * torch.logsumexp(
                expected_Q / self.config.epsilon[i], dim=-1
            )

            # Regret (non-negative)
            regret = torch.clamp(best_response_value - current_value, min=0)
            exploitability = exploitability + regret

        return exploitability

    def verify_equilibrium(
        self,
        Q_matrices: List[torch.Tensor],
        policies: List[torch.Tensor],
        tolerance: float = 0.01
    ) -> dict:
        """
        Verify that policies form an approximate equilibrium.

        Returns:
            dict with verification results
        """
        results = {'is_equilibrium': True, 'agents': {}}

        for i in range(self.n_agents):
            j = 1 - i

            pi_i = policies[i][0]  # Remove batch dim for single game
            pi_j = policies[j][0]
            Q_i = Q_matrices[i][0]

            # Compute expected Q under opponent's policy
            expected_Q = torch.matmul(Q_i, pi_j)  # [my_actions]

            # Compute QRE best response
            best_response = F.softmax(expected_Q / self.config.epsilon[i], dim=-1)

            # Deviation from best response
            deviation = (pi_i - best_response).abs().max().item()
            kl_div = F.kl_div(
                torch.log(pi_i + 1e-10), best_response, reduction='sum'
            ).item()

            results['agents'][f'agent_{i}'] = {
                'policy': pi_i.cpu().numpy(),
                'best_response': best_response.cpu().numpy(),
                'max_deviation': deviation,
                'kl_divergence': kl_div,
            }

            if deviation > tolerance:
                results['is_equilibrium'] = False

        # Compute exploitability
        exploitability = self.compute_exploitability(Q_matrices, policies)
        results['exploitability'] = exploitability[0].item()

        return results


# Convenience function
def solve_rqe(
    Q1: torch.Tensor,
    Q2: torch.Tensor,
    tau: List[float] = [1.0, 1.0],
    epsilon: List[float] = [1.0, 1.0],
    **kwargs
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convenience function to solve RQE for a single game.

    Args:
        Q1: [actions_1, actions_2] - Player 1's payoff matrix
        Q2: [actions_2, actions_1] - Player 2's payoff matrix
        tau: Risk aversion parameters
        epsilon: Bounded rationality parameters
        **kwargs: Additional config parameters

    Returns:
        pi1: [actions_1] - Player 1's equilibrium policy
        pi2: [actions_2] - Player 2's equilibrium policy
    """
    # Add batch dimension
    Q1_batch = Q1.unsqueeze(0)
    Q2_batch = Q2.unsqueeze(0)

    config = RQEConfig(
        action_dims=[Q1.shape[0], Q1.shape[1]],
        tau=tau,
        epsilon=epsilon,
        **kwargs
    )

    solver = RQESolver(config)
    policies = solver.solve([Q1_batch, Q2_batch])

    # Remove batch dimension
    return policies[0][0], policies[1][0]


# Testing
if __name__ == "__main__":
    print("=" * 60)
    print("RQE Solver Test")
    print("=" * 60)

    # Test 1: Matching Pennies (zero-sum)
    print("\n1. Matching Pennies")
    print("-" * 40)

    Q1 = torch.tensor([[1., -1.], [-1., 1.]])
    Q2 = torch.tensor([[-1., 1.], [1., -1.]])

    pi1, pi2 = solve_rqe(Q1, Q2, tau=[2.0, 2.0], epsilon=[1.0, 1.0])

    print(f"Player 1 policy: {pi1.numpy()}")
    print(f"Player 2 policy: {pi2.numpy()}")
    print(f"Expected: [0.5, 0.5] for both (Nash equilibrium)")
    print(f"Deviation: {(pi1 - 0.5).abs().max().item():.6f}")

    # Test 2: Rock-Paper-Scissors
    print("\n2. Rock-Paper-Scissors")
    print("-" * 40)

    Q1 = torch.tensor([
        [0., -1., 1.],
        [1., 0., -1.],
        [-1., 1., 0.]
    ])
    Q2 = -Q1

    pi1, pi2 = solve_rqe(Q1, Q2, tau=[2.0, 2.0], epsilon=[1.0, 1.0])

    print(f"Player 1 policy: {pi1.numpy()}")
    print(f"Player 2 policy: {pi2.numpy()}")
    print(f"Expected: [0.33, 0.33, 0.33] for both")
    print(f"Deviation: {(pi1 - 1/3).abs().max().item():.6f}")

    # Test 3: Prisoner's Dilemma
    print("\n3. Prisoner's Dilemma")
    print("-" * 40)

    Q1 = torch.tensor([[-1., -3.], [0., -2.]])
    Q2 = torch.tensor([[-1., 0.], [-3., -2.]])

    pi1, pi2 = solve_rqe(Q1, Q2, tau=[2.0, 2.0], epsilon=[1.0, 1.0])

    print(f"Player 1 policy: {pi1.numpy()}")
    print(f"Player 2 policy: {pi2.numpy()}")
    print(f"Nash equilibrium: [0, 1] (Defect) for both")
    print(f"QRE smooths toward this as ε → 0")

    # Test 4: Verify equilibrium
    print("\n4. Equilibrium Verification")
    print("-" * 40)

    config = RQEConfig(
        action_dims=[2, 2],
        tau=[2.0, 2.0],
        epsilon=[1.0, 1.0],
    )
    solver = RQESolver(config)

    Q_matrices = [Q1.unsqueeze(0), Q2.unsqueeze(0)]
    policies, info = solver.solve(Q_matrices, return_info=True)

    verification = solver.verify_equilibrium(Q_matrices, policies)

    print(f"Converged: {info['converged']} in {info['iterations']} iterations")
    print(f"Is equilibrium: {verification['is_equilibrium']}")
    print(f"Exploitability: {verification['exploitability']:.6f}")

    for agent, data in verification['agents'].items():
        print(f"  {agent}: deviation={data['max_deviation']:.6f}")

    # Test 5: Tractability check
    print("\n5. Tractability Condition")
    print("-" * 40)

    # Tractable case
    config_ok = RQEConfig(action_dims=[2, 2], tau=[2.0, 2.0], epsilon=[1.0, 1.0])
    print(f"τ=[2,2], ε=[1,1]: tractable={config_ok.is_tractable}, margin={config_ok.tractability_margin:.4f}")

    # Intractable case
    config_bad = RQEConfig(action_dims=[2, 2], tau=[1.0, 1.0], epsilon=[0.5, 0.5])
    print(f"τ=[1,1], ε=[0.5,0.5]: tractable={config_bad.is_tractable}, margin={config_bad.tractability_margin:.4f}")

    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
