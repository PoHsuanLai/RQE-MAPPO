"""
Markov RQE Solver: Risk-Averse Quantal Response Equilibrium for Finite-Horizon Markov Games

Implements Algorithm 1 from:
"Tractable Multi-Agent Reinforcement Learning Through Behavioral Economics"
(Mazumdar et al., ICLR 2025)

This solver computes Markov RQE for finite-horizon general-sum Markov games where:
- Players have bounded rationality (entropy regularization with parameter ε)
- Players are risk-averse (convex risk measure with parameter τ)

The algorithm uses backward induction:
1. Start from terminal time h = H
2. At each timestep h, construct payoff matrices that incorporate:
   - Immediate payoffs R_i(s, a)
   - Risk-adjusted future values via penalty function D_env
3. Solve RQE at each state using the matrix game solver
4. Propagate values backward to earlier timesteps

Notation (Game Theory):
- n: number of players
- S: state space
- A_i: action space for player i
- A = A_1 × A_2 × ... × A_n: joint action space
- R_i(s, a): immediate payoff for player i at state s with joint action a
- P(s' | s, a): transition probability
- π_i(a_i | s): player i's strategy at state s
- V_i(s): player i's value at state s
- Q_i(s, a): player i's payoff matrix at state s

Tractability Condition (for 2 players):
    ε_1 · ε_2 ≥ 1/(τ_1 · τ_2)
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
import torch
import torch.nn.functional as F
from enum import Enum
from tqdm import tqdm


class PenaltyType(Enum):
    """Penalty functions for risk measures (D in the paper)"""
    KL_DIVERGENCE = "kl"           # D(p, q) = KL(p || q)
    REVERSE_KL = "reverse_kl"      # D(p, q) = KL(q || p)
    CHI_SQUARED = "chi_squared"    # D(p, q) = χ²(p || q)
    TOTAL_VARIATION = "tv"         # D(p, q) = ||p - q||_1


class RegularizerType(Enum):
    """Regularizers for bounded rationality (ν in the paper)"""
    NEGATIVE_ENTROPY = "entropy"   # ν(π) = Σ π log π (negative entropy)
    LOG_BARRIER = "log_barrier"    # ν(π) = -Σ log π


@dataclass
class MarkovGameConfig:
    """
    Configuration for a finite-horizon Markov game.

    Game Theory Notation:
    - n_players: number of players
    - n_states: |S| - size of state space
    - action_dims: |A_i| for each player
    - horizon: H - number of timesteps

    From the paper (Mazumdar et al., ICLR 2025):
    - τ_i: risk-aversion parameter for player i (single parameter per agent)
    - ε_i: bounded rationality parameter for player i

    Tractability condition (Theorem 3): ε_1·ε_2 ≥ 1/(τ_1·τ_2)
    """
    n_players: int
    n_states: int
    action_dims: List[int]  # |A_i| for each player
    horizon: int

    # Risk aversion parameters (τ_i) - single parameter per agent matching paper
    # Higher τ = MORE risk-averse (opposite of some conventions)
    # τ controls both policy risk and environment risk
    tau: List[float] = None

    # Bounded rationality parameters (ε_i)
    # Higher ε = more random/exploratory
    epsilon: List[float] = None

    # Penalty function types
    penalty_policy: PenaltyType = PenaltyType.KL_DIVERGENCE
    penalty_env: PenaltyType = PenaltyType.KL_DIVERGENCE

    # Regularizer type
    regularizer: RegularizerType = RegularizerType.NEGATIVE_ENTROPY

    # Matrix game solver parameters
    solver_iterations: int = 50
    solver_lr: float = 0.3
    solver_tolerance: float = 1e-6

    def __post_init__(self):
        if self.tau is None:
            self.tau = [1.0] * self.n_players
        if self.epsilon is None:
            self.epsilon = [1.0] * self.n_players

        # Check tractability for 2-player case (Theorem 3 from paper)
        # Condition: ε_1·ε_2 ≥ 1/(τ_1·τ_2)
        if self.n_players == 2:
            # ξ* = 1/τ for KL divergence penalty
            xi_1 = 1.0 / self.tau[0]
            xi_2 = 1.0 / self.tau[1]

            lhs = self.epsilon[0] * self.epsilon[1]
            rhs = xi_1 * xi_2

            self.is_tractable = lhs >= rhs
            self.tractability_margin = lhs - rhs

            if not self.is_tractable:
                import warnings
                warnings.warn(
                    f"Tractability condition violated: "
                    f"ε_1·ε_2 = {lhs:.4f} < 1/(τ_1·τ_2) = {rhs:.4f}. "
                    f"Solver may not converge to unique equilibrium."
                )


@dataclass
class MarkovGame:
    """
    Finite-horizon Markov game specification.

    Notation:
    - payoffs[i][h]: R_i,h(s, a) - player i's payoff at time h
      Shape: [n_states, A_1, A_2, ..., A_n]
    - transitions[h]: P_h(s' | s, a) - transition probabilities at time h
      Shape: [n_states, A_1, A_2, ..., A_n, n_states]
    - action_masks[i]: mask for valid actions per state for player i
      Shape: [n_states, A_i], 1 = valid, 0 = invalid
    """
    config: MarkovGameConfig
    payoffs: List[List[torch.Tensor]]    # payoffs[player][timestep]
    transitions: List[torch.Tensor]       # transitions[timestep]
    action_masks: List[torch.Tensor] = None  # action_masks[player], shape [n_states, A_i]

    def __post_init__(self):
        """Validate game specification"""
        H = self.config.horizon
        S = self.config.n_states
        n = self.config.n_players

        # Validate payoffs
        assert len(self.payoffs) == n, f"Expected {n} players, got {len(self.payoffs)}"
        for i in range(n):
            assert len(self.payoffs[i]) == H, f"Player {i}: expected {H} timesteps"

        # Validate transitions
        assert len(self.transitions) == H, f"Expected {H} transition matrices"


class MarkovRQESolver:
    """
    Solver for Risk-Averse Quantal Response Equilibrium in Markov Games.

    Implements Algorithm 1 from the paper using backward induction.
    """

    def __init__(self, config: MarkovGameConfig):
        self.config = config
        self.n_players = config.n_players
        self.n_states = config.n_states
        self.H = config.horizon

    def solve(
        self,
        game: MarkovGame,
        device: torch.device = None
    ) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """
        Compute Markov RQE using backward induction.

        Args:
            game: MarkovGame specification
            device: torch device

        Returns:
            policies: policies[player][timestep] with shape [n_states, |A_i|]
            values: values[player][timestep] with shape [n_states]
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        H = self.H
        S = self.n_states
        n = self.n_players

        # Initialize value functions: V_i,H+1 = 0 (terminal condition)
        # values[i][h] is V_i,h(s) for all states s
        values = [[None for _ in range(H + 1)] for _ in range(n)]
        for i in range(n):
            values[i][H] = torch.zeros(S, device=device)

        # Initialize policies: π[i][h] is π_i,h(a | s)
        policies = [[None for _ in range(H)] for _ in range(n)]

        # Backward induction: h = H-1, H-2, ..., 0
        for h in tqdm(range(H - 1, -1, -1), desc="Backward induction", leave=False):
            # Construct payoff matrices Q_i,h(s, a) for all states
            Q_matrices = self._construct_payoff_matrices(
                game, values, h, device
            )

            # Solve RQE at each state
            state_policies, state_values = self._solve_stage_games(
                Q_matrices, device, game.action_masks
            )

            # Store results
            for i in range(n):
                policies[i][h] = state_policies[i]
                values[i][h] = state_values[i]

        return policies, values

    def _construct_payoff_matrices(
        self,
        game: MarkovGame,
        values: List[List[torch.Tensor]],
        h: int,
        device: torch.device
    ) -> List[torch.Tensor]:
        """
        Construct payoff matrices Q_i,h(s, a) incorporating risk over transitions.

        From Equation (13) in the paper:
        Q̂_i,h(s, a) = R_i,h(s, a) + inf_{P̃ ∈ Δ_S} [P̃ · V̂_i,h+1 + D_env(P̃, P_h(·|s,a))]

        For KL divergence penalty, this becomes the soft-minimum:
        Q̂_i,h(s, a) = R_i,h(s, a) - τ_env · log E_{s'~P}[exp(-V_i,h+1(s')/τ_env)]

        Args:
            game: MarkovGame
            values: Current value estimates
            h: Current timestep
            device: torch device

        Returns:
            Q_matrices[i]: shape [n_states, A_1, A_2, ..., A_n] for player i
        """
        n = self.n_players
        S = self.n_states

        Q_matrices = []

        for i in range(n):
            # Get immediate payoffs: R_i,h(s, a)
            R_ih = game.payoffs[i][h].to(device)  # [S, A_1, ..., A_n]

            # Get transition probabilities: P_h(s' | s, a)
            P_h = game.transitions[h].to(device)  # [S, A_1, ..., A_n, S]

            # Get future values: V_i,h+1(s')
            V_next = values[i][h + 1].to(device)  # [S]

            # Compute risk-adjusted future value
            # Use single tau parameter per agent (matching paper)
            tau_i = self.config.tau[i]

            if self.config.penalty_env == PenaltyType.KL_DIVERGENCE:
                # For KL penalty: use soft-minimum (entropic risk measure)
                # ρ_τ(V) = -τ · log E[exp(-V/τ)]
                #
                # Properties:
                # - Always ≤ E[V] (pessimistic/risk-averse)
                # - As τ → ∞: approaches E[V] (risk-neutral)
                # - As τ → 0: approaches min(V) (worst-case)
                # - Higher τ = LESS risk-averse (counterintuitive but standard)

                # Numerically stable computation using log-sum-exp trick
                # ρ_τ(V) = -τ · log E[exp(-V/τ)]
                #        = -τ · log (sum_s' P(s') exp(-V(s')/τ))
                #
                # Let z = -V/τ, then we compute -τ · log(P @ exp(z))
                # Use log-sum-exp: log(sum(w_i * exp(z_i))) = max(z) + log(sum(w_i * exp(z_i - max(z))))
                z = -V_next / tau_i  # [S]
                z_max = z.max()
                exp_z_stable = torch.exp(z - z_max)  # [S], all values <= 1

                # Weighted sum: sum_s' P(s'|s,a) * exp(z(s') - z_max)
                weighted_exp = torch.tensordot(P_h, exp_z_stable, dims=([[-1], [0]]))  # [S, A1, A2]

                # Final result: -τ * (z_max + log(weighted_exp))
                continuation = -tau_i * (z_max + torch.log(weighted_exp + 1e-10))

            elif self.config.penalty_env == PenaltyType.TOTAL_VARIATION:
                # For TV penalty: robust MDP formulation
                # Take worst-case over uncertainty set
                expected_V = torch.tensordot(P_h, V_next, dims=([[-1], [0]]))
                # Simple approximation: subtract uncertainty penalty
                continuation = expected_V - tau_i * torch.ones_like(expected_V)

            else:
                # Default: risk-neutral (expected value)
                continuation = torch.tensordot(P_h, V_next, dims=([[-1], [0]]))

            # Total payoff matrix
            Q_i = R_ih + continuation
            Q_matrices.append(Q_i)

        return Q_matrices

    def _solve_stage_games(
        self,
        Q_matrices: List[torch.Tensor],
        device: torch.device,
        action_masks: List[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Solve RQE for each state's stage game (VECTORIZED over all states).

        At each state s, we have a matrix game with payoffs Q_i(s, ·).
        We solve for the RQE policies and compute the equilibrium values.

        This version batches ALL states together for GPU parallelization.

        Args:
            Q_matrices[i]: [n_states, A_1, ..., A_n] payoff matrices
            device: torch device
            action_masks[i]: [n_states, A_i] action masks (1=valid, 0=invalid)

        Returns:
            policies[i]: [n_states, A_i] equilibrium strategies
            values[i]: [n_states] equilibrium values
        """
        n = self.n_players

        # Currently only support 2-player games
        assert n == 2, "Markov RQE solver currently only supports 2-player games"

        # Q_matrices[0] has shape [S, A1, A2]
        # Q_matrices[1] has shape [S, A1, A2]
        Q1 = Q_matrices[0]  # [S, A1, A2]
        Q2 = Q_matrices[1]  # [S, A1, A2]

        # For player 2's perspective, we need Q2 transposed: [S, A2, A1]
        Q2_transposed = Q2.transpose(1, 2)  # [S, A2, A1]

        # Get action masks if provided
        mask1 = action_masks[0].to(device) if action_masks is not None else None
        mask2 = action_masks[1].to(device) if action_masks is not None else None

        # Solve RQE for ALL states in parallel (batch_size = S)
        pi1, pi2 = self._solve_matrix_rqe(Q1, Q2_transposed, device, mask1, mask2)

        # pi1: [S, A1], pi2: [S, A2]
        policies = [pi1, pi2]

        # Compute equilibrium values for all states: V_i = π_i^T Q_i π_j
        # V_1[s] = sum_{a1,a2} π_1[s,a1] * Q_1[s,a1,a2] * π_2[s,a2]
        # Using einsum for clarity: V = einsum('sa,sab,sb->s', pi1, Q1, pi2)
        values_1 = torch.einsum('sa,sab,sb->s', pi1, Q1, pi2)
        values_2 = torch.einsum('sa,sab,sb->s', pi1, Q2, pi2)

        values = [values_1, values_2]

        return policies, values

    def _solve_matrix_rqe(
        self,
        Q1: torch.Tensor,
        Q2: torch.Tensor,
        device: torch.device,
        mask1: torch.Tensor = None,
        mask2: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Solve RQE for a 2-player matrix game using mirror descent.

        From the paper, we solve the 4-player game:
        - Player 1 minimizes: -π_1^T Q_1 p_1 + ε_1 ν(π_1)
        - Player 2 minimizes: -π_2^T Q_2^T p_2 + ε_2 ν(π_2)
        - Adversary 1 minimizes: π_1^T Q_1 p_1 + (1/τ_1) D(p_1, π_2)
        - Adversary 2 minimizes: π_2^T Q_2^T p_2 + (1/τ_2) D(p_2, π_1)

        Args:
            Q1: [batch, A_1, A_2] - Player 1's payoff matrix
            Q2: [batch, A_2, A_1] - Player 2's payoff matrix
            device: torch device
            mask1: [batch, A_1] - Action mask for player 1 (1=valid, 0=invalid)
            mask2: [batch, A_2] - Action mask for player 2 (1=valid, 0=invalid)

        Returns:
            pi1: [batch, A_1] - Player 1's equilibrium strategy
            pi2: [batch, A_2] - Player 2's equilibrium strategy
        """
        batch_size = Q1.shape[0]
        A1, A2 = Q1.shape[1], Q1.shape[2]

        eps1, eps2 = self.config.epsilon[0], self.config.epsilon[1]
        tau1, tau2 = self.config.tau[0], self.config.tau[1]

        # Sanitize Q matrices - replace nan/inf with zeros
        Q1 = torch.nan_to_num(Q1, nan=0.0, posinf=100.0, neginf=-100.0)
        Q2 = torch.nan_to_num(Q2, nan=0.0, posinf=100.0, neginf=-100.0)

        # Initialize strategies uniformly over valid actions
        if mask1 is not None:
            n_valid1 = mask1.sum(dim=-1, keepdim=True).clamp(min=1)
            pi1 = mask1.float() / n_valid1
        else:
            pi1 = torch.ones(batch_size, A1, device=device) / A1

        if mask2 is not None:
            n_valid2 = mask2.sum(dim=-1, keepdim=True).clamp(min=1)
            pi2 = mask2.float() / n_valid2
        else:
            pi2 = torch.ones(batch_size, A2, device=device) / A2

        # Adversarial beliefs
        p1 = pi2.clone()  # Adversary 1's belief about player 2
        p2 = pi1.clone()  # Adversary 2's belief about player 1

        for _ in range(self.config.solver_iterations):
            old_pi1, old_pi2 = pi1.clone(), pi2.clone()

            # === Update Player 1 ===
            # Gradient: Q_1 @ p_1 (expected payoff under adversary's belief)
            grad_pi1 = torch.bmm(Q1, p1.unsqueeze(-1)).squeeze(-1)  # [batch, A1]
            grad_pi1 = torch.clamp(grad_pi1, -50.0, 50.0)
            pi1 = self._mirror_step(pi1, grad_pi1, eps1, maximize=True, action_mask=mask1)

            # === Update Player 2 ===
            # Gradient: Q_2 @ p_2
            grad_pi2 = torch.bmm(Q2, p2.unsqueeze(-1)).squeeze(-1)  # [batch, A2]
            grad_pi2 = torch.clamp(grad_pi2, -50.0, 50.0)
            pi2 = self._mirror_step(pi2, grad_pi2, eps2, maximize=True, action_mask=mask2)

            # === Update Adversary 1 (belief about player 2) ===
            # Gradient: -Q_1^T @ π_1 + (1/τ_1) ∇D(p_1, π_2)
            grad_p1_payoff = -torch.bmm(Q1.transpose(1, 2), pi1.unsqueeze(-1)).squeeze(-1)
            grad_p1_penalty = self._penalty_gradient(p1, pi2, tau1)
            grad_p1 = grad_p1_payoff + grad_p1_penalty
            grad_p1 = torch.clamp(grad_p1, -50.0, 50.0)
            p1 = self._mirror_step(p1, grad_p1, 1.0 / tau1, maximize=False, action_mask=mask2)

            # === Update Adversary 2 (belief about player 1) ===
            grad_p2_payoff = -torch.bmm(Q2.transpose(1, 2), pi2.unsqueeze(-1)).squeeze(-1)
            grad_p2_penalty = self._penalty_gradient(p2, pi1, tau2)
            grad_p2 = grad_p2_payoff + grad_p2_penalty
            grad_p2 = torch.clamp(grad_p2, -50.0, 50.0)
            p2 = self._mirror_step(p2, grad_p2, 1.0 / tau2, maximize=False, action_mask=mask1)

            # Safety check: ensure valid probability distributions
            pi1 = torch.nan_to_num(pi1, nan=1.0/A1)
            pi2 = torch.nan_to_num(pi2, nan=1.0/A2)
            p1 = torch.nan_to_num(p1, nan=1.0/A2)
            p2 = torch.nan_to_num(p2, nan=1.0/A1)

            # Re-normalize in case of numerical issues
            pi1 = pi1 / (pi1.sum(dim=-1, keepdim=True) + 1e-10)
            pi2 = pi2 / (pi2.sum(dim=-1, keepdim=True) + 1e-10)
            p1 = p1 / (p1.sum(dim=-1, keepdim=True) + 1e-10)
            p2 = p2 / (p2.sum(dim=-1, keepdim=True) + 1e-10)

            # Check convergence
            diff = max(
                (pi1 - old_pi1).abs().max().item(),
                (pi2 - old_pi2).abs().max().item()
            )
            if diff < self.config.solver_tolerance:
                break

        return pi1, pi2

    def _mirror_step(
        self,
        policy: torch.Tensor,
        gradient: torch.Tensor,
        reg_coef: float,
        maximize: bool = True,
        action_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Mirror descent step with entropy regularization.

        For maximization: π_new ∝ π_old · exp(gradient / reg_coef)
        For minimization: π_new ∝ π_old · exp(-gradient / reg_coef)

        Uses log-domain computation for numerical stability.

        Args:
            policy: Current policy [batch, A]
            gradient: Gradient [batch, A]
            reg_coef: Regularization coefficient (epsilon)
            maximize: Whether to maximize (True) or minimize (False)
            action_mask: Optional mask [batch, A], 1 = valid, 0 = invalid
        """
        sign = 1.0 if maximize else -1.0
        log_policy = torch.log(policy + 1e-10)

        # Clip the gradient update to prevent overflow
        update = sign * gradient / (reg_coef + 1e-10)
        update = torch.clamp(update, -50.0, 50.0)  # Prevent exp overflow

        log_policy_new = log_policy + update

        # Apply action mask: set invalid actions to -inf before softmax
        if action_mask is not None:
            log_policy_new = log_policy_new.masked_fill(action_mask == 0, float('-inf'))

        return F.softmax(log_policy_new, dim=-1)

    def _penalty_gradient(
        self,
        p: torch.Tensor,
        pi_ref: torch.Tensor,
        tau: float
    ) -> torch.Tensor:
        """
        Compute gradient of penalty D(p, π_ref) w.r.t. p.

        For KL divergence D(p || π) = Σ p log(p/π):
        ∇_p D = log(p/π) + 1
        """
        if self.config.penalty_policy == PenaltyType.KL_DIVERGENCE:
            return (1.0 / tau) * (torch.log(p / (pi_ref + 1e-10) + 1e-10) + 1)
        elif self.config.penalty_policy == PenaltyType.REVERSE_KL:
            return (1.0 / tau) * (-pi_ref / (p + 1e-10))
        else:
            # Default: KL
            return (1.0 / tau) * (torch.log(p / (pi_ref + 1e-10) + 1e-10) + 1)


def create_random_markov_game(
    n_players: int = 2,
    n_states: int = 5,
    action_dims: List[int] = [3, 3],
    horizon: int = 10,
    **config_kwargs
) -> Tuple[MarkovGameConfig, MarkovGame]:
    """
    Create a random Markov game for testing.

    Returns:
        config: MarkovGameConfig
        game: MarkovGame with random payoffs and transitions
    """
    config = MarkovGameConfig(
        n_players=n_players,
        n_states=n_states,
        action_dims=action_dims,
        horizon=horizon,
        **config_kwargs
    )

    # Create random payoffs: R_i,h(s, a) ∈ [-1, 1]
    payoff_shape = [n_states] + action_dims
    payoffs = [
        [torch.rand(payoff_shape) * 2 - 1 for _ in range(horizon)]
        for _ in range(n_players)
    ]

    # Create random transitions: P_h(s' | s, a)
    trans_shape = [n_states] + action_dims + [n_states]
    transitions = []
    for h in range(horizon):
        P = torch.rand(trans_shape)
        # Normalize to get valid probabilities
        P = P / P.sum(dim=-1, keepdim=True)
        transitions.append(P)

    game = MarkovGame(config=config, payoffs=payoffs, transitions=transitions)

    return config, game


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Markov RQE Solver Test")
    print("=" * 70)

    # Test 1: Simple 2-player game
    print("\n1. Simple 2-Player Markov Game")
    print("-" * 50)

    config, game = create_random_markov_game(
        n_players=2,
        n_states=3,
        action_dims=[2, 2],
        horizon=5,
        tau=[2.0, 2.0],
        epsilon=[1.0, 1.0],
    )

    print(f"Game specification:")
    print(f"  Players: {config.n_players}")
    print(f"  States: {config.n_states}")
    print(f"  Actions: {config.action_dims}")
    print(f"  Horizon: {config.horizon}")
    print(f"  τ: {config.tau}")
    print(f"  ε: {config.epsilon}")
    print(f"  Tractable: {config.is_tractable}")

    solver = MarkovRQESolver(config)
    policies, values = solver.solve(game)

    print(f"\nEquilibrium computed successfully!")
    print(f"\nPolicies at h=0:")
    for i in range(config.n_players):
        print(f"  Player {i+1}:")
        for s in range(config.n_states):
            print(f"    State {s}: {policies[i][0][s].cpu().numpy()}")

    print(f"\nValues at h=0:")
    for i in range(config.n_players):
        print(f"  Player {i+1}: {values[i][0].cpu().numpy()}")

    # Test 2: Verify tractability condition
    print("\n2. Tractability Condition Test")
    print("-" * 50)

    # Tractable case
    config_ok, _ = create_random_markov_game(
        tau=[2.0, 2.0], epsilon=[1.0, 1.0]
    )
    print(f"τ=[2,2], ε=[1,1]: tractable={config_ok.is_tractable}, "
          f"margin={config_ok.tractability_margin:.4f}")

    # Intractable case (should warn)
    print("\nTesting intractable parameters (expect warning):")
    config_bad, _ = create_random_markov_game(
        tau=[1.0, 1.0], epsilon=[0.5, 0.5]
    )
    print(f"τ=[1,1], ε=[0.5,0.5]: tractable={config_bad.is_tractable}, "
          f"margin={config_bad.tractability_margin:.4f}")

    # Test 3: Deterministic transitions (matrix game per state)
    print("\n3. Deterministic Transitions (Pure Stage Games)")
    print("-" * 50)

    n_states, A1, A2, H = 2, 2, 2, 3

    # Create deterministic game: stay in same state
    config_det = MarkovGameConfig(
        n_players=2,
        n_states=n_states,
        action_dims=[A1, A2],
        horizon=H,
        tau=[2.0, 2.0],
        epsilon=[1.0, 1.0],
    )

    # Matching pennies payoffs at each state
    payoffs_det = []
    for i in range(2):
        player_payoffs = []
        for h in range(H):
            R = torch.zeros(n_states, A1, A2)
            for s in range(n_states):
                if i == 0:
                    R[s] = torch.tensor([[1., -1.], [-1., 1.]])
                else:
                    R[s] = torch.tensor([[-1., 1.], [1., -1.]])
            player_payoffs.append(R)
        payoffs_det.append(player_payoffs)

    # Deterministic transitions (stay in same state)
    transitions_det = []
    for h in range(H):
        P = torch.zeros(n_states, A1, A2, n_states)
        for s in range(n_states):
            P[s, :, :, s] = 1.0  # Always stay in state s
        transitions_det.append(P)

    game_det = MarkovGame(
        config=config_det,
        payoffs=payoffs_det,
        transitions=transitions_det
    )

    solver_det = MarkovRQESolver(config_det)
    policies_det, values_det = solver_det.solve(game_det)

    print("Matching Pennies at each state (should be ~[0.5, 0.5]):")
    for s in range(n_states):
        print(f"  State {s}:")
        print(f"    Player 1: {policies_det[0][0][s].cpu().numpy()}")
        print(f"    Player 2: {policies_det[1][0][s].cpu().numpy()}")

    # Test 4: Different risk parameters
    print("\n4. Effect of Risk Aversion")
    print("-" * 50)

    for tau in [0.5, 1.0, 2.0, 5.0]:
        config_risk, game_risk = create_random_markov_game(
            n_states=2,
            action_dims=[2, 2],
            horizon=3,
            tau=[tau, tau],
            epsilon=[1.0, 1.0],
        )

        solver_risk = MarkovRQESolver(config_risk)
        _, values_risk = solver_risk.solve(game_risk)

        avg_value = (values_risk[0][0].mean() + values_risk[1][0].mean()) / 2
        print(f"  τ = {tau}: avg equilibrium value = {avg_value:.4f}")

    print("\n" + "=" * 70)
    print("All tests complete!")
    print("=" * 70)
    print("\nKey Features:")
    print("  ✓ Backward induction for finite-horizon Markov games")
    print("  ✓ Risk-adjusted continuation values (entropic risk)")
    print("  ✓ Mirror descent for stage game equilibria")
    print("  ✓ Tractability condition checking")
    print("  ✓ Supports KL divergence penalty (extensible to others)")
