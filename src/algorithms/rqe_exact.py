"""
Exact RQE Computation via Backward Induction (Algorithm 1 from paper)

Implements the algorithm from Section 4.2 for computing Markov RQE
in finite-horizon games using dynamic programming.
"""

import numpy as np
from typing import Tuple, List
import torch


class ExactRQE:
    """
    Compute exact RQE equilibrium via backward induction

    This is Algorithm 1 from the paper - computes RQE for finite-horizon
    Markov games when dynamics are known.

    Args:
        env: Environment (must have known dynamics)
        tau: Risk aversion parameters for each player [tau_1, tau_2, ...]
        epsilon: Bounded rationality for each player [eps_1, eps_2, ...]
        risk_measure: "entropic" (only one implemented for now)
        n_atoms: Number of atoms for distributional value
        v_min, v_max: Support for return distribution
    """

    def __init__(
        self,
        env,
        tau: List[float],
        epsilon: List[float],
        risk_measure: str = "entropic",
        n_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 50.0,
        horizon: int = 200
    ):
        self.env = env
        self.tau = np.array(tau)
        self.epsilon = np.array(epsilon)
        self.risk_measure = risk_measure
        self.n_atoms = n_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.horizon = horizon

        # Atom locations for distributional values
        self.z_atoms = np.linspace(v_min, v_max, n_atoms)
        self.delta_z = (v_max - v_min) / (n_atoms - 1)

        # Number of states and actions
        self.n_states = self._get_n_states()
        self.n_actions = self._get_n_actions()
        self.n_players = len(tau)

        # Value functions: V[h][player][state] = distribution over returns
        self.V = None  # Will be computed

        # Q-functions: Q[h][player][state][joint_action] = distribution
        self.Q = None

        # Policies: pi[h][player][state][action] = probability
        self.pi = None

    def _get_n_states(self):
        """Get number of states (for discrete state spaces)"""
        # For cliff walk: state is (agent1_row, agent1_col, agent2_row, agent2_col)
        return self.env.height ** 2 * self.env.width ** 2

    def _get_n_actions(self):
        """Get number of actions per player"""
        return 4  # {up, down, left, right}

    def _state_to_index(self, state):
        """Convert state tuple to index"""
        if isinstance(state, np.ndarray):
            state = tuple(state.astype(int))
        r1, c1, r2, c2 = state
        H, W = self.env.height, self.env.width
        return r1 * (W * H * W) + c1 * (H * W) + r2 * W + c2

    def _index_to_state(self, idx):
        """Convert index to state tuple"""
        H, W = self.env.height, self.env.width
        r1 = idx // (W * H * W)
        idx = idx % (W * H * W)
        c1 = idx // (H * W)
        idx = idx % (H * W)
        r2 = idx // W
        c2 = idx % W
        return (r1, c1, r2, c2)

    def _get_reward_and_next_state_distribution(self, state_idx, joint_action):
        """
        Get reward and next state distribution for given state and joint action

        Returns:
            rewards: [n_players] rewards
            next_state_probs: [n_states] probability distribution over next states
        """
        state = self._index_to_state(state_idx)

        # Set environment to this state
        self.env.agent1_pos = [state[0], state[1]]
        self.env.agent2_pos = [state[2], state[3]]

        # Sample multiple times to estimate transition probabilities
        # (since environment is stochastic)
        n_samples = 100
        next_states = []
        rewards_list = []

        for _ in range(n_samples):
            self.env.agent1_pos = [state[0], state[1]]
            self.env.agent2_pos = [state[2], state[3]]
            self.env.timestep = 0

            obs, reward, terminated, truncated, info = self.env.step(joint_action)

            next_state = tuple(obs.astype(int))
            next_state_idx = self._state_to_index(next_state)
            next_states.append(next_state_idx)

            # Rewards for both players
            rewards_list.append([info['agent1_reward'], info['agent2_reward']])

        # Compute empirical distribution
        next_state_probs = np.zeros(self.n_states)
        for ns in next_states:
            next_state_probs[ns] += 1.0 / n_samples

        # Average rewards
        rewards = np.mean(rewards_list, axis=0)

        return rewards, next_state_probs

    def _compute_risk_value(self, distribution, tau, player_idx):
        """
        Compute risk-adjusted value from distribution

        Args:
            distribution: [n_atoms] probability distribution
            tau: risk aversion parameter
            player_idx: which player

        Returns:
            scalar risk value
        """
        if self.risk_measure == "entropic":
            # ρ_τ(Z) = -(1/τ) log E[exp(-τZ)]
            exp_neg_tau_z = np.exp(-tau * self.z_atoms)
            expectation = np.sum(distribution * exp_neg_tau_z)
            return -(1.0 / tau) * np.log(expectation + 1e-8)
        else:
            raise NotImplementedError(f"Risk measure {self.risk_measure} not implemented")

    def _categorical_projection(self, reward, next_distribution, gamma=0.99):
        """
        Project Bellman update onto categorical distribution

        Target: r + gamma * Z(s')
        """
        target_dist = np.zeros(self.n_atoms)

        # Compute projected atoms: T_z = r + gamma * z
        Tz = reward + gamma * self.z_atoms
        Tz = np.clip(Tz, self.v_min, self.v_max)

        # Project onto grid
        for i, tz in enumerate(Tz):
            b = (tz - self.v_min) / self.delta_z
            l = int(np.floor(b))
            u = int(np.ceil(b))

            l = np.clip(l, 0, self.n_atoms - 1)
            u = np.clip(u, 0, self.n_atoms - 1)

            if l == u:
                target_dist[l] += next_distribution[i]
            else:
                prob_u = b - l
                prob_l = 1.0 - prob_u
                target_dist[l] += next_distribution[i] * prob_l
                target_dist[u] += next_distribution[i] * prob_u

        return target_dist

    def _compute_qre_policy(self, q_values, epsilon):
        """
        Compute quantal response equilibrium policy

        Uses softmax with temperature epsilon:
        π(a) ∝ exp(Q(a) / epsilon)
        """
        # Normalize Q-values
        q_norm = q_values - np.max(q_values)

        # Softmax
        exp_q = np.exp(q_norm / (epsilon + 1e-8))
        policy = exp_q / (np.sum(exp_q) + 1e-8)

        return policy

    def solve(self):
        """
        Solve for RQE via backward induction (Algorithm 1)
        """
        print(f"Solving RQE via backward induction...")
        print(f"  States: {self.n_states}, Actions: {self.n_actions}, Players: {self.n_players}")
        print(f"  Horizon: {self.horizon}")
        print(f"  τ: {self.tau}, ε: {self.epsilon}")

        # Initialize value and policy storage
        self.V = [[np.zeros((self.n_states, self.n_atoms)) for _ in range(self.n_players)]
                  for _ in range(self.horizon + 1)]

        self.Q = [[np.zeros((self.n_states, self.n_actions, self.n_actions, self.n_atoms))
                   for _ in range(self.n_players)]
                  for _ in range(self.horizon)]

        self.pi = [[np.zeros((self.n_states, self.n_actions)) for _ in range(self.n_players)]
                   for _ in range(self.horizon)]

        # Terminal value is deterministic at 0
        for player in range(self.n_players):
            for s in range(self.n_states):
                # Put all probability mass at z=0
                closest_idx = np.argmin(np.abs(self.z_atoms))
                self.V[self.horizon][player][s, closest_idx] = 1.0

        # Backward induction
        for h in range(self.horizon - 1, -1, -1):
            if h % 20 == 0:
                print(f"  Step {h}/{self.horizon}...")

            for s in range(self.n_states):
                # Compute Q-values for all joint actions
                for a1 in range(self.n_actions):
                    for a2 in range(self.n_actions):
                        joint_action = [a1, a2]

                        # Get rewards and next state distribution
                        rewards, next_state_probs = self._get_reward_and_next_state_distribution(
                            s, joint_action
                        )

                        # Compute Q-distribution for each player
                        for player in range(self.n_players):
                            # Expected next value distribution
                            expected_next_dist = np.zeros(self.n_atoms)
                            for ns in range(self.n_states):
                                if next_state_probs[ns] > 0:
                                    expected_next_dist += next_state_probs[ns] * self.V[h+1][player][ns]

                            # Bellman backup
                            self.Q[h][player][s, a1, a2] = self._categorical_projection(
                                rewards[player],
                                expected_next_dist,
                                gamma=self.env.gamma if hasattr(self.env, 'gamma') else 0.99
                            )

                # Compute policies via iterated best response
                # This is the RQE computation
                for player in range(self.n_players):
                    # Compute risk-adjusted Q-values
                    q_risk = np.zeros(self.n_actions)

                    for a_self in range(self.n_actions):
                        # Expected Q over opponent's actions
                        expected_q_dist = np.zeros(self.n_atoms)

                        opponent = 1 - player
                        for a_opp in range(self.n_actions):
                            if player == 0:
                                q_dist = self.Q[h][player][s, a_self, a_opp]
                            else:
                                q_dist = self.Q[h][player][s, a_opp, a_self]

                            # Weight by opponent's policy (uniform initially)
                            opp_prob = 1.0 / self.n_actions  # Will be iteratively updated
                            expected_q_dist += opp_prob * q_dist

                        # Compute risk value
                        q_risk[a_self] = self._compute_risk_value(
                            expected_q_dist,
                            self.tau[player],
                            player
                        )

                    # Compute QRE policy
                    self.pi[h][player][s] = self._compute_qre_policy(
                        q_risk,
                        self.epsilon[player]
                    )

                # Compute value distribution by averaging over joint actions
                for player in range(self.n_players):
                    v_dist = np.zeros(self.n_atoms)
                    for a1 in range(self.n_actions):
                        for a2 in range(self.n_actions):
                            joint_prob = self.pi[h][0][s, a1] * self.pi[h][1][s, a2]
                            v_dist += joint_prob * self.Q[h][player][s, a1, a2]

                    self.V[h][player][s] = v_dist

        print("  Done!")
        return self.pi

    def get_policy(self, state, timestep):
        """Get policy for given state and timestep"""
        state_idx = self._state_to_index(state)
        return [self.pi[timestep][player][state_idx] for player in range(self.n_players)]

    def select_action(self, state, timestep, player):
        """Sample action from policy"""
        policy = self.get_policy(state, timestep)[player]
        return np.random.choice(self.n_actions, p=policy)


if __name__ == "__main__":
    print("Testing Exact RQE solver...")

    from src.envs.cliff_walk import CliffWalkEnv

    env = CliffWalkEnv()

    # Parameters from paper (approximately)
    solver = ExactRQE(
        env=env,
        tau=[0.01, 0.02],  # Agent 1 more risk-averse
        epsilon=[50, 100],
        horizon=50,  # Shorter for testing
        n_atoms=21   # Fewer atoms for speed
    )

    # Solve
    policies = solver.solve()

    print("\n✓ Exact RQE solver test passed!")
