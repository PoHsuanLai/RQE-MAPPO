"""
Run Markov RQE Solver on Cliff Walk Environment

Reproduces the Cliff Walk experiment from:
"Tractable Multi-Agent Reinforcement Learning Through Behavioral Economics"
(Mazumdar et al., ICLR 2025) - Figure 2

The paper shows two scenarios:
1. Agent 2 more risk-averse (τ₁=0.01, τ₂=0.02) → Agent 2 hides from obstacles
2. Agent 2 less risk-averse (τ₁=0.01, τ₂=0.005) → Agent 2 reaches goal

Key insight: Risk-averse agents prefer paths that avoid uncertainty (proximity to cliffs
and other agents), even if expected payoff is lower.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.markov_rqe_solver import (
    MarkovRQESolver, MarkovGameConfig, MarkovGame
)


def create_cliff_walk_markov_game(
    grid_size=(6, 6),  # 6x6 grid (matching paper Figure 2)
    horizon=200,
    tau=(0.01, 0.02),  # Single tau per agent (matching paper)
    epsilon=(50.0, 100.0),
    cliff_cells=None,
    agent1_start=(4, 2),   # Agent 1 starts at row 2, col 4 (from paper Figure 2)
    agent2_start=(1, 2),   # Agent 2 starts at row 2, col 1 (from paper Figure 2)
    agent1_goal=(0, 0),    # Top-left (pink "Ag 1 Goal" in figure)
    agent2_goal=(5, 0),    # Row 3, col 0 (blue "Ag 2 Goal" in figure - between cliffs)
    pd_close=0.9,          # MORE reliable when close (agents cooperate when together)
    pd_far=0.5,            # LESS reliable when far apart (environment is risky)
):
    """
    Convert Cliff Walk environment to tabular Markov game.

    State space: Joint positions (row1, col1, row2, col2)
    Actions: {up, down, left, right} for each agent

    Returns:
        config: MarkovGameConfig
        game: MarkovGame
        state_to_pos: Mapping from state index to positions
    """
    height, width = grid_size

    # Default cliff configuration (from paper's Figure 2)
    # Looking at the figure (6x6 grid):
    # - Agent 1 goal at (0,0) - top-left (pink)
    # - Agent 2 goal at (3,0) - row 3, col 0 (blue)
    # - Vertical cliff column on left side (rows 1-2, col 0) - between the two goals
    # - 2x2 cliff block in center area (rows 3-4, cols 2-3)
    if cliff_cells is None:
        cliff_cells = [
            # Left column (rows 1,2 at col 0) - between goals
            (1, 0), (2, 0), (3, 0), (4, 0),
            # Center 2x2 block (rows 2-3, cols 2-3)
            (2, 2), (2, 3),
            (3, 2), (3, 3)
        ]
    cliff_set = set(cliff_cells)

    # Build state space
    # State = (r1, c1, r2, c2) for non-cliff positions
    # We include a "cliff" absorbing state for each agent

    valid_positions = []
    for r in range(height):
        for c in range(width):
            if (r, c) not in cliff_set:
                valid_positions.append((r, c))

    # For simplicity, we'll use the full grid as state space
    # States where an agent is in cliff will have special handling

    # State indexing: (r1, c1, r2, c2) -> index
    def pos_to_idx(r1, c1, r2, c2):
        return r1 * (width * height * width) + c1 * (height * width) + r2 * width + c2

    def idx_to_pos(idx):
        r1 = idx // (width * height * width)
        rem = idx % (width * height * width)
        c1 = rem // (height * width)
        rem = rem % (height * width)
        r2 = rem // width
        c2 = rem % width
        return (r1, c1, r2, c2)

    n_states = height * width * height * width
    n_actions = 4  # up, down, left, right

    # Action effects
    action_deltas = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }

    def apply_action(r, c, action):
        """Apply action, staying in bounds"""
        dr, dc = action_deltas[action]
        new_r = max(0, min(height - 1, r + dr))
        new_c = max(0, min(width - 1, c + dc))
        return new_r, new_c

    def get_distance(r1, c1, r2, c2):
        """Manhattan distance"""
        return abs(r1 - r2) + abs(c1 - c2)

    print(f"Building Markov game with {n_states} states...")
    print(f"Grid: {height}x{width}, Horizon: {horizon}")
    print(f"Agent 1: start={agent1_start}, goal={agent1_goal}")
    print(f"Agent 2: start={agent2_start}, goal={agent2_goal}")
    print(f"Risk parameters: τ={tau}, ε={epsilon}")

    # Build payoff matrices R_i,h(s, a) and transition P_h(s'|s, a)
    # For efficiency, we'll use sparse representations or compute on-the-fly
    # But for the solver, we need dense tensors

    # Due to memory constraints, we'll use a smaller effective state space
    # Only consider states reachable from start positions

    # Actually, let's simplify: use a smaller grid or reduced state space
    # The paper uses a 6x6 grid which gives 6^4 = 1296 joint states
    # That's manageable

    assert height <= 8 and width <= 8, "Grid too large for tabular solver"

    # Initialize tensors
    # payoffs[player][timestep]: [n_states, n_actions, n_actions]
    # transitions[timestep]: [n_states, n_actions, n_actions, n_states]

    # For efficiency, we'll build time-homogeneous payoffs and transitions
    # (same for all timesteps h)

    print("Building payoff matrices...")

    # Payoff for player i at state s taking joint action (a1, a2)
    # R_i(s, a) = reward based on next state

    R1 = torch.zeros(n_states, n_actions, n_actions)
    R2 = torch.zeros(n_states, n_actions, n_actions)
    P = torch.zeros(n_states, n_actions, n_actions, n_states)

    for s in tqdm(range(n_states), desc="Building states"):
        r1, c1, r2, c2 = idx_to_pos(s)

        # Skip if already in cliff (absorbing)
        in_cliff_1 = (r1, c1) in cliff_set
        in_cliff_2 = (r2, c2) in cliff_set

        for a1 in range(n_actions):
            for a2 in range(n_actions):
                # Determine stochasticity based on proximity
                # When agents are CLOSE (same cell or adjacent), MORE randomness (lower pd)
                # When agents are FAR apart, LESS randomness (higher pd)
                dist = get_distance(r1, c1, r2, c2)
                # dist == 0 means same cell, dist == 1 means adjacent
                pd = pd_close if dist <= 1 else pd_far  # pd_close < pd_far

                # Compute expected payoff and transition under stochastic dynamics
                # For each possible outcome...

                total_prob = 0.0
                expected_r1 = 0.0
                expected_r2 = 0.0

                # Iterate over actual actions (intended or random)
                for actual_a1 in range(n_actions):
                    for actual_a2 in range(n_actions):
                        # Probability of this action combination
                        p1 = pd if actual_a1 == a1 else (1 - pd) / 4
                        p2 = pd if actual_a2 == a2 else (1 - pd) / 4
                        prob = p1 * p2

                        # Apply actions
                        if in_cliff_1:
                            new_r1, new_c1 = r1, c1  # Stay in cliff
                        else:
                            new_r1, new_c1 = apply_action(r1, c1, actual_a1)

                        if in_cliff_2:
                            new_r2, new_c2 = r2, c2  # Stay in cliff
                        else:
                            new_r2, new_c2 = apply_action(r2, c2, actual_a2)

                        # Compute rewards (matching paper: +1 for goal, -2 for cliff)
                        # Scale rewards to be meaningful relative to ε parameters
                        # With ε=50-100, we need rewards ~O(ε) for meaningful policies
                        reward_scale = 50.0  # Scale factor to match paper's ε values
                        rew1 = 0.0
                        rew2 = 0.0

                        if (new_r1, new_c1) in cliff_set and not in_cliff_1:
                            rew1 = -2.0 * reward_scale  # Cliff penalty
                        elif (new_r1, new_c1) == agent1_goal:
                            rew1 = 1.0 * reward_scale   # Goal reward

                        if (new_r2, new_c2) in cliff_set and not in_cliff_2:
                            rew2 = -2.0 * reward_scale  # Cliff penalty
                        elif (new_r2, new_c2) == agent2_goal:
                            rew2 = 1.0 * reward_scale   # Goal reward

                        # Next state
                        s_next = pos_to_idx(new_r1, new_c1, new_r2, new_c2)

                        P[s, a1, a2, s_next] += prob
                        expected_r1 += prob * rew1
                        expected_r2 += prob * rew2
                        total_prob += prob

                R1[s, a1, a2] = expected_r1
                R2[s, a1, a2] = expected_r2

    print("Building game structure...")

    # Create time-homogeneous game (same payoffs and transitions at each timestep)
    payoffs = [
        [R1.clone() for _ in range(horizon)],
        [R2.clone() for _ in range(horizon)]
    ]
    transitions = [P.clone() for _ in range(horizon)]

    # Create config
    # Note: With very small τ values (0.01, 0.005), we need more iterations
    # and a smaller learning rate for stable convergence
    config = MarkovGameConfig(
        n_players=2,
        n_states=n_states,
        action_dims=[n_actions, n_actions],
        horizon=horizon,
        tau=list(tau),
        epsilon=list(epsilon),
        solver_iterations=500,  # Increased from 100
        solver_lr=0.1,          # Decreased from 0.3
    )

    game = MarkovGame(config=config, payoffs=payoffs, transitions=transitions)

    # Create state mapping
    state_info = {
        'pos_to_idx': pos_to_idx,
        'idx_to_pos': idx_to_pos,
        'grid_size': grid_size,
        'cliff_cells': cliff_set,
        'agent1_start': agent1_start,
        'agent2_start': agent2_start,
        'agent1_goal': agent1_goal,
        'agent2_goal': agent2_goal,
    }

    return config, game, state_info


def visualize_policy(
    policies,
    state_info,
    title="Cliff Walk RQE Policy",
    save_path=None
):
    """
    Visualize the equilibrium policies on the grid.

    Shows the most likely action direction for each agent at each position.
    """
    height, width = state_info['grid_size']
    cliff_cells = state_info['cliff_cells']
    pos_to_idx = state_info['pos_to_idx']
    agent1_goal = state_info['agent1_goal']
    agent2_goal = state_info['agent2_goal']
    agent1_start = state_info['agent1_start']
    agent2_start = state_info['agent2_start']

    # Get start state
    r1_s, c1_s = agent1_start
    r2_s, c2_s = agent2_start
    start_state = pos_to_idx(r1_s, c1_s, r2_s, c2_s)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    action_arrows = {
        0: (0, 0.3),   # up
        1: (0, -0.3),  # down
        2: (-0.3, 0),  # left
        3: (0.3, 0),   # right
    }

    for agent_idx, ax in enumerate(axes):
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect('equal')
        ax.set_title(f"Player {agent_idx + 1} Policy at h=0")

        # Draw grid
        for i in range(height + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        for j in range(width + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5)

        # Draw cliffs
        for (r, c) in cliff_cells:
            rect = Rectangle((c - 0.5, height - r - 1 - 0.5), 1, 1,
                            facecolor='black', edgecolor='black')
            ax.add_patch(rect)

        # Draw goals
        gr, gc = agent1_goal
        rect = Rectangle((gc - 0.5, height - gr - 1 - 0.5), 1, 1,
                        facecolor='lightcoral', alpha=0.5, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(gc, height - gr - 1, 'G1', ha='center', va='center', fontsize=10, fontweight='bold')

        gr, gc = agent2_goal
        rect = Rectangle((gc - 0.5, height - gr - 1 - 0.5), 1, 1,
                        facecolor='lightblue', alpha=0.5, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
        ax.text(gc, height - gr - 1, 'G2', ha='center', va='center', fontsize=10, fontweight='bold')

        # Draw policy arrows for this agent
        # We show the policy when the OTHER agent is at their start position
        other_agent = 1 - agent_idx
        if other_agent == 0:
            other_r, other_c = agent1_start
        else:
            other_r, other_c = agent2_start

        for r in range(height):
            for c in range(width):
                if (r, c) in cliff_cells:
                    continue

                # Get state index
                if agent_idx == 0:
                    s = pos_to_idx(r, c, other_r, other_c)
                else:
                    s = pos_to_idx(other_r, other_c, r, c)

                # Get policy for this state
                policy = policies[agent_idx][0][s].cpu().numpy()

                # Draw arrow in direction of most likely action
                best_action = np.argmax(policy)
                dx, dy = action_arrows[best_action]

                # Scale by probability
                prob = policy[best_action]
                color = 'red' if agent_idx == 0 else 'blue'
                alpha = 0.3 + 0.7 * prob

                ax.arrow(c, height - r - 1, dx * prob, dy * prob,
                        head_width=0.15, head_length=0.1,
                        fc=color, ec=color, alpha=alpha)

        # Mark start positions
        sr, sc = agent1_start
        ax.plot(sc, height - sr - 1, 'ro', markersize=12, markeredgecolor='darkred', markeredgewidth=2)
        sr, sc = agent2_start
        ax.plot(sc, height - sr - 1, 'bs', markersize=12, markeredgecolor='darkblue', markeredgewidth=2)

        ax.set_xlabel('Column')
        ax.set_ylabel('Row (flipped)')
        ax.invert_yaxis()

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def simulate_trajectory(policies, state_info, config, max_steps=50, debug=True):
    """
    Simulate a trajectory using the RQE policies.

    Returns the most likely path (greedy w.r.t. policy).
    """
    height, width = state_info['grid_size']
    cliff_cells = state_info['cliff_cells']
    pos_to_idx = state_info['pos_to_idx']
    idx_to_pos = state_info['idx_to_pos']
    agent1_goal = state_info['agent1_goal']
    agent2_goal = state_info['agent2_goal']

    # Start state
    r1, c1 = state_info['agent1_start']
    r2, c2 = state_info['agent2_start']

    trajectory = [(r1, c1, r2, c2)]

    action_names = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
    action_deltas = {
        0: (-1, 0),  # up
        1: (1, 0),   # down
        2: (0, -1),  # left
        3: (0, 1),   # right
    }

    if debug:
        print(f"  Starting at: Agent1=({r1},{c1}), Agent2=({r2},{c2})")

    for t in range(min(max_steps, config.horizon)):
        s = pos_to_idx(r1, c1, r2, c2)

        # Get greedy actions
        policy1 = policies[0][t][s].cpu().numpy()
        policy2 = policies[1][t][s].cpu().numpy()

        # Debug: Check for nan
        if np.any(np.isnan(policy1)) or np.any(np.isnan(policy2)):
            if debug:
                print(f"  Step {t}: NaN in policy! policy1={policy1}, policy2={policy2}")
            break

        a1 = np.argmax(policy1)
        a2 = np.argmax(policy2)

        if debug and t < 5:  # Print first 5 steps
            print(f"  Step {t}: s={s}, policy1={policy1}, a1={action_names[a1]}")
            print(f"           policy2={policy2}, a2={action_names[a2]}")

        # Apply actions (deterministic for visualization)
        dr1, dc1 = action_deltas[a1]
        dr2, dc2 = action_deltas[a2]

        new_r1 = max(0, min(height - 1, r1 + dr1))
        new_c1 = max(0, min(width - 1, c1 + dc1))
        new_r2 = max(0, min(height - 1, r2 + dr2))
        new_c2 = max(0, min(width - 1, c2 + dc2))

        # Check cliff
        if (new_r1, new_c1) in cliff_cells:
            if debug and t < 5:
                print(f"  Agent 1 would hit cliff at ({new_r1},{new_c1}), staying at ({r1},{c1})")
            new_r1, new_c1 = r1, c1
        if (new_r2, new_c2) in cliff_cells:
            if debug and t < 5:
                print(f"  Agent 2 would hit cliff at ({new_r2},{new_c2}), staying at ({r2},{c2})")
            new_r2, new_c2 = r2, c2

        # Check if agents moved
        moved = (new_r1 != r1 or new_c1 != c1 or new_r2 != r2 or new_c2 != c2)

        r1, c1, r2, c2 = new_r1, new_c1, new_r2, new_c2
        trajectory.append((r1, c1, r2, c2))

        if debug and t < 5:
            print(f"  -> Agent1=({r1},{c1}), Agent2=({r2},{c2}), moved={moved}")

        # Check if both reached goals
        if (r1, c1) == agent1_goal and (r2, c2) == agent2_goal:
            print(f"Both agents reached goals at step {t+1}!")
            break

        # Early termination if stuck (no movement for many steps)
        if len(trajectory) > 5:
            last_5 = trajectory[-5:]
            if all(pos == last_5[0] for pos in last_5):
                if debug:
                    print(f"  Agents stuck at same position for 5 steps, stopping.")
                break

    if debug:
        print(f"  Trajectory length: {len(trajectory)}")

    return trajectory


def visualize_trajectory(trajectory, state_info, title="Trajectory", save_path=None):
    """Visualize a trajectory on the grid.

    Coordinate system: row 0 at top, row 5 at bottom (matching paper Figure 2).
    """
    height, width = state_info['grid_size']
    cliff_cells = state_info['cliff_cells']
    agent1_goal = state_info['agent1_goal']
    agent2_goal = state_info['agent2_goal']

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    ax.set_aspect('equal')

    # Draw grid
    for i in range(height + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
    for j in range(width + 1):
        ax.axvline(j - 0.5, color='gray', linewidth=0.5)

    # Draw cliffs - use row directly (y-axis will be inverted)
    for (r, c) in cliff_cells:
        rect = Rectangle((c - 0.5, r - 0.5), 1, 1,
                        facecolor='black', edgecolor='black')
        ax.add_patch(rect)

    # Draw goals
    gr, gc = agent1_goal
    rect = Rectangle((gc - 0.5, gr - 0.5), 1, 1,
                    facecolor='lightcoral', alpha=0.5, edgecolor='red', linewidth=2)
    ax.add_patch(rect)
    ax.text(gc, gr, 'G1', ha='center', va='center', fontsize=12, fontweight='bold')

    gr, gc = agent2_goal
    rect = Rectangle((gc - 0.5, gr - 0.5), 1, 1,
                    facecolor='lightblue', alpha=0.5, edgecolor='blue', linewidth=2)
    ax.add_patch(rect)
    ax.text(gc, gr, 'G2', ha='center', va='center', fontsize=12, fontweight='bold')

    # Draw trajectories - use row directly
    traj1 = [(c1, r1) for (r1, c1, r2, c2) in trajectory]
    traj2 = [(c2, r2) for (r1, c1, r2, c2) in trajectory]

    # Agent 1 path (red)
    for i in range(len(traj1) - 1):
        ax.annotate('', xy=traj1[i+1], xytext=traj1[i],
                   arrowprops=dict(arrowstyle='->', color='red', lw=2, alpha=0.7))
    ax.plot(*zip(*traj1), 'ro-', markersize=6, alpha=0.5, label='Agent 1')

    # Agent 2 path (blue)
    for i in range(len(traj2) - 1):
        ax.annotate('', xy=traj2[i+1], xytext=traj2[i],
                   arrowprops=dict(arrowstyle='->', color='blue', lw=2, alpha=0.7))
    ax.plot(*zip(*traj2), 'bs-', markersize=6, alpha=0.5, label='Agent 2')

    # Mark start and end
    r1_s, c1_s, r2_s, c2_s = trajectory[0]
    ax.plot(c1_s, r1_s, 'ro', markersize=15, markeredgecolor='darkred', markeredgewidth=2, label='A1 Start')
    ax.plot(c2_s, r2_s, 'bs', markersize=15, markeredgecolor='darkblue', markeredgewidth=2, label='A2 Start')

    r1_e, c1_e, r2_e, c2_e = trajectory[-1]
    ax.plot(c1_e, r1_e, 'r*', markersize=20, markeredgecolor='darkred', markeredgewidth=1)
    ax.plot(c2_e, r2_e, 'b*', markersize=20, markeredgecolor='darkblue', markeredgewidth=1)

    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.set_title(title)
    ax.legend(loc='upper right')

    # Invert y-axis so row 0 is at top (matching paper convention)
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory to {save_path}")

    plt.show()


def main():
    print("=" * 70)
    print("Markov RQE on Cliff Walk - Reproducing Paper Results")
    print("=" * 70)

    # Use paper's horizon or smaller for faster computation
    horizon = 100  # Paper uses 200

    # ========================================
    # Scenario 1: Agent 2 More Risk-Averse
    # From paper Figure 2 (left):
    #   τ₁=0.01, τ₂=0.02  (τ₂ > τ₁ means Agent 2 is MORE risk-averse)
    #   ε₁=50, ε₂=100
    # Expected: Agent 2 goes UP and around to avoid Agent 1
    # ========================================
    print("\n" + "=" * 70)
    print("Scenario 1: Agent 2 More Risk-Averse")
    print("τ₁=0.01, τ₂=0.02 (higher τ = MORE risk-averse)")
    print("ε₁=50, ε₂=100")
    print("=" * 70)

    # Use EXACT parameters from paper Figure 2 (left panel)
    # Tractability: ε₁·ε₂ = 5000 = 1/(τ₁·τ₂) (at boundary)
    config1, game1, state_info = create_cliff_walk_markov_game(
        horizon=horizon,
        tau=[0.01, 0.02],
        epsilon=[50.0, 100.0],  # EXACT paper values
    )

    print("\nSolving for Markov RQE...")
    solver1 = MarkovRQESolver(config1)

    try:
        policies1, values1 = solver1.solve(game1)
        print("Equilibrium computed!")

        # Get start state values
        start_idx = state_info['pos_to_idx'](*state_info['agent1_start'], *state_info['agent2_start'])
        print(f"\nValues at start state (h=0):")
        print(f"  Player 1: {values1[0][0][start_idx].item():.4f}")
        print(f"  Player 2: {values1[1][0][start_idx].item():.4f}")

        # Debug: Check values at goal states
        goal1_idx = state_info['pos_to_idx'](*state_info['agent1_goal'], *state_info['agent2_start'])
        goal2_idx = state_info['pos_to_idx'](*state_info['agent1_start'], *state_info['agent2_goal'])
        print(f"\nValue at Agent 1's goal (if agent 2 at start): {values1[0][0][goal1_idx].item():.4f}")
        print(f"Value at Agent 2's goal (if agent 1 at start): {values1[1][0][goal2_idx].item():.4f}")

        # Simulate trajectory
        print("\nSimulating greedy trajectory...")
        traj1 = simulate_trajectory(policies1, state_info, config1, max_steps=100)

        # Visualize
        visualize_trajectory(
            traj1, state_info,
            title="Scenario 1: Agent 2 More Risk-Averse\n(τ₁=0.01, τ₂=0.02)",
            save_path="results/cliffwalk_rqe_scenario1.png"
        )

    except Exception as e:
        print(f"Error in Scenario 1: {e}")
        import traceback
        traceback.print_exc()

    # ========================================
    # Scenario 2: Agent 2 Less Risk-Averse
    # From paper Figure 2 (right):
    #   τ₁=0.01, τ₂=0.005  (τ₂ < τ₁ means Agent 2 is LESS risk-averse)
    #   ε₁=100, ε₂=200
    # Expected: Both agents reach their goals via bottom path
    # ========================================
    print("\n" + "=" * 70)
    print("Scenario 2: Agent 2 Less Risk-Averse")
    print("τ₁=0.01, τ₂=0.005 (lower τ = LESS risk-averse)")
    print("ε₁=100, ε₂=200")
    print("=" * 70)

    # Use default grid_size from function (6x7 matching paper)
    config2, game2, state_info = create_cliff_walk_markov_game(
        horizon=horizon,
        tau=[0.01, 0.005],
        epsilon=[100.0, 200.0],
    )

    print("\nSolving for Markov RQE...")
    solver2 = MarkovRQESolver(config2)

    try:
        policies2, values2 = solver2.solve(game2)
        print("Equilibrium computed!")

        # Get start state values
        start_idx = state_info['pos_to_idx'](*state_info['agent1_start'], *state_info['agent2_start'])
        print(f"\nValues at start state (h=0):")
        print(f"  Player 1: {values2[0][0][start_idx].item():.4f}")
        print(f"  Player 2: {values2[1][0][start_idx].item():.4f}")

        # Simulate trajectory
        print("\nSimulating greedy trajectory...")
        traj2 = simulate_trajectory(policies2, state_info, config2, max_steps=100)

        # Visualize
        visualize_trajectory(
            traj2, state_info,
            title="Scenario 2: Agent 2 Less Risk-Averse\n(τ₁=0.01, τ₂=0.005)",
            save_path="results/cliffwalk_rqe_scenario2.png"
        )

    except Exception as e:
        print(f"Error in Scenario 2: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 70)
    print("Experiment Complete!")
    print("=" * 70)
    print("\nExpected behavior from paper:")
    print("  Scenario 1: Agent 2 (more risk-averse) hides far from obstacles")
    print("  Scenario 2: Agent 2 (less risk-averse) successfully reaches goal")
    print("\nNote: Due to smaller horizon and solver approximations,")
    print("results may differ slightly from paper.")


if __name__ == "__main__":
    # Create results directory if needed
    os.makedirs("results", exist_ok=True)
    main()
