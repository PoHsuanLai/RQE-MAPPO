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

Usage:
    # Default paper scenarios
    python scripts/run_markov_rqe_cliffwalk.py

    # Custom parameters with value visualization
    python scripts/run_markov_rqe_cliffwalk.py --tau 1.0 1.0 --epsilon 0.1 0.1 --visualize_values

    # Compare different tau values
    python scripts/run_markov_rqe_cliffwalk.py --compare_tau
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import Rectangle, FancyArrowPatch
from pathlib import Path
from tqdm import tqdm
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms.markov_rqe_solver import (
    MarkovRQESolver, MarkovGameConfig, MarkovGame
)
from src.envs.cliff_walk import CliffWalkEnv, visualize_trajectory as visualize_trajectory_common


def create_cliff_walk_markov_game(
    horizon=200,
    tau=(1.0, 1.0),
    epsilon=(1.0, 1.0),
    reward_scale=1.0,
    enable_collision=False,
    n_samples=100,
    deterministic=False,
):
    """
    Build tabular Markov game by sampling from the actual CliffWalkEnv.

    This automatically captures all environment dynamics including:
    - Proximity-based stochasticity
    - Collision mechanics (if enabled)
    - Any other environment features

    Args:
        horizon: Episode length
        tau: Risk aversion parameters per agent
        epsilon: Entropy coefficients per agent
        reward_scale: Reward scaling factor
        enable_collision: Enable collision dynamics
        n_samples: Number of samples per (state, action) pair to estimate transitions
        deterministic: Use more deterministic environment

    Returns:
        config: MarkovGameConfig
        game: MarkovGame
        state_info: State mapping information
    """
    # Create environment
    env = CliffWalkEnv(
        grid_size=(6, 6),
        horizon=horizon,
        reward_scale=reward_scale,
        return_joint_reward=False,
        enable_collision=enable_collision,
    )
    if deterministic:
        env.pd_close = 0.95
        env.pd_far = 0.85

    height, width = env.height, env.width
    n_actions = 4

    # State indexing
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
    cliff_set = set(env.cliff_cells)

    print(f"Building Markov game from environment with {n_states} states...")
    print(f"Grid: {height}x{width}, Horizon: {horizon}")
    print(f"Collision enabled: {enable_collision}")
    print(f"Reward scale: {reward_scale}")
    print(f"Samples per (s,a): {n_samples}")
    print(f"Risk parameters: τ={tau}, ε={epsilon}")

    # Initialize tensors
    R1 = torch.zeros(n_states, n_actions, n_actions)
    R2 = torch.zeros(n_states, n_actions, n_actions)
    P = torch.zeros(n_states, n_actions, n_actions, n_states)

    # Goal positions
    agent1_goal = env.agent1_goal
    agent2_goal = env.agent2_goal

    # Sample transitions from environment
    for s in tqdm(range(n_states), desc="Sampling transitions"):
        r1, c1, r2, c2 = idx_to_pos(s)

        # Skip cliff states (absorbing with zero reward)
        if (r1, c1) in cliff_set or (r2, c2) in cliff_set:
            for a1 in range(n_actions):
                for a2 in range(n_actions):
                    P[s, a1, a2, s] = 1.0
            continue

        # Goal states are also absorbing - agent stays at goal and keeps getting reward
        agent1_at_goal = (r1, c1) == agent1_goal
        agent2_at_goal = (r2, c2) == agent2_goal

        if agent1_at_goal or agent2_at_goal:
            # Self-loop with goal rewards
            for a1 in range(n_actions):
                for a2 in range(n_actions):
                    P[s, a1, a2, s] = 1.0
                    if agent1_at_goal:
                        R1[s, a1, a2] = 1.0 * reward_scale  # Goal reward
                    if agent2_at_goal:
                        R2[s, a1, a2] = 1.0 * reward_scale  # Goal reward
            continue

        for a1 in range(n_actions):
            for a2 in range(n_actions):
                # Sample multiple times to estimate transition distribution
                next_state_counts = {}
                total_r1 = 0.0
                total_r2 = 0.0

                for _ in range(n_samples):
                    # Set environment to this state
                    env.agent1_pos = [r1, c1]
                    env.agent2_pos = [r2, c2]
                    env.timestep = 0

                    # Take action
                    _, rewards, terminated, _, info = env.step((a1, a2))

                    # Get next state
                    new_r1, new_c1 = env.agent1_pos
                    new_r2, new_c2 = env.agent2_pos

                    # If terminated (fell in cliff), mark as cliff state
                    if terminated:
                        # Check which agent fell
                        if (new_r1, new_c1) in cliff_set:
                            pass  # Agent 1 in cliff
                        if (new_r2, new_c2) in cliff_set:
                            pass  # Agent 2 in cliff

                    s_next = pos_to_idx(new_r1, new_c1, new_r2, new_c2)
                    next_state_counts[s_next] = next_state_counts.get(s_next, 0) + 1
                    total_r1 += rewards[0]
                    total_r2 += rewards[1]

                # Convert counts to probabilities
                for s_next, count in next_state_counts.items():
                    P[s, a1, a2, s_next] = count / n_samples

                # Average rewards
                R1[s, a1, a2] = total_r1 / n_samples
                R2[s, a1, a2] = total_r2 / n_samples

    print("Building game structure...")

    # Create time-homogeneous game
    payoffs = [
        [R1.clone() for _ in range(horizon)],
        [R2.clone() for _ in range(horizon)]
    ]
    transitions = [P.clone() for _ in range(horizon)]

    # Create action masks for border states
    # Actions: UP=0, DOWN=1, LEFT=2, RIGHT=3
    # Mask invalid actions that would bump into walls
    action_mask1 = torch.ones(n_states, n_actions)  # [S, 4]
    action_mask2 = torch.ones(n_states, n_actions)  # [S, 4]

    for s in range(n_states):
        r1, c1, r2, c2 = idx_to_pos(s)

        # Agent 1 action masks based on position (r1, c1)
        if r1 == 0:  # Top border - can't go UP
            action_mask1[s, 0] = 0
        if r1 == height - 1:  # Bottom border - can't go DOWN
            action_mask1[s, 1] = 0
        if c1 == 0:  # Left border - can't go LEFT
            action_mask1[s, 2] = 0
        if c1 == width - 1:  # Right border - can't go RIGHT
            action_mask1[s, 3] = 0

        # Agent 2 action masks based on position (r2, c2)
        if r2 == 0:  # Top border - can't go UP
            action_mask2[s, 0] = 0
        if r2 == height - 1:  # Bottom border - can't go DOWN
            action_mask2[s, 1] = 0
        if c2 == 0:  # Left border - can't go LEFT
            action_mask2[s, 2] = 0
        if c2 == width - 1:  # Right border - can't go RIGHT
            action_mask2[s, 3] = 0

    action_masks = [action_mask1, action_mask2]

    # Create config
    config = MarkovGameConfig(
        n_players=2,
        n_states=n_states,
        action_dims=[n_actions, n_actions],
        horizon=horizon,
        tau=list(tau),
        epsilon=list(epsilon),
        solver_iterations=500,
        solver_lr=0.1,
    )

    game = MarkovGame(config=config, payoffs=payoffs, transitions=transitions, action_masks=action_masks)

    # State info
    state_info = {
        'pos_to_idx': pos_to_idx,
        'idx_to_pos': idx_to_pos,
        'grid_size': (height, width),
        'cliff_cells': cliff_set,
        'agent1_start': tuple(env.agent1_start),
        'agent2_start': tuple(env.agent2_start),
        'agent1_goal': env.agent1_goal,
        'agent2_goal': env.agent2_goal,
        'enable_collision': enable_collision,
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


def simulate_trajectory(policies, state_info, config, max_steps=50, debug=True, enable_collision=False):
    """
    Simulate a trajectory using the RQE policies.

    Returns the most likely path (greedy w.r.t. policy).

    Args:
        policies: RQE policies [agent][timestep][state]
        state_info: Grid info dict
        config: MarkovGameConfig
        max_steps: Maximum simulation steps
        debug: Print debug output
        enable_collision: If True, apply collision dynamics during simulation
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

    def random_push(r, c):
        """Push agent in a random valid direction."""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        np.random.shuffle(directions)
        for dr, dc in directions:
            new_r = max(0, min(height - 1, r + dr))
            new_c = max(0, min(width - 1, c + dc))
            if (new_r, new_c) not in cliff_cells:
                return new_r, new_c
        return r, c  # Stay if no valid direction

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

        # Use argmax (first index wins ties)
        a1 = np.argmax(policy1)
        a2 = np.argmax(policy2)

        if debug and t < 10:  # Print first 10 steps
            print(f"  Step {t}: pos1=({r1},{c1}), pos2=({r2},{c2})")
            print(f"    Agent1 policy: UP={policy1[0]:.3f}, DOWN={policy1[1]:.3f}, LEFT={policy1[2]:.3f}, RIGHT={policy1[3]:.3f} -> {action_names[a1]}")
            print(f"    Agent2 policy: UP={policy2[0]:.3f}, DOWN={policy2[1]:.3f}, LEFT={policy2[2]:.3f}, RIGHT={policy2[3]:.3f} -> {action_names[a2]}")

        # Apply actions (deterministic for visualization)
        dr1, dc1 = action_deltas[a1]
        dr2, dc2 = action_deltas[a2]

        new_r1 = max(0, min(height - 1, r1 + dr1))
        new_c1 = max(0, min(width - 1, c1 + dc1))
        new_r2 = max(0, min(height - 1, r2 + dr2))
        new_c2 = max(0, min(width - 1, c2 + dc2))

        # Check collision if enabled
        collision_occurred = False
        if enable_collision:
            # Case 1: Same cell collision
            same_cell = (new_r1 == new_r2 and new_c1 == new_c2)

            # Case 2: Position swap (agents pass through each other)
            swap_collision = (new_r1 == r2 and new_c1 == c2 and new_r2 == r1 and new_c2 == c1)

            if same_cell or swap_collision:
                collision_occurred = True
                if debug:
                    collision_type = "same cell" if same_cell else "swap"
                    print(f"    COLLISION ({collision_type})! Pushing both agents randomly.")
                new_r1, new_c1 = random_push(new_r1, new_c1)
                new_r2, new_c2 = random_push(new_r2, new_c2)

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
        # But don't count agents at their goals as "stuck" - they're supposed to stay there
        if len(trajectory) > 5:
            last_5 = trajectory[-5:]
            # Check if agent 1 is stuck (not at goal and not moving)
            agent1_stuck = all(pos[0:2] == last_5[0][0:2] for pos in last_5) and (r1, c1) != agent1_goal
            # Check if agent 2 is stuck (not at goal and not moving)
            agent2_stuck = all(pos[2:4] == last_5[0][2:4] for pos in last_5) and (r2, c2) != agent2_goal

            if agent1_stuck and agent2_stuck:
                if debug:
                    print(f"  Both agents stuck (not at goals) for 5 steps, stopping.")
                break

    if debug:
        print(f"  Trajectory length: {len(trajectory)}")

    return trajectory


def visualize_trajectory(trajectory, state_info, title="Trajectory", save_path=None):
    """Visualize a trajectory on the grid using the common visualization function.

    Wrapper that extracts parameters from state_info and calls the common
    visualize_trajectory function from cliff_walk.py.

    Coordinate system: row 0 at top, row 5 at bottom (matching paper Figure 2).
    """
    height, width = state_info['grid_size']
    cliff_cells = list(state_info['cliff_cells'])
    agent1_goal = state_info['agent1_goal']
    agent2_goal = state_info['agent2_goal']

    # Use the common visualization function from cliff_walk.py
    visualize_trajectory_common(
        trajectory=trajectory,
        save_path=save_path,
        title=title,
        grid_size=height,  # Assuming square grid
        cliff_cells=cliff_cells,
        agent1_goal=agent1_goal,
        agent2_goal=agent2_goal,
        show=False  # Don't show interactively, just save
    )


def visualize_values_along_trajectory(trajectory, values, state_info, config,
                                       title="RQE Values Along Trajectory", save_path=None,
                                       max_steps=12):
    """
    Visualize RQE value heatmaps at each timestep along the trajectory.

    For each timestep, shows the value landscape given the opponent's ACTUAL
    position at that step. This reveals how the value function changes as
    agents move.

    Args:
        trajectory: List of (r1, c1, r2, c2) positions
        values: RQE values [agent][timestep][state]
        state_info: Grid info dict
        config: MarkovGameConfig
        title: Plot title
        save_path: Where to save
        max_steps: Maximum number of timesteps to show (for readability)
    """
    height, width = state_info['grid_size']
    cliff_cells = state_info['cliff_cells']
    pos_to_idx = state_info['pos_to_idx']
    agent1_goal = state_info['agent1_goal']
    agent2_goal = state_info['agent2_goal']

    # Limit steps to show
    n_steps = min(len(trajectory), max_steps, config.horizon)

    # Create figure: 2 rows (agents) x n_steps columns
    fig, axes = plt.subplots(2, n_steps, figsize=(3 * n_steps, 7))

    # Handle case where n_steps == 1
    if n_steps == 1:
        axes = axes.reshape(2, 1)

    for t in range(n_steps):
        r1, c1, r2, c2 = trajectory[t]

        for agent_idx in range(2):
            ax = axes[agent_idx, t]
            goal = agent1_goal if agent_idx == 0 else agent2_goal

            # Opponent's actual position at this timestep
            if agent_idx == 0:
                opp_pos = (r2, c2)
                my_pos = (r1, c1)
            else:
                opp_pos = (r1, c1)
                my_pos = (r2, c2)

            # Build value grid given opponent at their actual position
            values_grid = np.zeros((height, width))
            for r in range(height):
                for c in range(width):
                    if agent_idx == 0:
                        s = pos_to_idx(r, c, opp_pos[0], opp_pos[1])
                    else:
                        s = pos_to_idx(opp_pos[0], opp_pos[1], r, c)
                    # Use time-indexed values
                    values_grid[r, c] = values[agent_idx][t][s].item()

            im = ax.imshow(values_grid, cmap='RdYlGn', origin='upper')

            # Add value text (smaller font for many columns)
            fontsize = max(4, 7 - n_steps // 4)
            for r in range(height):
                for c in range(width):
                    if (r, c) not in cliff_cells:
                        val = values_grid[r, c]
                        ax.text(c, r, f'{val:.0f}', ha='center', va='center',
                               fontsize=fontsize, color='white', fontweight='bold',
                               path_effects=[pe.withStroke(linewidth=1.5, foreground='black')])

            # Mark cliffs
            for (cr, cc) in cliff_cells:
                ax.add_patch(plt.Rectangle((cc - 0.5, cr - 0.5), 1, 1,
                                           fill=True, facecolor='black', edgecolor='white', linewidth=1))

            # Mark goals
            ax.plot(agent1_goal[1], agent1_goal[0], 'r^', markersize=8,
                    markeredgecolor='white', markeredgewidth=1)
            ax.plot(agent2_goal[1], agent2_goal[0], 'bs', markersize=8,
                    markeredgecolor='white', markeredgewidth=1)

            # Mark my position with circle
            my_color = 'red' if agent_idx == 0 else 'blue'
            ax.plot(my_pos[1], my_pos[0], 'o', color=my_color, markersize=12,
                    markeredgecolor='white', markeredgewidth=2)

            # Mark opponent position with X
            ax.plot(opp_pos[1], opp_pos[0], 'mX', markersize=12,
                    markeredgecolor='white', markeredgewidth=1.5)

            # Title shows timestep and opponent position
            if agent_idx == 0:
                ax.set_title(f't={t}\nOpp@{opp_pos}', fontsize=8)
            else:
                ax.set_title(f'Opp@{opp_pos}', fontsize=8)

            ax.set_xticks([])
            ax.set_yticks([])

    # Add row labels
    axes[0, 0].set_ylabel('Agent 1', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Agent 2', fontsize=10, fontweight='bold')

    plt.suptitle(f'{title}\n(Circle=My pos, X=Opponent pos, Values shown for each cell)',
                 fontsize=11)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory values to {save_path}")

    plt.show()


def visualize_trajectory_with_values(trajectory, policies, values, state_info, config,
                                     title="Trajectory with Values", save_path=None):
    """
    Visualize trajectory with policy arrows and actual values at each step.

    This shows the TRUE joint-state values and policies along the trajectory.
    """
    height, width = state_info['grid_size']
    cliff_cells = state_info['cliff_cells']
    pos_to_idx = state_info['pos_to_idx']
    agent1_goal = state_info['agent1_goal']
    agent2_goal = state_info['agent2_goal']

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    action_arrows = {
        0: (0, -0.35),   # up (negative y in display)
        1: (0, 0.35),    # down
        2: (-0.35, 0),   # left
        3: (0.35, 0),    # right
    }

    for agent_idx, ax in enumerate(axes):
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(-0.5, height - 0.5)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # Draw grid
        for i in range(height + 1):
            ax.axhline(i - 0.5, color='gray', linewidth=0.5, alpha=0.5)
        for j in range(width + 1):
            ax.axvline(j - 0.5, color='gray', linewidth=0.5, alpha=0.5)

        # Draw cliffs
        for (r, c) in cliff_cells:
            rect = Rectangle((c - 0.5, r - 0.5), 1, 1,
                            facecolor='black', edgecolor='black')
            ax.add_patch(rect)

        # Draw goals
        gr, gc = agent1_goal
        rect = Rectangle((gc - 0.5, gr - 0.5), 1, 1,
                        facecolor='lightcoral', alpha=0.4, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(gc, gr, 'G1', ha='center', va='center', fontsize=10, fontweight='bold', color='darkred')

        gr, gc = agent2_goal
        rect = Rectangle((gc - 0.5, gr - 0.5), 1, 1,
                        facecolor='lightblue', alpha=0.4, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
        ax.text(gc, gr, 'G2', ha='center', va='center', fontsize=10, fontweight='bold', color='darkblue')

        # Draw trajectory with values and policies
        color = 'red' if agent_idx == 0 else 'blue'

        for t, (r1, c1, r2, c2) in enumerate(trajectory):
            if t >= len(trajectory) - 1:
                break
            if t >= config.horizon:
                break

            s = pos_to_idx(r1, c1, r2, c2)
            my_r, my_c = (r1, c1) if agent_idx == 0 else (r2, c2)

            # Get value and policy at this joint state
            val = values[agent_idx][t][s].item()
            policy = policies[agent_idx][t][s].cpu().numpy()
            best_action = np.argmax(policy)

            # Draw position marker with step number
            circle = plt.Circle((my_c, my_r), 0.2, color=color, alpha=0.7)
            ax.add_patch(circle)
            ax.text(my_c, my_r, str(t), ha='center', va='center',
                   fontsize=7, color='white', fontweight='bold')

            # Draw arrow for best action
            dx, dy = action_arrows[best_action]
            ax.arrow(my_c, my_r, dx * 0.8, dy * 0.8,
                    head_width=0.12, head_length=0.08,
                    fc=color, ec='black', linewidth=0.5, alpha=0.8)

            # Show value and action below/beside the cell
            # Only show every few steps to avoid clutter
            if t % 3 == 0 or t < 5:
                ax.text(my_c + 0.35, my_r + 0.35, f'V={val:.1f}',
                       fontsize=5, color='black', alpha=0.8,
                       path_effects=[pe.withStroke(linewidth=1, foreground='white')])

        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.set_title(f'Agent {agent_idx + 1} Trajectory\n(circles=positions, arrows=greedy actions)')

    plt.suptitle(f'{title}\n(Numbers = timestep, V = value at joint state)', fontsize=12)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved trajectory+values to {save_path}")

    plt.show()


def compare_tau_values(output_dir="results/rqe_solver", horizon=30):
    """Compare RQE values for different tau values."""

    tau_values = [0.1, 0.5, 1.0, 2.0]
    epsilon = 0.1

    print("=" * 70)
    print("Comparing RQE Values for Different Tau Values")
    print("=" * 70)

    fig, axes = plt.subplots(2, len(tau_values), figsize=(5 * len(tau_values), 10))

    for tau_idx, tau in enumerate(tau_values):
        print(f"\nSolving for tau={tau}...")

        config, game, state_info = create_cliff_walk_markov_game(
            horizon=horizon,
            tau=[tau, tau],
            epsilon=[epsilon, epsilon],
            deterministic=True,
        )

        solver = MarkovRQESolver(config)
        policies, values = solver.solve(game)

        height, width = state_info['grid_size']
        cliff_cells = state_info['cliff_cells']
        pos_to_idx = state_info['pos_to_idx']
        agent1_goal = state_info['agent1_goal']
        agent2_goal = state_info['agent2_goal']
        agent1_start = state_info['agent1_start']
        agent2_start = state_info['agent2_start']
        goals = [agent1_goal, agent2_goal]
        other_starts = [agent2_start, agent1_start]

        for agent_idx in range(2):
            ax = axes[agent_idx, tau_idx]
            goal = goals[agent_idx]
            other_start = other_starts[agent_idx]

            # Build value grid
            values_grid = np.zeros((height, width))
            for r in range(height):
                for c in range(width):
                    if agent_idx == 0:
                        s = pos_to_idx(r, c, other_start[0], other_start[1])
                    else:
                        s = pos_to_idx(other_start[0], other_start[1], r, c)
                    values_grid[r, c] = values[agent_idx][0][s].item()

            im = ax.imshow(values_grid, cmap='RdYlGn', origin='upper')
            ax.set_title(f'Agent {agent_idx + 1}, tau={tau}')

            # Add value text
            for r in range(height):
                for c in range(width):
                    if (r, c) not in cliff_cells:
                        val = values_grid[r, c]
                        ax.text(c, r, f'{val:.1f}', ha='center', va='center',
                               fontsize=6, color='white', fontweight='bold',
                               path_effects=[pe.withStroke(linewidth=2, foreground='black')])

            # Mark cliffs
            for (cr, cc) in cliff_cells:
                ax.add_patch(plt.Rectangle((cc - 0.5, cr - 0.5), 1, 1,
                                           fill=True, facecolor='black', edgecolor='white'))

            # Mark goal
            ax.plot(goal[1], goal[0], 'y*', markersize=15, markeredgecolor='black')

            plt.colorbar(im, ax=ax)

    plt.suptitle(f'RQE Solver Values for Different Risk Levels (epsilon={epsilon})', fontsize=14)
    plt.tight_layout()

    save_path = Path(output_dir) / "rqe_solver_tau_comparison.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    print(f"\nSaved comparison to: {save_path}")

    plt.show()


def run_custom(tau, epsilon, horizon=50, output_dir="results/rqe_solver",
               deterministic=True, reward_scale=50.0, enable_collision=False,
               n_samples=100):
    """Run RQE solver with custom parameters and visualize."""

    print("=" * 70)
    print(f"Running Markov RQE Solver")
    print("=" * 70)
    print(f"Parameters: tau={tau}, epsilon={epsilon}, horizon={horizon}")
    print(f"Deterministic: {deterministic}, Reward scale: {reward_scale}")
    print(f"Collision: {enable_collision}, Samples: {n_samples}")

    config, game, state_info = create_cliff_walk_markov_game(
        horizon=horizon,
        tau=tau,
        epsilon=epsilon,
        reward_scale=reward_scale,
        enable_collision=enable_collision,
        n_samples=n_samples,
        deterministic=deterministic,
    )

    print("\nSolving for Markov RQE...")
    solver = MarkovRQESolver(config)
    policies, values = solver.solve(game)
    print("Equilibrium computed!")

    # Get start state values
    start_idx = state_info['pos_to_idx'](*state_info['agent1_start'], *state_info['agent2_start'])
    print(f"\nValues at start state (h=0):")
    print(f"  Player 1: {values[0][0][start_idx].item():.4f}")
    print(f"  Player 2: {values[1][0][start_idx].item():.4f}")

    tau_str = f"{tau[0]}_{tau[1]}" if isinstance(tau, list) else str(tau)
    eps_str = f"{epsilon[0]}_{epsilon[1]}" if isinstance(epsilon, list) else str(epsilon)
    det_str = "det" if deterministic else "stoch"

    # Simulate trajectory first (needed for value visualization)
    print("\nSimulating greedy trajectory...")
    trajectory = simulate_trajectory(policies, state_info, config, max_steps=100, debug=True, enable_collision=enable_collision)

    # Visualize trajectory on grid
    traj_path = Path(output_dir) / f"rqe_trajectory_tau{tau_str}_eps{eps_str}_{det_str}.png"
    visualize_trajectory(
        trajectory, state_info,
        title=f'RQE Trajectory (tau={tau}, eps={epsilon}, {det_str})',
        save_path=str(traj_path)
    )

    # Visualize values along the actual trajectory (main visualization)
    values_path = Path(output_dir) / f"rqe_values_tau{tau_str}_eps{eps_str}_{det_str}.png"
    visualize_values_along_trajectory(
        trajectory, values, state_info, config,
        title=f'RQE Values Along Trajectory (tau={tau}, eps={epsilon}, {det_str})',
        save_path=str(values_path),
        max_steps=12
    )

    # Visualize trajectory with policy arrows and values annotated
    traj_val_path = Path(output_dir) / f"rqe_trajectory_annotated_tau{tau_str}_eps{eps_str}_{det_str}.png"
    visualize_trajectory_with_values(
        trajectory, policies, values, state_info, config,
        title=f'RQE Trajectory with Policies (tau={tau}, eps={epsilon}, {det_str})',
        save_path=str(traj_val_path)
    )

    return policies, values, state_info


def run_paper_scenarios(output_dir="results/rqe_solver"):
    """Run the paper's two scenarios."""

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
    parser = argparse.ArgumentParser(
        description="Run Markov RQE Solver on Cliff Walk Environment"
    )
    parser.add_argument(
        "--tau", type=float, nargs=2, default=None,
        help="Tau values for each agent (e.g., --tau 1.0 1.0)"
    )
    parser.add_argument(
        "--epsilon", type=float, nargs=2, default=None,
        help="Epsilon values for each agent (e.g., --epsilon 0.1 0.1)"
    )
    parser.add_argument(
        "--horizon", type=int, default=50,
        help="Planning horizon (default: 50)"
    )
    parser.add_argument(
        "--visualize_values", action="store_true",
        help="Visualize RQE value functions"
    )
    parser.add_argument(
        "--compare_tau", action="store_true",
        help="Compare RQE values for different tau values"
    )
    parser.add_argument(
        "--paper", action="store_true",
        help="Run paper's two scenarios (Figure 2)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/rqe_solver",
        help="Output directory for visualizations"
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic (less stochastic) dynamics"
    )
    parser.add_argument(
        "--enable_collision", action="store_true",
        help="Enable collision dynamics (agents push each other on collision)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=100,
        help="Number of samples per (state, action) pair for transition estimation (default: 100)"
    )
    parser.add_argument(
        "--reward_scale", type=float, default=1.0,
        help="Reward scaling factor (default: 1.0). Keep small relative to tau for numerical stability."
    )

    args = parser.parse_args()

    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)

    if args.compare_tau:
        # Compare different tau values
        compare_tau_values(output_dir=args.output_dir, horizon=args.horizon)
    elif args.paper:
        # Run paper scenarios
        run_paper_scenarios(output_dir=args.output_dir)
    elif args.tau is not None and args.epsilon is not None:
        # Custom parameters
        run_custom(
            tau=args.tau,
            epsilon=args.epsilon,
            horizon=args.horizon,
            output_dir=args.output_dir,
            deterministic=args.deterministic,
            reward_scale=args.reward_scale,
            enable_collision=args.enable_collision,
            n_samples=args.n_samples,
        )
    else:
        # Default: run paper scenarios
        print("No arguments provided. Running paper scenarios (use --help for options)")
        run_paper_scenarios(output_dir=args.output_dir)
