"""
Cliff Walk Environment from RQE Paper

A multi-agent grid-world where agents must navigate to their respective goals
while avoiding cliffs. Proximity between agents increases stochasticity.

Based on Figure 2 from "Tractable Multi-Agent Reinforcement Learning
Through Behavioral Economics" (Mazumdar et al., ICLR 2025)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Box, Discrete
from typing import Tuple, Dict, Optional


class CliffWalkEnv(gym.Env):
    """
    Two-agent Cliff Walk environment

    Grid layout:
    - Black cells: Cliff (terminal, reward -2)
    - Colored cells: Agent goals (reward +1)
    - White cells: Normal cells (reward 0 per step)

    Dynamics:
    - Actions: {up, down, left, right}
    - Default: pd = 0.9 (intended action 90% of time)
    - When agents are at least 1 cell apart: pd = 0.5 (high randomness, 50% random)
    - With probability pd: intended action, else random movement

    Args:
        grid_size: Tuple of (height, width)
        horizon: Episode length
        cliff_cells: List of (row, col) cliff positions
        agent1_start: (row, col) for agent 1
        agent2_start: (row, col) for agent 2
        agent1_goal: (row, col) for agent 1's goal
        agent2_goal: (row, col) for agent 2's goal
        return_joint_reward: If True, returns sum of rewards as scalar (for RL training).
                            If False, returns tuple of (reward1, reward2) (per paper spec).
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    # Action mappings
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def __init__(
        self,
        grid_size: Tuple[int, int] = (6, 6),
        horizon: int = 200,
        cliff_cells: Optional[list] = None,
        agent1_start: Tuple[int, int] = (4, 2),  # Below center cliff
        agent2_start: Tuple[int, int] = (1, 2),  # Above center cliff
        agent1_goal: Tuple[int, int] = (0, 0),   # Top-left
        agent2_goal: Tuple[int, int] = (5, 0),   # Bottom-left
        render_mode: Optional[str] = None,
        return_joint_reward: bool = True,  # Default: sum for RL training
        # Reward shaping options
        reward_scale: float = 1.0,  # Scale all rewards by this factor
        corner_reward: float = 0.0,  # One-time reward for reaching safe corners
        agent1_corner: Tuple[int, int] = (0, 5),  # Top-right corner (safe waypoint for agent 1)
        agent2_corner: Tuple[int, int] = (5, 5),  # Bottom-right corner (safe waypoint for agent 2)
        # Collision dynamics
        enable_collision: bool = False,  # Enable collision pushing mechanics
    ):
        super().__init__()

        # Collision dynamics
        self.enable_collision = enable_collision

        self.height, self.width = grid_size
        self.horizon = horizon
        self.render_mode = render_mode
        self.return_joint_reward = return_joint_reward

        # Reward shaping
        self.reward_scale = reward_scale
        self.corner_reward = corner_reward
        self.agent1_corner = agent1_corner
        self.agent2_corner = agent2_corner

        # Default cliff configuration (from paper's Figure 2)
        # 4 blocks between the goals (left column) + 4 blocks in center = 8 total
        if cliff_cells is None:
            cliff_cells = [
                # Left column between goals
                (1, 0), (2, 0), (3, 0), (4, 0),
                # Center 2x2 block
                (2, 2), (2, 3),
                (3, 2), (3, 3)
            ]

        self.cliff_cells = set(cliff_cells)
        self.agent1_start = agent1_start
        self.agent2_start = agent2_start
        self.agent1_goal = agent1_goal
        self.agent2_goal = agent2_goal

        # State: (agent1_row, agent1_col, agent2_row, agent2_col)
        self.observation_space = spaces.Box(
            low=0,
            high=max(self.height, self.width) - 1,
            shape=(4,),
            dtype=np.int32
        )

        # Actions: 4 directions for each agent
        self.action_space = spaces.MultiDiscrete([4, 4])

        # Dynamics parameters (from paper)
        self.pd_close = 0.9  # Probability of intended action when agents close (same cell)
        self.pd_far = 0.5    # Probability of intended action when agents far (>= 1 cell apart)

        # Current state
        self.agent1_pos = None
        self.agent2_pos = None
        self.timestep = 0

        # Track one-time rewards
        self._agent1_got_corner = False
        self._agent2_got_corner = False

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.agent1_pos = list(self.agent1_start)
        self.agent2_pos = list(self.agent2_start)
        self.timestep = 0

        # Reset one-time reward flags
        self._agent1_got_corner = False
        self._agent2_got_corner = False

        obs = np.array(self.agent1_pos + self.agent2_pos, dtype=np.float32)
        info = {}

        return obs, info

    def _get_distance(self, pos1, pos2):
        """Manhattan distance between two positions"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _are_agents_close(self):
        """Check if agents are adjacent (distance < 1, i.e., same cell or touching)"""
        return self._get_distance(self.agent1_pos, self.agent2_pos) < 1

    def _apply_action(self, pos, action, pd):
        """
        Apply action with stochasticity

        With probability pd: move in intended direction
        With probability (1-pd): random movement

        If action would move agent into wall, agent stays in place.
        """
        # Determine actual action (intended or random)
        if self.np_random.random() < pd:
            actual_action = action
        else:
            actual_action = self.np_random.integers(0, 4)

        # Calculate intended new position
        new_pos = pos.copy()
        if actual_action == self.UP:
            new_pos[0] = new_pos[0] - 1
        elif actual_action == self.DOWN:
            new_pos[0] = new_pos[0] + 1
        elif actual_action == self.LEFT:
            new_pos[1] = new_pos[1] - 1
        elif actual_action == self.RIGHT:
            new_pos[1] = new_pos[1] + 1

        # Check if new position is valid (within bounds)
        if 0 <= new_pos[0] < self.height and 0 <= new_pos[1] < self.width:
            return new_pos
        else:
            # Hit wall - stay in place
            return pos.copy()

    def _is_cliff(self, pos):
        """Check if position is a cliff"""
        return tuple(pos) in self.cliff_cells

    def _is_goal(self, pos, agent_id):
        """Check if position is the goal for given agent"""
        if agent_id == 1:
            return tuple(pos) == self.agent1_goal
        else:
            return tuple(pos) == self.agent2_goal

    def _random_push(self, pos):
        """
        Push agent in a random direction.
        Returns new position (may be same if hitting wall).
        Does NOT check for cliffs - that's handled separately.
        """
        push_dir = self.np_random.integers(0, 4)
        new_pos = pos.copy()

        if push_dir == self.UP:
            new_pos[0] = new_pos[0] - 1
        elif push_dir == self.DOWN:
            new_pos[0] = new_pos[0] + 1
        elif push_dir == self.LEFT:
            new_pos[1] = new_pos[1] - 1
        elif push_dir == self.RIGHT:
            new_pos[1] = new_pos[1] + 1

        # Check bounds - if hitting wall, stay in place
        if 0 <= new_pos[0] < self.height and 0 <= new_pos[1] < self.width:
            return new_pos
        else:
            return pos.copy()

    def step(self, action):
        """
        Execute one step

        Args:
            action: (action_agent1, action_agent2)

        Returns:
            obs, reward, terminated, truncated, info
        """
        action1, action2 = action

        # Determine stochasticity based on proximity
        if self._are_agents_close():
            pd = self.pd_close
        else:
            pd = self.pd_far

        # Apply actions
        new_pos1 = self._apply_action(self.agent1_pos, action1, pd)
        new_pos2 = self._apply_action(self.agent2_pos, action2, pd)

        # Collision detection:
        # 1. Both agents end up in same cell
        # 2. Agents swap positions (adjacent agents move toward each other)
        collision_occurred = False
        if self.enable_collision:
            # Case 1: Same cell collision
            same_cell = (new_pos1[0] == new_pos2[0] and new_pos1[1] == new_pos2[1])

            # Case 2: Position swap (agents pass through each other)
            # This happens when agent1 moves to agent2's old position AND vice versa
            swap_collision = (
                new_pos1[0] == self.agent2_pos[0] and new_pos1[1] == self.agent2_pos[1] and
                new_pos2[0] == self.agent1_pos[0] and new_pos2[1] == self.agent1_pos[1]
            )

            if same_cell or swap_collision:
                collision_occurred = True
                # Push both agents in random directions
                new_pos1 = self._random_push(new_pos1)
                new_pos2 = self._random_push(new_pos2)

        # Check if either agent hit cliff (terminal)
        agent1_in_cliff = self._is_cliff(new_pos1)
        agent2_in_cliff = self._is_cliff(new_pos2)

        # If agent hits cliff, they stay stuck
        if not agent1_in_cliff:
            self.agent1_pos = new_pos1
        if not agent2_in_cliff:
            self.agent2_pos = new_pos2

        # Compute rewards (separated per agent as per paper)
        reward1 = 0.0
        reward2 = 0.0

        if agent1_in_cliff:
            reward1 = -2.0
        elif self._is_goal(self.agent1_pos, 1):
            reward1 = 1.0

        if agent2_in_cliff:
            reward2 = -2.0
        elif self._is_goal(self.agent2_pos, 2):
            reward2 = 1.0

        # Add one-time corner rewards (reward shaping)
        corner_reward1 = 0.0
        corner_reward2 = 0.0
        if self.corner_reward > 0:
            if tuple(self.agent1_pos) == self.agent1_corner and not self._agent1_got_corner:
                corner_reward1 = self.corner_reward
                self._agent1_got_corner = True
            if tuple(self.agent2_pos) == self.agent2_corner and not self._agent2_got_corner:
                corner_reward2 = self.corner_reward
                self._agent2_got_corner = True

        # Apply reward scaling and add corner rewards
        reward1 = reward1 * self.reward_scale + corner_reward1
        reward2 = reward2 * self.reward_scale + corner_reward2

        # Return reward based on configuration
        if self.return_joint_reward:
            # Return scalar sum for RL training compatibility
            reward = reward1 + reward2
        else:
            # Return separated rewards as tuple (per paper specification)
            reward = (reward1, reward2)

        # Check termination
        self.timestep += 1
        terminated = agent1_in_cliff or agent2_in_cliff
        truncated = self.timestep >= self.horizon

        obs = np.array(self.agent1_pos + self.agent2_pos, dtype=np.float32)
        info = {
            'agent1_pos': tuple(self.agent1_pos),
            'agent2_pos': tuple(self.agent2_pos),
            'agent1_reward': reward1,
            'agent2_reward': reward2,
            'agent1_at_goal': self._is_goal(self.agent1_pos, 1),
            'agent2_at_goal': self._is_goal(self.agent2_pos, 2),
            'agents_close': self._are_agents_close(),
            'agent1_at_corner': tuple(self.agent1_pos) == self.agent1_corner,
            'agent2_at_corner': tuple(self.agent2_pos) == self.agent2_corner,
            'agent1_got_corner': self._agent1_got_corner,
            'agent2_got_corner': self._agent2_got_corner,
            'collision': collision_occurred,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'human':
            self._render_text()
        elif self.render_mode == 'rgb_array':
            return self._render_rgb()

    def _render_text(self):
        """Text-based rendering"""
        grid = [['.' for _ in range(self.width)] for _ in range(self.height)]

        # Mark cliffs
        for (r, c) in self.cliff_cells:
            grid[r][c] = 'X'

        # Mark goals
        gr, gc = self.agent1_goal
        grid[gr][gc] = 'G1'
        gr, gc = self.agent2_goal
        grid[gr][gc] = 'G2'

        # Mark agents (overwrites goals if on them)
        r1, c1 = self.agent1_pos
        r2, c2 = self.agent2_pos

        if (r1, c1) == (r2, c2):
            grid[r1][c1] = 'AB'  # Both agents
        else:
            grid[r1][c1] = 'A1'
            grid[r2][c2] = 'A2'

        # Print
        print(f"\nTimestep: {self.timestep}/{self.horizon}")
        print("+" + "-" * (self.width * 3 - 1) + "+")
        for row in grid:
            print("|" + " ".join(f"{cell:>2}" for cell in row) + "|")
        print("+" + "-" * (self.width * 3 - 1) + "+")

    def _render_rgb(self):
        """RGB rendering for visualization"""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        fig, ax = plt.subplots(figsize=(8, 8))

        # Draw grid
        for i in range(self.height + 1):
            ax.plot([0, self.width], [i, i], 'k-', linewidth=0.5)
        for j in range(self.width + 1):
            ax.plot([j, j], [0, self.height], 'k-', linewidth=0.5)

        # Draw cliffs (black)
        for (r, c) in self.cliff_cells:
            rect = Rectangle((c, self.height - r - 1), 1, 1, facecolor='black')
            ax.add_patch(rect)

        # Draw goals
        gr, gc = self.agent1_goal
        rect = Rectangle((gc, self.height - gr - 1), 1, 1, facecolor='pink', alpha=0.5)
        ax.add_patch(rect)
        ax.text(gc + 0.5, self.height - gr - 0.5, 'G1', ha='center', va='center', fontsize=12)

        gr, gc = self.agent2_goal
        rect = Rectangle((gc, self.height - gr - 1), 1, 1, facecolor='lightblue', alpha=0.5)
        ax.add_patch(rect)
        ax.text(gc + 0.5, self.height - gr - 0.5, 'G2', ha='center', va='center', fontsize=12)

        # Draw agents
        r1, c1 = self.agent1_pos
        ax.plot(c1 + 0.5, self.height - r1 - 0.5, 'ro', markersize=15, label='Agent 1')

        r2, c2 = self.agent2_pos
        ax.plot(c2 + 0.5, self.height - r2 - 0.5, 'bs', markersize=15, label='Agent 2')

        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect('equal')
        ax.legend()
        ax.set_title(f'Cliff Walk - Timestep {self.timestep}/{self.horizon}')

        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)

        return data


def register_cliff_walk():
    """Register Cliff Walk environment with Gymnasium"""
    gym.register(
        id='CliffWalk-v0',
        entry_point='src.envs.cliff_walk:CliffWalkEnv',
        max_episode_steps=200,
    )


if __name__ == "__main__":
    # Test the environment with separated rewards
    print("Testing Cliff Walk Environment (separated rewards)...")

    env = CliffWalkEnv(render_mode='human', return_joint_reward=False)
    obs, info = env.reset(seed=42)

    print(f"Initial observation: {obs}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    # Run a few random steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        env.render()
        print(f"Action: {action}, Reward: {reward} (agent1={reward[0]:.1f}, agent2={reward[1]:.1f})")
        print(f"Info: {info}")

        if terminated or truncated:
            print("Episode ended!")
            break

    print("\n✓ Environment test (separated rewards) passed!")

    # Test with joint reward
    print("\nTesting Cliff Walk Environment (joint reward for RL)...")
    env2 = CliffWalkEnv(return_joint_reward=True)
    obs, info = env2.reset(seed=42)

    for i in range(3):
        action = env2.action_space.sample()
        obs, reward, terminated, truncated, info = env2.step(action)
        print(f"Action: {action}, Joint Reward: {reward:.1f} (agent1={info['agent1_reward']:.1f}, agent2={info['agent2_reward']:.1f})")

        if terminated or truncated:
            print("Episode ended!")
            break

    print("\n✓ All environment tests passed!")

# =============================================================================
# PettingZoo-compatible Cliff Walk Environment
# =============================================================================

class CliffWalkPettingZoo:
    """
    PettingZoo Parallel API wrapper for CliffWalk environment.

    This allows CliffWalk to work with RLlib's multi-agent framework.
    Each agent gets its own observation (with its position first).
    Each agent gets its own reward (individual, not shared).
    """

    metadata = {"render_modes": ["human"], "name": "cliff_walk_v0"}

    def __init__(
        self,
        grid_size=(6, 6),
        horizon=100,
        reward_scale=50.0,
        corner_reward=0.0,
        deterministic=False,
        enable_collision=False,
    ):
        self.grid_size = grid_size
        self.horizon = horizon
        self.reward_scale = reward_scale
        self.corner_reward = corner_reward
        self.deterministic = deterministic
        self.enable_collision = enable_collision

        # Create underlying environment
        self._env = CliffWalkEnv(
            grid_size=grid_size,
            horizon=horizon,
            return_joint_reward=False,
            reward_scale=reward_scale,
            corner_reward=corner_reward,
            enable_collision=enable_collision,
        )
        if deterministic:
            self._env.pd_close = 0.95
            self._env.pd_far = 0.85

        # PettingZoo required attributes
        self.possible_agents = ["agent_0", "agent_1"]
        self.agents = self.possible_agents.copy()

        # Observation: [my_row, my_col, opp_row, opp_col] normalized to [0, 1]
        self._obs_dim = 4
        self._action_dim = 4

    def observation_space(self, agent):
        return Box(low=0.0, high=1.0, shape=(self._obs_dim,), dtype=np.float32)

    def action_space(self, agent):
        return Discrete(self._action_dim)

    def _get_obs(self, raw_obs):
        """Convert raw observation to per-agent observations."""
        # raw_obs: [agent1_row, agent1_col, agent2_row, agent2_col]
        obs_normalized = raw_obs / (self.grid_size[0] - 1)  # Normalize to [0, 1]

        # Agent 0's view: [my_row, my_col, opp_row, opp_col]
        agent0_obs = obs_normalized.copy()

        # Agent 1's view: [my_row, my_col, opp_row, opp_col] = [a2, a1]
        agent1_obs = np.array([
            obs_normalized[2], obs_normalized[3],
            obs_normalized[0], obs_normalized[1]
        ], dtype=np.float32)

        return {
            "agent_0": agent0_obs,
            "agent_1": agent1_obs,
        }

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents.copy()
        raw_obs, info = self._env.reset(seed=seed)
        observations = self._get_obs(raw_obs)
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions):
        """
        Execute actions for all agents.

        Args:
            actions: Dict mapping agent_id to action

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        # Convert to array format for underlying env
        action_array = np.array([actions["agent_0"], actions["agent_1"]])

        raw_obs, _, terminated, truncated, info = self._env.step(action_array)

        observations = self._get_obs(raw_obs)

        # Individual rewards (not shared!)
        rewards = {
            "agent_0": info["agent1_reward"],
            "agent_1": info["agent2_reward"],
        }

        # Termination applies to all agents
        terminations = {agent: terminated for agent in self.agents}
        terminations["__all__"] = terminated

        truncations = {agent: truncated for agent in self.agents}
        truncations["__all__"] = truncated

        infos = {
            "agent_0": {
                "pos": info["agent1_pos"],
                "at_goal": info["agent1_at_goal"],
            },
            "agent_1": {
                "pos": info["agent2_pos"],
                "at_goal": info["agent2_at_goal"],
            },
        }

        return observations, rewards, terminations, truncations, infos

    def render(self):
        return self._env.render()

    def close(self):
        pass


def env_creator(config):
    """Create CliffWalk environment for RLlib."""
    return CliffWalkPettingZoo(
        grid_size=config.get("grid_size", (6, 6)),
        horizon=config.get("horizon", 100),
        reward_scale=config.get("reward_scale", 50.0),
        corner_reward=config.get("corner_reward", 0.0),
        deterministic=config.get("deterministic", False),
        enable_collision=config.get("enable_collision", False),
    )


# =============================================================================
# Trajectory Simulation and Visualization Utilities
# =============================================================================

def simulate_trajectory(
    env: CliffWalkEnv,
    policy_fn,
    max_steps: int = 100,
    deterministic: bool = True,
    seed: int = None
) -> list:
    """
    Simulate a trajectory using a policy function.

    Args:
        env: CliffWalkEnv instance
        policy_fn: Function that takes (obs, env) and returns (action1, action2)
                   obs is the raw observation array [r1, c1, r2, c2]
        max_steps: Maximum number of steps
        deterministic: If True, use deterministic environment dynamics
        seed: Random seed for environment reset

    Returns:
        trajectory: List of (r1, c1, r2, c2) positions
    """
    # Optionally make env more deterministic for visualization
    original_pd_close = env.pd_close
    original_pd_far = env.pd_far
    if deterministic:
        env.pd_close = 1.0
        env.pd_far = 1.0

    obs, _ = env.reset(seed=seed)

    trajectory = []
    r1, c1 = env.agent1_pos
    r2, c2 = env.agent2_pos
    trajectory.append((r1, c1, r2, c2))

    agent1_goal = env.agent1_goal
    agent2_goal = env.agent2_goal

    for step in range(max_steps):
        # Get actions from policy
        actions = policy_fn(obs, env)
        action1, action2 = actions

        # Step environment
        obs, reward, terminated, truncated, info = env.step((action1, action2))

        # Record new positions
        r1, c1 = env.agent1_pos
        r2, c2 = env.agent2_pos
        trajectory.append((r1, c1, r2, c2))

        # Check if both reached goals
        if (r1, c1) == agent1_goal and (r2, c2) == agent2_goal:
            print(f"Both agents reached goals at step {step+1}!")
            break

        # Early termination if stuck
        if len(trajectory) > 5:
            last_5 = trajectory[-5:]
            agent1_stuck = all(pos[0:2] == last_5[0][0:2] for pos in last_5) and (r1, c1) != agent1_goal
            agent2_stuck = all(pos[2:4] == last_5[0][2:4] for pos in last_5) and (r2, c2) != agent2_goal

            if agent1_stuck and agent2_stuck:
                print(f"Both agents stuck for 5 steps, stopping.")
                break

        if terminated or truncated:
            break

    # Restore original dynamics
    env.pd_close = original_pd_close
    env.pd_far = original_pd_far

    return trajectory


def visualize_trajectory(
    trajectory: list,
    save_path: str = None,
    title: str = "Trajectory",
    grid_size: int = 6,
    cliff_cells: list = None,
    agent1_goal: tuple = (0, 0),
    agent2_goal: tuple = (5, 0),
    show: bool = False
):
    """
    Visualize a trajectory on the grid.

    Args:
        trajectory: List of (r1, c1, r2, c2) positions
        save_path: Path to save the figure (optional)
        title: Plot title
        grid_size: Size of the grid
        cliff_cells: List of cliff cell positions
        agent1_goal: Agent 1's goal position
        agent2_goal: Agent 2's goal position
        show: Whether to display the plot
    """
    import matplotlib
    if save_path and not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if cliff_cells is None:
        cliff_cells = [(1, 0), (2, 0), (3, 0), (4, 0), (2, 2), (2, 3), (3, 2), (3, 3)]

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_aspect('equal')

    # Draw grid
    for i in range(grid_size + 1):
        ax.axhline(i - 0.5, color='gray', linewidth=0.5)
        ax.axvline(i - 0.5, color='gray', linewidth=0.5)

    # Draw cliffs
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

    # Draw trajectories
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
    ax.set_title(f'{title}\n({len(trajectory)} steps)')
    ax.legend(loc='upper right')

    # Invert y-axis so row 0 is at top
    ax.invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Trajectory saved to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def get_normalized_obs(env: CliffWalkEnv, agent_id: int) -> np.ndarray:
    """
    Get normalized observation for an agent.

    Args:
        env: CliffWalkEnv instance
        agent_id: 0 for agent 1, 1 for agent 2

    Returns:
        Normalized observation array [my_r, my_c, opp_r, opp_c] in [0, 1]
    """
    r1, c1 = env.agent1_pos
    r2, c2 = env.agent2_pos
    h, w = env.height, env.width

    # Normalize to [0, 1]
    norm_r1, norm_c1 = r1 / (h - 1), c1 / (w - 1)
    norm_r2, norm_c2 = r2 / (h - 1), c2 / (w - 1)

    if agent_id == 0:
        return np.array([norm_r1, norm_c1, norm_r2, norm_c2], dtype=np.float32)
    else:
        return np.array([norm_r2, norm_c2, norm_r1, norm_c1], dtype=np.float32)
