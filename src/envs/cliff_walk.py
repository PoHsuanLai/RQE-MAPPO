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
        return_joint_reward: bool = True  # Default: sum for RL training
    ):
        super().__init__()

        self.height, self.width = grid_size
        self.horizon = horizon
        self.render_mode = render_mode
        self.return_joint_reward = return_joint_reward

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

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        self.agent1_pos = list(self.agent1_start)
        self.agent2_pos = list(self.agent2_start)
        self.timestep = 0

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
            'agents_close': self._are_agents_close()
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
