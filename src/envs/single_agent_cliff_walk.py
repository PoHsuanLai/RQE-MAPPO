"""
Single-Agent Cliff Walk (Agent 2 uses scripted policy)

Simplifies the multi-agent cliff walk to single-agent for easier comparison
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.envs.cliff_walk import CliffWalkEnv


class SingleAgentCliffWalk(gym.Wrapper):
    """
    Wrapper that makes Cliff Walk single-agent by using a scripted policy for Agent 2

    Agent 2 strategy: Move right towards goal, avoiding cliffs
    """

    def __init__(self, env=None):
        if env is None:
            env = CliffWalkEnv()
        super().__init__(env)

        # Override action space to be single discrete (only control agent 1)
        self.action_space = spaces.Discrete(4)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs, info

    def _get_agent2_action(self, agent2_pos):
        """Simple scripted policy for agent 2: move towards goal"""
        goal_row, goal_col = self.env.agent2_goal
        row, col = agent2_pos

        # Avoid cliffs (columns 2-3)
        if col < 2:
            return self.env.RIGHT  # Move right to get past cliffs
        elif col == 2 or col == 3:
            # In cliff zone - try to go around
            if row < 3:
                return self.env.UP  # Go up to avoid
            else:
                return self.env.DOWN  # Go down to avoid
        else:
            # Past cliffs - head to goal
            if row < goal_row:
                return self.env.DOWN
            elif row > goal_row:
                return self.env.UP
            elif col < goal_col:
                return self.env.RIGHT
            else:
                return self.env.LEFT  # Should not happen if at goal

    def step(self, action_agent1):
        """
        Step with single action (for agent 1), agent 2 uses scripted policy
        """
        # Get scripted action for agent 2
        action_agent2 = self._get_agent2_action(self.env.agent2_pos)

        # Combine actions
        action = [action_agent1, action_agent2]

        # Execute in underlying environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        return obs, reward, terminated, truncated, info


if __name__ == "__main__":
    print("Testing Single-Agent Cliff Walk...")

    env = SingleAgentCliffWalk()
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")

    obs, info = env.reset(seed=42)
    print(f"Initial obs: {obs}")

    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            print(f"Episode ended at step {i+1}")
            break

    print("\n✓ Single-agent environment test passed!")
