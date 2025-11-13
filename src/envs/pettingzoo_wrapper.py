"""
Wrapper for PettingZoo environments to work with RQE-MAPPO

Converts PettingZoo API to a format compatible with our training loop
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pettingzoo.utils import parallel_to_aec, aec_to_parallel


class PettingZooWrapper:
    """
    Wrapper for PettingZoo parallel environments

    Handles:
    - Converting observations to tensors
    - Stacking multi-agent observations
    - Handling variable number of agents
    - Episode termination
    """

    def __init__(self, env_fn, device="cpu"):
        """
        Args:
            env_fn: Function that returns a PettingZoo parallel environment
            device: torch device
        """
        self.env = env_fn()
        self.device = device

        # Get agent information
        self.agents = self.env.possible_agents
        self.n_agents = len(self.agents)

        # Get observation and action spaces
        sample_agent = self.agents[0]
        self.obs_shape = self.env.observation_space(sample_agent).shape
        self.obs_dim = int(np.prod(self.obs_shape))

        # Handle discrete and continuous action spaces
        action_space = self.env.action_space(sample_agent)
        if hasattr(action_space, 'n'):
            self.action_dim = action_space.n
            self.action_type = 'discrete'
        else:
            self.action_dim = action_space.shape[0]
            self.action_type = 'continuous'

        print(f"Environment: {self.env.metadata.get('name', 'Unknown')}")
        print(f"  Agents: {self.n_agents}")
        print(f"  Obs dim: {self.obs_dim}")
        print(f"  Action dim: {self.action_dim}")
        print(f"  Action type: {self.action_type}")

    def reset(self) -> torch.Tensor:
        """
        Reset environment

        Returns:
            observations: [n_agents, obs_dim]
        """
        obs_dict, _ = self.env.reset()

        # Stack observations for all agents
        obs_list = []
        for agent in self.agents:
            obs = obs_dict[agent]
            if isinstance(obs, dict):
                # Handle dict observations (e.g., image + vector)
                obs = np.concatenate([v.flatten() for v in obs.values()])
            else:
                obs = obs.flatten()
            obs_list.append(obs)

        observations = torch.FloatTensor(np.stack(obs_list)).to(self.device)
        return observations

    def step(
        self,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Take a step in the environment

        Args:
            actions: [n_agents] for discrete, [n_agents, action_dim] for continuous

        Returns:
            observations: [n_agents, obs_dim]
            rewards: [n_agents]
            dones: [n_agents]
            info: dict
        """
        # Convert actions to dict
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        action_dict = {}
        for i, agent in enumerate(self.agents):
            if self.action_type == 'discrete':
                action_dict[agent] = int(actions[i])
            else:
                action_dict[agent] = actions[i]

        # Step environment
        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = self.env.step(action_dict)

        # Update active agents (they might change after reset)
        active_agents = list(obs_dict.keys())
        if len(active_agents) == 0:
            # Episode ended, return zeros
            observations = torch.zeros(self.n_agents, self.obs_dim).to(self.device)
            rewards = torch.zeros(self.n_agents).to(self.device)
            return observations, rewards, True, info_dict

        # Stack observations
        obs_list = []
        for agent in active_agents:
            obs = obs_dict.get(agent, np.zeros(self.obs_shape))
            if isinstance(obs, dict):
                obs = np.concatenate([v.flatten() for v in obs.values()])
            else:
                obs = obs.flatten()
            obs_list.append(obs)

        observations = torch.FloatTensor(np.stack(obs_list)).to(self.device)

        # Stack rewards
        rewards = torch.FloatTensor([
            reward_dict.get(agent, 0.0) for agent in active_agents
        ]).to(self.device)

        # Check if episode is done (any agent done)
        dones = torch.BoolTensor([
            term_dict.get(agent, False) or trunc_dict.get(agent, False)
            for agent in active_agents
        ]).to(self.device)

        # Episode done if any agent is done
        done = dones.any().item() or len(active_agents) < self.n_agents

        return observations, rewards, done, info_dict

    def close(self):
        """Close environment"""
        self.env.close()


def make_env(env_name: str, **kwargs):
    """
    Factory function to create environments

    Args:
        env_name: Name of the environment
            - "simple_spread": MPE simple spread (cooperative)
            - "simple_adversary": MPE adversary (mixed)
            - "simple_tag": MPE tag (competitive)

    Returns:
        env_fn: Function that creates the environment
    """
    if env_name == "simple_spread":
        from pettingzoo.mpe import simple_spread_v3
        def env_fn():
            return simple_spread_v3.parallel_env(
                N=kwargs.get('n_agents', 3),
                local_ratio=kwargs.get('local_ratio', 0.5),
                max_cycles=kwargs.get('max_cycles', 25),
                continuous_actions=False
            )

    elif env_name == "simple_adversary":
        from pettingzoo.mpe import simple_adversary_v3
        def env_fn():
            return simple_adversary_v3.parallel_env(
                N=kwargs.get('n_good', 2),
                max_cycles=kwargs.get('max_cycles', 25),
                continuous_actions=False
            )

    elif env_name == "simple_tag":
        from pettingzoo.mpe import simple_tag_v3
        def env_fn():
            return simple_tag_v3.parallel_env(
                num_good=kwargs.get('n_good', 1),
                num_adversaries=kwargs.get('n_adversaries', 3),
                num_obstacles=kwargs.get('n_obstacles', 2),
                max_cycles=kwargs.get('max_cycles', 25),
                continuous_actions=False
            )

    elif env_name == "simple_push":
        from pettingzoo.mpe import simple_push_v3
        def env_fn():
            return simple_push_v3.parallel_env(
                max_cycles=kwargs.get('max_cycles', 25),
                continuous_actions=False
            )

    else:
        raise ValueError(f"Unknown environment: {env_name}")

    return env_fn


if __name__ == "__main__":
    # Test the wrapper
    print("Testing PettingZooWrapper...")

    # Test simple_spread (cooperative)
    print("\n=== Testing simple_spread ===")
    env_fn = make_env("simple_spread", n_agents=3)
    env = PettingZooWrapper(env_fn)

    obs = env.reset()
    print(f"Initial obs shape: {obs.shape}")

    # Take random actions
    for step in range(5):
        actions = torch.randint(0, env.action_dim, (env.n_agents,))
        next_obs, rewards, done, info = env.step(actions)

        print(f"Step {step}: rewards={rewards.numpy()}, done={done}")

        if done:
            print("Episode finished!")
            break

    env.close()

    print("\n✓ All tests passed!")
