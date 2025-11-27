#!/usr/bin/env python3
"""
Quick test of PSRO with RQE on CliffWalk (reduced parameters for fast testing)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from src.algorithms.psro_rqe import PSRO_RQE, PSRORQEConfig
from src.envs.cliff_walk import CliffWalkEnv

import functools
print = functools.partial(print, flush=True)


def create_env():
    """Factory function to create CliffWalk environment"""
    return CliffWalkEnv(grid_size=(4, 4), horizon=100)  # Smaller for quick test


def main():
    print("=" * 80)
    print("Quick Test: PSRO with RQE on CliffWalk")
    print("=" * 80)
    print()

    # Get environment info
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.nvec[0]
    print(f"Environment: CliffWalk (4x4, reduced for testing)")
    print(f"Observation dim: {obs_dim}")
    print(f"Action space: {n_actions}")
    print()

    # Configure PSRO-RQE with reduced parameters
    config = PSRORQEConfig(
        n_agents=2,
        obs_dim=obs_dim,
        action_dims=[n_actions, n_actions],

        # RQE parameters
        tau=[2.0, 2.0],
        epsilon=[0.5, 0.5],
        rqe_iterations=20,
        rqe_lr=0.5,
        lr_schedule="sqrt",

        # PSRO parameters (REDUCED for quick test)
        psro_iterations=2,  # Just 2 iterations
        oracle_episodes=100,  # Train for only 100 episodes
        eval_episodes=20,  # Evaluate with only 20 episodes
        initialization_policies=2,

        # Oracle configuration (PPO)
        oracle_type="ppo",
        lr_policy=3e-4,
        lr_value=3e-4,
        gamma=0.99,
        ppo_epochs=4,
        ppo_clip=0.2,
        gae_lambda=0.95,

        # Network architecture
        hidden_dims=(64, 64),  # Smaller network
        activation="relu",

        # Population management
        max_population_size=5,
    )

    print("Quick Test Configuration:")
    print(f"  PSRO iterations: {config.psro_iterations}")
    print(f"  Oracle episodes: {config.oracle_episodes}")
    print(f"  Evaluation episodes: {config.eval_episodes}")
    print()

    # Create PSRO instance
    psro = PSRO_RQE(config, create_env)

    # Run PSRO
    final_meta_strategy, policy_populations = psro.run()

    print("\n" + "=" * 80)
    print("Quick Test Complete!")
    print("=" * 80)
    print(f"Final population sizes: {[len(pop) for pop in policy_populations]}")
    print(f"Final meta-strategy:")
    for i, strategy in enumerate(final_meta_strategy):
        print(f"  Agent {i}: {strategy[0].cpu().numpy()}")

    # Quick evaluation
    print("\nEvaluating final meta-strategy (10 episodes)...")
    env = create_env()
    successes = 0
    total_reward = 0

    for ep in range(10):
        policies = [psro.get_policy_from_meta_strategy(i) for i in range(2)]
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            actions = []
            with torch.no_grad():
                for policy in policies:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(psro.device)
                    action_probs = policy(obs_tensor)
                    action = torch.multinomial(action_probs, 1).item()
                    actions.append(action)

            next_obs, reward, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated
            ep_reward += reward
            obs = next_obs

        total_reward += ep_reward
        if ep_reward > 0:
            successes += 1

    print(f"Success rate: {successes}/10")
    print(f"Average reward: {total_reward/10:.2f}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
