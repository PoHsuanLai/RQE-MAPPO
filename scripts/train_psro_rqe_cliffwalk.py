#!/usr/bin/env python3
"""
Train PSRO with RQE on CliffWalk environment

This implements Policy Space Response Oracles (PSRO) using RQE as the meta-game solver.
The key advantage: RQE meta-game solving is tractable and efficient!
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
    return CliffWalkEnv(grid_size=(6, 6), horizon=200)


def evaluate_meta_strategy(psro: PSRO_RQE, n_episodes: int = 100):
    """
    Evaluate the learned meta-strategy

    Args:
        psro: Trained PSRO instance
        n_episodes: Number of episodes to evaluate

    Returns:
        results: Dictionary with evaluation metrics
    """
    print("\n" + "=" * 80)
    print("Evaluating Meta-Strategy")
    print("=" * 80)

    env = create_env()
    episode_rewards = []
    episode_lengths = []
    successes = 0

    for episode in range(n_episodes):
        # Sample policies from meta-strategy
        policies = [psro.get_policy_from_meta_strategy(i) for i in range(psro.config.n_agents)]

        obs, _ = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done and step < 200:
            # Get actions from sampled policies
            actions = []
            with torch.no_grad():
                for policy in policies:
                    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(psro.device)
                    action_probs = policy(obs_tensor)
                    action = torch.multinomial(action_probs, 1).item()
                    actions.append(action)

            # Step environment
            next_obs, reward, terminated, truncated, _ = env.step(actions)
            done = terminated or truncated

            episode_reward += reward
            obs = next_obs
            step += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(step)

        if episode_reward > 0:
            successes += 1

    # Compute statistics
    results = {
        'avg_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'avg_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': successes / n_episodes,
        'best_reward': np.max(episode_rewards),
        'worst_reward': np.min(episode_rewards),
    }

    print(f"\nResults over {n_episodes} episodes:")
    print(f"  Average reward: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"  Average length: {results['avg_length']:.1f} ± {results['std_length']:.1f}")
    print(f"  Success rate: {successes}/{n_episodes} ({results['success_rate']*100:.1f}%)")
    print(f"  Best reward: {results['best_reward']:.2f}")
    print(f"  Worst reward: {results['worst_reward']:.2f}")

    return results


def main():
    print("=" * 80)
    print("PSRO with RQE on CliffWalk")
    print("=" * 80)
    print()

    # Get environment info
    env = create_env()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.nvec[0]
    print(f"Environment: CliffWalk")
    print(f"Observation dim: {obs_dim}")
    print(f"Action space: {n_actions} (per agent)")
    print()

    # Configure PSRO-RQE
    config = PSRORQEConfig(
        n_agents=2,
        obs_dim=obs_dim,
        action_dims=[n_actions, n_actions],

        # RQE parameters
        tau=[2.0, 2.0],  # Risk aversion
        epsilon=[0.5, 0.5],  # Bounded rationality
        rqe_iterations=20,  # Use optimized sqrt schedule default
        rqe_lr=0.5,
        lr_schedule="sqrt",  # Sqrt schedule finds better equilibria

        # PSRO parameters
        psro_iterations=5,  # Start with 5 iterations
        oracle_episodes=500,  # Train best response for 500 episodes
        eval_episodes=50,  # Evaluate each policy pair for 50 episodes
        initialization_policies=2,  # Start with 2 random policies per agent

        # Oracle configuration (PPO)
        oracle_type="ppo",
        lr_policy=3e-4,
        lr_value=3e-4,
        gamma=0.99,
        ppo_epochs=4,
        ppo_clip=0.2,
        gae_lambda=0.95,

        # Network architecture
        hidden_dims=(128, 128),
        activation="relu",

        # Population management
        max_population_size=10,  # Keep at most 10 policies per agent
    )

    print("Configuration:")
    print(f"  PSRO iterations: {config.psro_iterations}")
    print(f"  Oracle episodes: {config.oracle_episodes}")
    print(f"  Evaluation episodes: {config.eval_episodes}")
    print(f"  RQE solver: {config.rqe_iterations} iterations with {config.lr_schedule} schedule")
    print(f"  Risk aversion (tau): {config.tau}")
    print(f"  Bounded rationality (epsilon): {config.epsilon}")
    print()

    # Create PSRO instance
    psro = PSRO_RQE(config, create_env)

    # Run PSRO
    final_meta_strategy, policy_populations = psro.run()

    # Evaluate final meta-strategy
    results = evaluate_meta_strategy(psro, n_episodes=100)

    print("\n" + "=" * 80)
    print("PSRO Training Complete!")
    print("=" * 80)
    print(f"Final population sizes: {[len(pop) for pop in policy_populations]}")
    print(f"Final meta-strategy:")
    for i, strategy in enumerate(final_meta_strategy):
        print(f"  Agent {i}: {strategy[0].cpu().numpy()}")
    print()
    print("Key insight: PSRO separates learning (RL oracle) from equilibrium solving (RQE).")
    print("This leverages RQE's tractability where it matters most: the meta-game!")
    print("=" * 80)

    return psro, results


if __name__ == "__main__":
    psro, results = main()
