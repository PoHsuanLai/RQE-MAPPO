#!/usr/bin/env python3
"""
Train Deep RQE Q-Learning on Atari Boxing

Uses optimized model-agnostic Deep RQE with CNN for visual observations.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.deep_rqe import DeepRQE_QLearning, DeepRQEConfig
from src.algorithms.cnn_networks import CNNQNetwork
from pettingzoo.atari import boxing_v2


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep RQE Q-Learning on Atari Boxing")

    # Environment
    parser.add_argument("--render", action="store_true", help="Render environment during training")

    # RQE parameters
    parser.add_argument("--tau", type=float, nargs=2, default=[2.0, 2.0], help="Risk-aversion parameters")
    parser.add_argument("--epsilon", type=float, nargs=2, default=[0.5, 0.5], help="Bounded rationality parameters")

    # Training
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=100000, help="Replay buffer size")
    parser.add_argument("--update_freq", type=int, default=4, help="Update every N steps")
    parser.add_argument("--target_update_freq", type=int, default=1000, help="Target network update frequency (steps)")

    # RQE Solver
    parser.add_argument("--rqe_iterations", type=int, default=3, help="RQE solver iterations")
    parser.add_argument("--rqe_lr", type=float, default=0.5, help="RQE solver learning rate")

    # CNN
    parser.add_argument("--features_dim", type=int, default=512, help="CNN features dimension")

    # Exploration
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting epsilon for exploration")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Final epsilon")
    parser.add_argument("--epsilon_decay", type=int, default=50000, help="Epsilon decay steps")

    # Logging
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N episodes")
    parser.add_argument("--save_interval", type=int, default=100, help="Save model every N episodes")
    parser.add_argument("--output_dir", type=str, default="results/deep_rqe_boxing", help="Output directory")

    return parser.parse_args()


def preprocess_observation(obs):
    """
    Preprocess Atari observation

    Args:
        obs: [210, 160, 3] RGB image

    Returns:
        [3, 210, 160] normalized tensor
    """
    # Convert to float and normalize
    obs = obs.astype(np.float32) / 255.0

    # Transpose to [C, H, W]
    obs = np.transpose(obs, (2, 0, 1))

    return obs


def train_deep_rqe_boxing(args):
    """Train Deep RQE Q-Learning on Boxing"""

    print("=" * 80)
    print("Training Deep RQE Q-Learning on Atari Boxing")
    print("=" * 80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Create environment
    env = boxing_v2.parallel_env(render_mode="human" if args.render else None)
    env.reset()

    # Get action space size (Atari has 18 actions)
    n_actions = env.action_space("first_0").n
    print(f"\nEnvironment: Atari Boxing")
    print(f"  Agents: 2")
    print(f"  Actions per agent: {n_actions}")
    print(f"  Observation: RGB images [210, 160, 3]")

    # Configuration
    config = DeepRQEConfig(
        n_agents=2,
        action_dims=[n_actions, n_actions],

        # RQE parameters
        tau=args.tau,
        epsilon=args.epsilon,

        # Custom CNN Q-network
        q_network_class=CNNQNetwork,
        q_network_kwargs={
            "input_channels": 3,
            "features_dim": args.features_dim,
        },

        # Training
        lr_critic=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        update_frequency=args.update_freq,

        # Warmup: skip RQE solver during initial exploration (speeds up training significantly)
        warmup_steps=10000,  # Skip solver for first 10k steps

        # Optimized RQE solver
        rqe_iterations=args.rqe_iterations,
        rqe_lr=args.rqe_lr,
        rqe_momentum=0.9,
    )

    print(f"\nConfiguration:")
    print(f"  Risk-aversion (tau): {config.tau}")
    print(f"  Bounded rationality (epsilon): {config.epsilon}")
    print(f"  Tractability: ε1*ε2 = {config.epsilon[0]*config.epsilon[1]:.3f}, 1/(τ1*τ2) = {1/(config.tau[0]*config.tau[1]):.3f}")
    print(f"  Learning rate: {config.lr_critic}")
    print(f"  Gamma: {config.gamma}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Buffer size: {config.buffer_size}")
    print(f"\n  RQE Solver (Optimized):")
    print(f"    • Iterations: {config.rqe_iterations}")
    print(f"    • Learning rate: {config.rqe_lr}")
    print(f"    • Warm start: Enabled")
    print(f"    • Update frequency: Every {config.update_frequency} steps")
    print(f"    • GPU-only: Yes")
    print(f"\n  CNN Architecture:")
    print(f"    • Features dim: {args.features_dim}")
    print(f"    • Nature DQN architecture")

    # Create agent
    agent = DeepRQE_QLearning(config)

    # Training tracking
    episode_rewards = {agent: [] for agent in env.possible_agents}
    all_rewards_agent0 = []
    all_rewards_agent1 = []
    total_steps = 0

    print(f"\nTraining for {args.episodes} episodes...")
    print("=" * 80)

    # Training loop
    for episode in tqdm(range(args.episodes), desc="Training"):
        observations, infos = env.reset()
        episode_reward = {agent: 0 for agent in env.possible_agents}
        done = False
        step = 0

        # Epsilon decay for exploration
        epsilon_greedy = args.epsilon_end + (args.epsilon_start - args.epsilon_end) * \
                        np.exp(-total_steps / args.epsilon_decay)

        while not done:
            # Get observations for both agents
            agent_names = list(observations.keys())
            if len(agent_names) < 2:
                break

            obs_0 = preprocess_observation(observations[agent_names[0]])
            obs_1 = preprocess_observation(observations[agent_names[1]])

            # For simplicity, use agent 0's observation (both agents see similar view)
            # In a more sophisticated version, you could use both observations
            obs = obs_0

            # Select actions
            actions = agent.select_action(obs, epsilon_greedy=epsilon_greedy)

            # Step environment
            env_actions = {agent_names[0]: actions[0], agent_names[1]: actions[1]}
            next_observations, rewards, terminations, truncations, infos = env.step(env_actions)

            # Check if episode done
            done = any(terminations.values()) or any(truncations.values())

            # Prepare next observation
            if not done and len(next_observations) >= 2:
                next_obs = preprocess_observation(next_observations[agent_names[0]])
            else:
                next_obs = obs  # Use current obs if episode ended

            # Update agent
            agent_rewards = [rewards.get(agent_names[0], 0), rewards.get(agent_names[1], 0)]
            agent.update(obs, actions, agent_rewards, next_obs, done)

            # Update target networks
            if total_steps % args.target_update_freq == 0:
                agent.update_target_networks(tau=0.005)

            # Track rewards
            for agent_name in agent_names:
                episode_reward[agent_name] += rewards.get(agent_name, 0)

            observations = next_observations
            step += 1
            total_steps += 1

        # Store episode rewards
        agent_names = list(env.possible_agents)
        all_rewards_agent0.append(episode_reward.get(agent_names[0], 0))
        all_rewards_agent1.append(episode_reward.get(agent_names[1], 0))

        # Logging
        if (episode + 1) % args.log_interval == 0:
            avg_reward_0 = np.mean(all_rewards_agent0[-args.log_interval:])
            avg_reward_1 = np.mean(all_rewards_agent1[-args.log_interval:])
            print(f"Episode {episode+1}/{args.episodes} | "
                  f"Avg Reward: [{avg_reward_0:.2f}, {avg_reward_1:.2f}] | "
                  f"Buffer: {len(agent.buffer)} | "
                  f"Steps: {total_steps} | "
                  f"Epsilon: {epsilon_greedy:.3f}")

        # Save model
        if (episode + 1) % args.save_interval == 0:
            save_path = os.path.join(run_dir, f"model_ep{episode+1}.pt")
            torch.save({
                'episode': episode,
                'q_networks': [net.state_dict() for net in agent.q_networks],
                'optimizers': [opt.state_dict() for opt in agent.optimizers],
                'config': config,
            }, save_path)
            print(f"  Model saved to {save_path}")

    env.close()

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    # Plot results
    print("\nPlotting results...")
    plt.figure(figsize=(12, 5))

    # Plot episode rewards
    plt.subplot(1, 2, 1)
    window = 50
    if len(all_rewards_agent0) >= window:
        smoothed_0 = np.convolve(all_rewards_agent0, np.ones(window)/window, mode='valid')
        smoothed_1 = np.convolve(all_rewards_agent1, np.ones(window)/window, mode='valid')
        plt.plot(smoothed_0, label='Agent 0', alpha=0.8)
        plt.plot(smoothed_1, label='Agent 1', alpha=0.8)
    else:
        plt.plot(all_rewards_agent0, label='Agent 0', alpha=0.5)
        plt.plot(all_rewards_agent1, label='Agent 1', alpha=0.5)

    plt.xlabel('Episode')
    plt.ylabel('Episode Reward (smoothed)')
    plt.title('Deep RQE Q-Learning on Atari Boxing')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot reward distribution
    plt.subplot(1, 2, 2)
    last_episodes = min(100, len(all_rewards_agent0))
    plt.scatter(all_rewards_agent0[-last_episodes:], all_rewards_agent1[-last_episodes:], alpha=0.5)
    plt.xlabel('Agent 0 Reward')
    plt.ylabel('Agent 1 Reward')
    plt.title(f'Reward Distribution (Last {last_episodes} Episodes)')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(run_dir, 'training_results.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {plot_path}")

    # Save final model
    final_path = os.path.join(run_dir, 'model_final.pt')
    torch.save({
        'episode': args.episodes,
        'q_networks': [net.state_dict() for net in agent.q_networks],
        'optimizers': [opt.state_dict() for opt in agent.optimizers],
        'config': config,
        'rewards_agent0': all_rewards_agent0,
        'rewards_agent1': all_rewards_agent1,
    }, final_path)
    print(f"Final model saved to: {final_path}")

    # Print final statistics
    print(f"\nFinal Statistics (last 100 episodes):")
    print(f"  Agent 0: {np.mean(all_rewards_agent0[-100:]):.2f} ± {np.std(all_rewards_agent0[-100:]):.2f}")
    print(f"  Agent 1: {np.mean(all_rewards_agent1[-100:]):.2f} ± {np.std(all_rewards_agent1[-100:]):.2f}")

    return agent, all_rewards_agent0, all_rewards_agent1


if __name__ == "__main__":
    args = parse_args()
    agent, rewards_0, rewards_1 = train_deep_rqe_boxing(args)
