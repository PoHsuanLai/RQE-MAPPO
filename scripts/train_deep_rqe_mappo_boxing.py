#!/usr/bin/env python3
"""
Train Deep RQE-MAPPO on Atari Boxing

Uses optimized model-agnostic Deep RQE-MAPPO with CNN for visual observations.
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.algorithms.deep_rqe import DeepRQE_MAPPO, DeepRQEConfig
from src.algorithms.cnn_networks import CNNQNetwork, CNNActor
from pettingzoo.atari import boxing_v2


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deep RQE-MAPPO on Atari Boxing")

    # Environment
    parser.add_argument("--render", action="store_true", help="Render environment during training")

    # RQE parameters
    parser.add_argument("--tau", type=float, nargs=2, default=[2.0, 2.0], help="Risk-aversion parameters")
    parser.add_argument("--epsilon", type=float, nargs=2, default=[0.5, 0.5], help="Bounded rationality parameters")

    # Training
    parser.add_argument("--episodes", type=int, default=5000, help="Number of episodes to train")
    parser.add_argument("--lr_actor", type=float, default=1e-4, help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=1e-4, help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    # RQE Solver
    parser.add_argument("--rqe_iterations", type=int, default=3, help="RQE solver iterations")
    parser.add_argument("--rqe_lr", type=float, default=0.5, help="RQE solver learning rate")

    # CNN
    parser.add_argument("--features_dim", type=int, default=512, help="CNN features dimension")

    # Logging
    parser.add_argument("--log_interval", type=int, default=50, help="Log every N episodes")
    parser.add_argument("--save_interval", type=int, default=500, help="Save model every N episodes")
    parser.add_argument("--output_dir", type=str, default="results/deep_rqe_mappo_boxing", help="Output directory")

    return parser.parse_args()


def preprocess_observation(obs):
    """Preprocess Atari observation"""
    obs = obs.astype(np.float32) / 255.0
    obs = np.transpose(obs, (2, 0, 1))
    return obs


def train_deep_rqe_mappo_boxing(args):
    """Train Deep RQE-MAPPO on Boxing"""

    print("=" * 80)
    print("Training Deep RQE-MAPPO on Atari Boxing")
    print("=" * 80)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # Create environment
    env = boxing_v2.parallel_env(render_mode="human" if args.render else None)
    env.reset()

    n_actions = env.action_space("first_0").n
    print(f"\nEnvironment: Atari Boxing")
    print(f"  Agents: 2")
    print(f"  Actions per agent: {n_actions}")
    print(f"  Observation: RGB images [210, 160, 3]")

    # Configuration
    config = DeepRQEConfig(
        n_agents=2,
        action_dims=[n_actions, n_actions],
        tau=args.tau,
        epsilon=args.epsilon,

        # Custom CNN networks
        q_network_class=CNNQNetwork,
        actor_class=CNNActor,
        q_network_kwargs={"input_channels": 3, "features_dim": args.features_dim},
        actor_kwargs={"input_channels": 3, "features_dim": args.features_dim, "action_type": "discrete"},

        # Training
        lr_critic=args.lr_critic,
        lr_actor=args.lr_actor,
        gamma=args.gamma,
        batch_size=args.batch_size,

        # Optimized RQE solver
        rqe_iterations=args.rqe_iterations,
        rqe_lr=args.rqe_lr,
        rqe_momentum=0.9,
    )

    print(f"\nConfiguration:")
    print(f"  Risk-aversion (tau): {config.tau}")
    print(f"  Bounded rationality (epsilon): {config.epsilon}")
    print(f"  Actor LR: {config.lr_actor}, Critic LR: {config.lr_critic}")
    print(f"  RQE Solver: {config.rqe_iterations} iterations, LR {config.rqe_lr}")

    # Create agent
    agent = DeepRQE_MAPPO(config, action_type="discrete")

    # Training tracking
    all_rewards_agent0 = []
    all_rewards_agent1 = []

    print(f"\nTraining for {args.episodes} episodes...")
    print("=" * 80)

    # Training loop
    for episode in range(args.episodes):
        observations, infos = env.reset()
        episode_reward = {agent: 0 for agent in env.possible_agents}

        # Collect trajectory
        trajectory_obs = []
        trajectory_actions = []
        trajectory_rewards = []
        trajectory_next_obs = []
        trajectory_dones = []

        done = False

        while not done:
            agent_names = list(observations.keys())
            if len(agent_names) < 2:
                break

            obs = preprocess_observation(observations[agent_names[0]])

            # Select actions from actor
            actions, log_probs = agent.select_action(obs, deterministic=False)

            # Step environment
            env_actions = {agent_names[0]: actions[0], agent_names[1]: actions[1]}
            next_observations, rewards, terminations, truncations, infos = env.step(env_actions)

            done = any(terminations.values()) or any(truncations.values())

            if not done and len(next_observations) >= 2:
                next_obs = preprocess_observation(next_observations[agent_names[0]])
            else:
                next_obs = obs

            # Store transition
            trajectory_obs.append(obs)
            trajectory_actions.append(actions)
            trajectory_rewards.append([rewards.get(agent_names[0], 0), rewards.get(agent_names[1], 0)])
            trajectory_next_obs.append(next_obs)
            trajectory_dones.append(done)

            for agent_name in agent_names:
                episode_reward[agent_name] += rewards.get(agent_name, 0)

            observations = next_observations

        # Update agent with trajectory
        if len(trajectory_obs) > 0:
            trajectories = {
                'obs': np.array(trajectory_obs),
                'actions': [[trajectory_actions[t][i] for t in range(len(trajectory_actions))] for i in range(2)],
                'rewards': [[trajectory_rewards[t][i] for t in range(len(trajectory_rewards))] for i in range(2)],
                'next_obs': np.array(trajectory_next_obs),
                'dones': np.array(trajectory_dones, dtype=float)
            }
            agent.update(trajectories)

        # Store episode rewards
        agent_names = list(env.possible_agents)
        all_rewards_agent0.append(episode_reward.get(agent_names[0], 0))
        all_rewards_agent1.append(episode_reward.get(agent_names[1], 0))

        # Logging
        if (episode + 1) % args.log_interval == 0:
            avg_reward_0 = np.mean(all_rewards_agent0[-args.log_interval:])
            avg_reward_1 = np.mean(all_rewards_agent1[-args.log_interval:])
            print(f"Episode {episode+1}/{args.episodes} | "
                  f"Avg Reward: [{avg_reward_0:.2f}, {avg_reward_1:.2f}]")

        # Save model
        if (episode + 1) % args.save_interval == 0:
            save_path = os.path.join(run_dir, f"model_ep{episode+1}.pt")
            torch.save({
                'episode': episode,
                'actors': [actor.state_dict() for actor in agent.actors],
                'critics': [critic.state_dict() for critic in agent.critics],
                'config': config,
            }, save_path)
            print(f"  Model saved to {save_path}")

    env.close()

    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    window = 50
    if len(all_rewards_agent0) >= window:
        smoothed_0 = np.convolve(all_rewards_agent0, np.ones(window)/window, mode='valid')
        smoothed_1 = np.convolve(all_rewards_agent1, np.ones(window)/window, mode='valid')
        plt.plot(smoothed_0, label='Agent 0', alpha=0.8)
        plt.plot(smoothed_1, label='Agent 1', alpha=0.8)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward (smoothed)')
    plt.title('Deep RQE-MAPPO on Atari Boxing')
    plt.legend()
    plt.grid(True, alpha=0.3)

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

    final_path = os.path.join(run_dir, 'model_final.pt')
    torch.save({
        'episode': args.episodes,
        'actors': [actor.state_dict() for actor in agent.actors],
        'critics': [critic.state_dict() for critic in agent.critics],
        'config': config,
        'rewards_agent0': all_rewards_agent0,
        'rewards_agent1': all_rewards_agent1,
    }, final_path)
    print(f"Final model saved to: {final_path}")

    print(f"\nFinal Statistics (last 100 episodes):")
    print(f"  Agent 0: {np.mean(all_rewards_agent0[-100:]):.2f} ± {np.std(all_rewards_agent0[-100:]):.2f}")
    print(f"  Agent 1: {np.mean(all_rewards_agent1[-100:]):.2f} ± {np.std(all_rewards_agent1[-100:]):.2f}")

    return agent, all_rewards_agent0, all_rewards_agent1


if __name__ == "__main__":
    args = parse_args()
    agent, rewards_0, rewards_1 = train_deep_rqe_mappo_boxing(args)
