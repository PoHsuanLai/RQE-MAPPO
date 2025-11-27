#!/usr/bin/env python3
"""
Train True RQE-MAPPO on PettingZoo Simple Spread Environment

Uses the standalone True RQE-MAPPO implementation with self-play.
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import os
from pettingzoo.mpe import simple_spread_v3

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from algorithms.true_rqe_mappo import TrueRQE_MAPPO, TrueRQEConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train True RQE-MAPPO on Simple Spread environment"
    )

    # RQE parameters
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Risk aversion parameter (lower = more risk-averse)"
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Bounded rationality (entropy coefficient)"
    )
    parser.add_argument(
        "--n_atoms",
        type=int,
        default=51,
        help="Number of atoms in distributional critic"
    )
    parser.add_argument(
        "--v_min",
        type=float,
        default=-200.0,
        help="Minimum value for distributional critic support"
    )
    parser.add_argument(
        "--v_max",
        type=float,
        default=100.0,
        help="Maximum value for distributional critic support"
    )

    # Environment parameters
    parser.add_argument(
        "--num_agents",
        type=int,
        default=3,
        help="Number of agents (and landmarks)"
    )
    parser.add_argument(
        "--max_cycles",
        type=int,
        default=25,
        help="Maximum cycles per episode"
    )

    # Training parameters
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1000000,
        help="Total timesteps to train"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size for training"
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=10,
        help="Number of PPO epochs per update"
    )
    parser.add_argument(
        "--actor_lr",
        type=float,
        default=3e-4,
        help="Actor learning rate"
    )
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=3e-4,
        help="Critic learning rate"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor"
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="GAE lambda"
    )

    # Self-play parameters
    parser.add_argument(
        "--use_self_play",
        action="store_true",
        default=True,
        help="Use self-play for training"
    )
    parser.add_argument(
        "--population_size",
        type=int,
        default=5,
        help="Self-play population size"
    )

    # Logging
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log every N updates"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/home/r13921098/True RQE-MAPPO/results/simple_spread",
        help="Directory to save results"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Experiment name"
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=50,
        help="Save checkpoint every N updates"
    )

    return parser.parse_args()


def collect_rollout(env, agents, batch_size, max_cycles):
    """Collect a batch of experience from the environment"""

    # Storage for transitions
    n_agents = len(agents.actors)
    observations = []
    actions = []
    rewards = []
    dones = []
    next_observations = []
    log_probs = []

    timesteps = 0
    total_reward = 0
    episode_lengths = []

    while timesteps < batch_size:
        obs, _ = env.reset()
        episode_length = 0

        for step in range(max_cycles):
            # Convert observations to tensor [1, n_agents, obs_dim]
            obs_list = [obs[f"agent_{i}"] for i in range(n_agents)]
            obs_tensor = torch.FloatTensor(np.array(obs_list)).unsqueeze(0).to(agents.device)  # Add batch dim and move to device

            # Get actions from all agents
            actions_batch, log_probs_batch, _ = agents.get_actions(
                obs_tensor, deterministic=False
            )

            # Convert to environment format
            action_dict = {
                f"agent_{i}": actions_batch[0, i].item()  # Remove batch dim
                for i in range(n_agents)
            }

            # Step environment
            next_obs, reward_dict, done_dict, trunc_dict, _ = env.step(action_dict)

            # Store experience in [n_agents, ...] format
            next_obs_list = [next_obs[f"agent_{i}"] for i in range(n_agents)]
            reward_list = [reward_dict[f"agent_{i}"] for i in range(n_agents)]
            done_any = any(done_dict.values()) or any(trunc_dict.values())

            observations.append(obs_list)
            actions.append([actions_batch[0, i].item() for i in range(n_agents)])
            rewards.append(reward_list)
            dones.append(done_any)
            next_observations.append(next_obs_list)
            log_probs.append([log_probs_batch[0, i].item() for i in range(n_agents)])

            total_reward += np.mean(reward_list)

            obs = next_obs
            timesteps += 1
            episode_length += 1

            if done_any:
                break

        episode_lengths.append(episode_length)

    # Convert to tensors [batch, n_agents, ...]
    obs_tensor = torch.FloatTensor(np.array(observations))  # [batch, n_agents, obs_dim]
    actions_tensor = torch.LongTensor(np.array(actions))  # [batch, n_agents]
    rewards_tensor = torch.FloatTensor(np.array(rewards))  # [batch, n_agents]
    dones_tensor = torch.FloatTensor(np.array(dones))  # [batch]
    next_obs_tensor = torch.FloatTensor(np.array(next_observations))  # [batch, n_agents, obs_dim]
    log_probs_tensor = torch.FloatTensor(np.array(log_probs))  # [batch, n_agents]

    avg_reward = total_reward / len(observations)
    avg_length = np.mean(episode_lengths)

    return obs_tensor, actions_tensor, log_probs_tensor, rewards_tensor, dones_tensor, next_obs_tensor, avg_reward, avg_length


def main():
    args = parse_args()

    # Create environment
    env = simple_spread_v3.parallel_env(
        N=args.num_agents,
        max_cycles=args.max_cycles,
        continuous_actions=False
    )
    env.reset()

    # Get observation and action dimensions
    obs_dim = env.observation_space("agent_0").shape[0]
    action_dim = env.action_space("agent_0").n

    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Number of agents: {args.num_agents}")

    # Create config
    config = TrueRQEConfig(
        n_agents=args.num_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        tau=args.tau,
        epsilon=args.epsilon,
        n_atoms=args.n_atoms,
        v_min=args.v_min,
        v_max=args.v_max,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        use_self_play=args.use_self_play,
        population_size=args.population_size,
    )

    # Create agents
    agents = TrueRQE_MAPPO(config)

    # Experiment name
    exp_name = args.exp_name or f"TrueRQE_MAPPO_SimpleSpread_tau{args.tau}_eps{args.epsilon}"

    # Create checkpoint directory
    checkpoint_dir = os.path.join(args.save_dir, exp_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("=" * 70)
    print(f"Starting True RQE-MAPPO Training on Simple Spread")
    print("=" * 70)
    print(f"Risk aversion (tau): {args.tau}")
    print(f"Bounded rationality (epsilon): {args.epsilon}")
    print(f"Self-play: {args.use_self_play}")
    print(f"Total timesteps: {args.total_timesteps}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    print("=" * 70)

    # Training loop
    total_timesteps = 0
    update = 0
    best_reward = float('-inf')

    while total_timesteps < args.total_timesteps:
        # Collect rollout
        obs, actions, log_probs, rewards, dones, next_obs, avg_reward, avg_length = collect_rollout(
            env, agents, args.batch_size, args.max_cycles
        )

        # Update agents
        metrics = agents.update(obs, actions, log_probs, rewards, dones, next_obs)

        total_timesteps += len(obs)
        update += 1

        # Logging
        if update % args.log_interval == 0:
            print(f"Update {update} | Timesteps {total_timesteps}")
            print(f"  Avg Reward: {avg_reward:.2f}")
            print(f"  Avg Episode Length: {avg_length:.1f}")
            print(f"  Actor Loss: {metrics['actor_loss']:.4f}")
            print(f"  Critic Loss: {metrics['critic_loss']:.4f}")
            print("-" * 70)

        # Save checkpoint
        if update % args.checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{update:06d}.pt")
            checkpoint = {
                'update': update,
                'total_timesteps': total_timesteps,
                'actors': [actor.state_dict() for actor in agents.actors],
                'critics': [critic.state_dict() for critic in agents.critics],
                'actor_optimizers': [opt.state_dict() for opt in agents.actor_optimizers],
                'critic_optimizers': [opt.state_dict() for opt in agents.critic_optimizers],
                'config': config,
                'avg_reward': avg_reward,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

            # Save best checkpoint
            if avg_reward > best_reward:
                best_reward = avg_reward
                best_checkpoint_path = os.path.join(checkpoint_dir, "best_checkpoint.pt")
                torch.save(checkpoint, best_checkpoint_path)
                print(f"Best checkpoint updated: {best_checkpoint_path} (reward: {best_reward:.2f})")

    # Save final checkpoint
    final_checkpoint_path = os.path.join(checkpoint_dir, "final_checkpoint.pt")
    final_checkpoint = {
        'update': update,
        'total_timesteps': total_timesteps,
        'actors': [actor.state_dict() for actor in agents.actors],
        'critics': [critic.state_dict() for critic in agents.critics],
        'actor_optimizers': [opt.state_dict() for opt in agents.actor_optimizers],
        'critic_optimizers': [opt.state_dict() for opt in agents.critic_optimizers],
        'config': config,
        'avg_reward': avg_reward,
    }
    torch.save(final_checkpoint, final_checkpoint_path)

    print("=" * 70)
    print("Training completed!")
    print(f"Final checkpoint saved: {final_checkpoint_path}")
    print(f"Best checkpoint: {os.path.join(checkpoint_dir, 'best_checkpoint.pt')}")
    print(f"Best reward: {best_reward:.2f}")
    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
