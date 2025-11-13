"""
Training script for RQE-MAPPO on Traffic scenarios

Usage:
    python -m src.train_traffic --scenario intersection --tau 0.5 --epsilon 0.01
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from src.algorithms import RQE_MAPPO, RQEConfig
from src.envs import TrafficGridEnv
from src.utils import RolloutBuffer, Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train RQE-MAPPO on Traffic")

    # Environment
    parser.add_argument("--scenario", type=str, default="intersection",
                        choices=["intersection", "merge", "passing"],
                        help="Traffic scenario")
    parser.add_argument("--n_vehicles", type=int, default=3,
                        help="Number of vehicles")
    parser.add_argument("--grid_size", type=float, default=20.0,
                        help="Grid size")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Max steps per episode")

    # RQE parameters
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Risk aversion parameter")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Bounded rationality parameter")
    parser.add_argument("--risk_measure", type=str, default="entropic",
                        choices=["entropic", "cvar", "mean_variance"],
                        help="Risk measure")

    # Network
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 128],
                        help="Hidden layer dimensions")

    # Training
    parser.add_argument("--total_timesteps", type=int, default=500000,
                        help="Total training timesteps")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Batch size")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="PPO epochs")
    parser.add_argument("--n_minibatches", type=int, default=4,
                        help="Minibatches")

    # PPO
    parser.add_argument("--lr_actor", type=float, default=3e-4)
    parser.add_argument("--lr_critic", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--clip_param", type=float, default=0.2)

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--eval_interval", type=int, default=25000)
    parser.add_argument("--save_interval", type=int, default=100000)

    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")

    return parser.parse_args()


def train(args):
    """Main training loop"""

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    env = TrafficGridEnv(
        n_vehicles=args.n_vehicles,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        scenario=args.scenario,
        device=device
    )

    # Create config
    config = RQEConfig(
        n_agents=env.n_vehicles,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        tau=args.tau,
        epsilon=args.epsilon,
        risk_measure=args.risk_measure,
        hidden_dims=args.hidden_dims,
        lr_actor=args.lr_actor,
        lr_critic=args.lr_critic,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_param=args.clip_param,
        n_epochs=args.n_epochs,
        n_minibatches=args.n_minibatches
    )

    # Create agent
    agent = RQE_MAPPO(config).to(device)

    # Create buffer
    buffer = RolloutBuffer(
        buffer_size=args.batch_size,
        n_agents=env.n_vehicles,
        obs_dim=env.obs_dim,
        device=device
    )

    # Create logger
    if args.exp_name is None:
        args.exp_name = f"traffic_{args.scenario}_tau{args.tau}_eps{args.epsilon}_seed{args.seed}"

    logger = Logger(
        log_dir=args.log_dir,
        exp_name=args.exp_name
    )

    # Save config
    logger.save_config(vars(args))

    # Training loop
    obs = env.reset()
    episode_reward = torch.zeros(env.n_vehicles, device=device)
    episode_length = 0
    episode_count = 0
    collisions_total = 0
    goals_reached_total = 0

    print("\n" + "="*70)
    print(f"Training RQE-MAPPO on {args.scenario} scenario")
    print("="*70 + "\n")

    for timestep in range(args.total_timesteps):
        # Collect experience
        with torch.no_grad():
            actions, log_probs, entropies = agent.get_actions(obs.unsqueeze(0))
            actions = actions.squeeze(0)
            log_probs = log_probs.squeeze(0)

        # Step
        next_obs, rewards, done, info = env.step(actions)

        # Store
        buffer.add(obs, actions, log_probs, rewards, done, next_obs)

        # Update stats
        episode_reward += rewards
        episode_length += 1

        # Episode end
        if done:
            # Log episode
            logger.log_episode(
                episode=episode_count,
                episode_reward=episode_reward.mean().item(),
                episode_length=episode_length,
                timestep=timestep,
                collisions=info['collisions'],
                goals_reached=info['goals_reached']
            )

            collisions_total += info['collisions']
            goals_reached_total += info['goals_reached']

            # Reset
            obs = env.reset()
            episode_reward = torch.zeros(env.n_vehicles, device=device)
            episode_length = 0
            episode_count += 1

        else:
            obs = next_obs

        # Update policy
        if len(buffer) >= args.batch_size:
            data = buffer.get()

            stats = agent.update(
                observations=data['observations'],
                actions=data['actions'],
                old_log_probs=data['log_probs'],
                rewards=data['rewards'],
                dones=data['dones'],
                next_observations=data['next_observations']
            )

            # Add safety metrics
            stats['total_collisions'] = collisions_total
            stats['total_goals_reached'] = goals_reached_total
            stats['collision_rate'] = collisions_total / max(episode_count, 1)

            logger.log_metrics(stats, timestep, prefix="train/")

            buffer.clear()

        # Evaluation
        if (timestep + 1) % args.eval_interval == 0:
            eval_rewards, eval_collisions, eval_goals = evaluate(agent, env, n_episodes=10)

            logger.log_scalar("eval/mean_reward", eval_rewards.mean(), timestep)
            logger.log_scalar("eval/collision_rate", eval_collisions.mean(), timestep)
            logger.log_scalar("eval/goal_rate", eval_goals.mean(), timestep)

            print(f"\n[Eval @ {timestep+1}]")
            print(f"  Reward: {eval_rewards.mean():.2f} ± {eval_rewards.std():.2f}")
            print(f"  Collisions: {eval_collisions.mean():.2f}")
            print(f"  Goals: {eval_goals.mean():.2f}\n")

        # Save model
        if (timestep + 1) % args.save_interval == 0:
            save_dir = Path(args.log_dir) / args.exp_name / "checkpoints"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"model_{timestep+1}.pt"
            agent.save(str(save_path))
            print(f"Model saved: {save_path}")

    # Final save
    save_dir = Path(args.log_dir) / args.exp_name / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    final_path = save_dir / "model_final.pt"
    agent.save(str(final_path))

    logger.close()

    print("\n" + "="*70)
    print("Training complete!")
    print(f"Total collisions: {collisions_total}")
    print(f"Total goals reached: {goals_reached_total}")
    print("="*70)


def evaluate(agent, env, n_episodes=10):
    """Evaluate agent"""
    episode_rewards = []
    episode_collisions = []
    episode_goals = []

    for _ in range(n_episodes):
        obs = env.reset()
        episode_reward = torch.zeros(env.n_vehicles, device=obs.device)
        done = False

        while not done:
            with torch.no_grad():
                actions, _, _ = agent.get_actions(obs.unsqueeze(0), deterministic=True)
                actions = actions.squeeze(0)

            obs, rewards, done, info = env.step(actions)
            episode_reward += rewards

        episode_rewards.append(episode_reward.mean().item())
        episode_collisions.append(info['collisions'])
        episode_goals.append(info['goals_reached'])

    return np.array(episode_rewards), np.array(episode_collisions), np.array(episode_goals)


if __name__ == "__main__":
    args = parse_args()
    train(args)
