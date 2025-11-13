"""
Training script for RQE-MAPPO

Usage:
    python src/train.py --env simple_spread --tau 0.5 --epsilon 0.01
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from src.algorithms import RQE_MAPPO, RQEConfig
from src.envs import PettingZooWrapper, make_env
from src.utils import RolloutBuffer, Logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train RQE-MAPPO")

    # Environment
    parser.add_argument("--env", type=str, default="simple_spread",
                        choices=["simple_spread", "simple_adversary", "simple_tag", "simple_push"],
                        help="Environment name")
    parser.add_argument("--n_agents", type=int, default=3,
                        help="Number of agents")
    parser.add_argument("--max_cycles", type=int, default=25,
                        help="Max cycles per episode")

    # RQE parameters
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Risk aversion parameter (lower = more risk-averse)")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Bounded rationality parameter (entropy coefficient)")
    parser.add_argument("--risk_measure", type=str, default="entropic",
                        choices=["entropic", "cvar", "mean_variance"],
                        help="Risk measure")

    # Network architecture
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 64],
                        help="Hidden layer dimensions")

    # Training
    parser.add_argument("--total_timesteps", type=int, default=1000000,
                        help="Total training timesteps")
    parser.add_argument("--batch_size", type=int, default=2048,
                        help="Batch size for training")
    parser.add_argument("--n_epochs", type=int, default=10,
                        help="Number of PPO epochs")
    parser.add_argument("--n_minibatches", type=int, default=4,
                        help="Number of minibatches")

    # PPO parameters
    parser.add_argument("--lr_actor", type=float, default=3e-4,
                        help="Actor learning rate")
    parser.add_argument("--lr_critic", type=float, default=1e-3,
                        help="Critic learning rate")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Discount factor")
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help="GAE lambda")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help="PPO clip parameter")

    # Logging
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Log directory")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Experiment name")
    parser.add_argument("--eval_interval", type=int, default=10000,
                        help="Evaluation interval (timesteps)")
    parser.add_argument("--save_interval", type=int, default=50000,
                        help="Save interval (timesteps)")

    # Misc
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (auto, cpu, cuda)")

    return parser.parse_args()


def train(args):
    """Main training loop"""

    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    env_fn = make_env(
        args.env,
        n_agents=args.n_agents,
        max_cycles=args.max_cycles
    )
    env = PettingZooWrapper(env_fn, device=device)

    # Create config
    config = RQEConfig(
        n_agents=env.n_agents,
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
        n_agents=env.n_agents,
        obs_dim=env.obs_dim,
        device=device
    )

    # Create logger
    if args.exp_name is None:
        args.exp_name = f"{args.env}_tau{args.tau}_eps{args.epsilon}_seed{args.seed}"

    logger = Logger(
        log_dir=args.log_dir,
        exp_name=args.exp_name
    )

    # Save config
    logger.save_config(vars(args))

    # Training loop
    obs = env.reset()
    episode_reward = torch.zeros(env.n_agents, device=device)
    episode_length = 0
    episode_count = 0

    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    for timestep in range(args.total_timesteps):
        # Collect experience
        with torch.no_grad():
            actions, log_probs, entropies = agent.get_actions(obs.unsqueeze(0))
            actions = actions.squeeze(0)
            log_probs = log_probs.squeeze(0)

        # Step environment
        next_obs, rewards, done, info = env.step(actions)

        # Store transition
        buffer.add(obs, actions, log_probs, rewards, done, next_obs)

        # Update episode stats
        episode_reward += rewards
        episode_length += 1

        # Check if episode is done
        if done or episode_length >= args.max_cycles:
            # Log episode
            logger.log_episode(
                episode=episode_count,
                episode_reward=episode_reward.mean().item(),
                episode_length=episode_length,
                timestep=timestep
            )

            # Reset episode
            obs = env.reset()
            episode_reward = torch.zeros(env.n_agents, device=device)
            episode_length = 0
            episode_count += 1

        else:
            obs = next_obs

        # Update policy
        if len(buffer) >= args.batch_size:
            # Get all data from buffer
            data = buffer.get()

            # Update agent
            stats = agent.update(
                observations=data['observations'],
                actions=data['actions'],
                old_log_probs=data['log_probs'],
                rewards=data['rewards'],
                dones=data['dones'],
                next_observations=data['next_observations']
            )

            # Log training stats
            logger.log_metrics(stats, timestep, prefix="train/")

            # Clear buffer
            buffer.clear()

        # Evaluation
        if (timestep + 1) % args.eval_interval == 0:
            eval_rewards = evaluate(agent, env, n_episodes=10)
            logger.log_scalar("eval/mean_reward", eval_rewards.mean(), timestep)
            print(f"\n[Eval @ {timestep+1}] Mean reward: {eval_rewards.mean():.2f} ± {eval_rewards.std():.2f}\n")

        # Save model
        if (timestep + 1) % args.save_interval == 0:
            save_dir = Path(args.log_dir) / args.exp_name / "checkpoints"
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / f"model_{timestep+1}.pt"
            agent.save(str(save_path))
            print(f"Model saved to: {save_path}")

    # Final save
    save_dir = Path(args.log_dir) / args.exp_name / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    final_path = save_dir / "model_final.pt"
    agent.save(str(final_path))
    print(f"Final model saved to: {final_path}")

    # Close
    env.close()
    logger.close()

    print("\n" + "="*50)
    print("Training complete!")
    print("="*50)


def evaluate(agent, env, n_episodes=10):
    """Evaluate agent"""
    episode_rewards = []

    for _ in range(n_episodes):
        obs = env.reset()
        episode_reward = torch.zeros(env.n_agents, device=obs.device)
        done = False

        while not done:
            with torch.no_grad():
                actions, _, _ = agent.get_actions(obs.unsqueeze(0), deterministic=True)
                actions = actions.squeeze(0)

            obs, rewards, done, _ = env.step(actions)
            episode_reward += rewards

        episode_rewards.append(episode_reward.mean().item())

    return np.array(episode_rewards)


if __name__ == "__main__":
    args = parse_args()
    train(args)
