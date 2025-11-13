"""
Train single-agent Distributional RQE-PPO using Stable Baselines3 infrastructure

Uses SB3's highly optimized PPO implementation with our distributional critic
for risk-averse learning. Much faster than our custom implementation!
"""

import gymnasium as gym
import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

from src.algorithms.rqe_ppo_sb3 import RQE_PPO_SB3
from src.algorithms.true_rqe_ppo_sb3 import TrueRQE_PPO_SB3
from src.envs.risky_cartpole import register_risky_envs
from stable_baselines3.common.callbacks import BaseCallback


class EvalCallback(BaseCallback):
    """
    Callback for evaluating and logging during training
    """
    def __init__(self, eval_env, eval_freq=10, n_eval_episodes=10, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.evaluations = []

    def _on_step(self) -> bool:
        # Evaluate every eval_freq rollouts (n_steps timesteps)
        if self.n_calls % (self.eval_freq * self.model.n_steps) == 0:
            iteration = self.n_calls // self.model.n_steps

            # Evaluate
            episode_rewards = []
            episode_lengths = []

            for _ in range(self.n_eval_episodes):
                obs, _ = self.eval_env.reset()
                terminated = False
                truncated = False
                episode_reward = 0
                episode_length = 0

                while not (terminated or truncated):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = self.eval_env.step(action)
                    episode_reward += reward
                    episode_length += 1

                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)

            self.evaluations.append({
                'iteration': iteration,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'mean_length': mean_length,
            })

            if self.verbose > 0:
                print("-" * 60)
                print(f"EVAL | Iteration {iteration} | Reward: {mean_reward:.2f} ± {std_reward:.2f} | Length: {mean_length:.1f}")
                print("-" * 60)

        return True


def train(
    env_name: str = "RiskyCartPole-medium-v0",
    tau: float = 1.0,
    epsilon: float = 0.01,
    risk_measure: str = "entropic",
    total_timesteps: int = 1_000_000,
    n_steps: int = 2048,
    eval_interval: int = 10,
    save_dir: str = "checkpoints/single_agent",
    seed: int = 42,
    use_true_rqe: bool = False,
    normalize_weights: bool = True,
    weight_clip: float = 10.0,
    use_clipping: bool = True
):
    """
    Train single-agent Distributional RQE-PPO using SB3 infrastructure

    Args:
        env_name: Gym environment name
        tau: Risk aversion (lower = more risk-averse)
        epsilon: Bounded rationality (entropy coefficient)
        risk_measure: "entropic", "cvar", "mean_variance"
        total_timesteps: Total training timesteps
        n_steps: Steps per rollout
        eval_interval: Evaluate every N rollouts
        save_dir: Directory to save checkpoints
        seed: Random seed
        use_true_rqe: If True, use true RQE gradient with exponential weights
        normalize_weights: Normalize importance weights (only for true RQE)
        weight_clip: Clip extreme weights (only for true RQE)
        use_clipping: Use PPO clipping (only for true RQE)
    """
    algorithm_name = "True RQE-PPO" if use_true_rqe else "Practical RQE-PPO"

    print("=" * 80)
    print(f"Training Distributional {algorithm_name} (SB3-based, Single-Agent)")
    print("=" * 80)
    print(f"Environment: {env_name}")
    print(f"Risk aversion (tau): {tau}")
    print(f"Bounded rationality (epsilon): {epsilon}")
    print(f"Risk measure: {risk_measure}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Seed: {seed}")
    if use_true_rqe:
        print(f"Using TRUE RQE gradient (exponential weights)")
        print(f"  - Normalize weights: {normalize_weights}")
        print(f"  - Weight clip: {weight_clip}")
        print(f"  - Use PPO clipping: {use_clipping}")
    else:
        print(f"Using PRACTICAL RQE (risk-adjusted GAE)")
    print("=" * 80)

    # Register custom environments
    register_risky_envs()

    # Create environments
    env = gym.make(env_name)
    eval_env = gym.make(env_name)

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Create RQE-PPO agent with SB3 infrastructure
    if use_true_rqe:
        # TRUE RQE with exponential importance weighting
        model = TrueRQE_PPO_SB3(
            "MlpPolicy",
            env,
            tau=tau,
            risk_measure=risk_measure,
            n_atoms=51,
            v_min=0.0,
            v_max=600.0,
            learning_rate=1e-4,  # Lower LR for stability with importance sampling
            n_steps=n_steps,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            normalize_advantage=False,  # We use weights, not advantages
            ent_coef=epsilon,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_clipping=use_clipping,
            normalize_weights=normalize_weights,
            weight_clip=weight_clip,
            verbose=1,
            seed=seed,
        )
    else:
        # PRACTICAL RQE with risk-adjusted GAE
        model = RQE_PPO_SB3(
            "MlpPolicy",
            env,
            tau=tau,
            risk_measure=risk_measure,
            n_atoms=51,
            v_min=0.0,
            v_max=600.0,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            normalize_advantage=True,
            ent_coef=epsilon,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            seed=seed,
        )

    # Create callback for evaluation
    callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=eval_interval,
        n_eval_episodes=10,
        verbose=1,
    )

    print("\nStarting training...")
    print("-" * 80)

    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )

    # Final evaluation
    print("\n" + "=" * 80)
    print("Training complete!")
    print("=" * 80)

    episode_rewards = []
    episode_lengths = []

    for _ in range(20):
        obs, _ = eval_env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        episode_length = 0

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

    print(f"Final evaluation (20 episodes):")
    print(f"  Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"  Length: {np.mean(episode_lengths):.1f}")
    print("=" * 80)

    # Save final model
    suffix = "true" if use_true_rqe else "practical"
    final_path = save_path / f"agent_tau{tau}_{suffix}_sb3.zip"
    model.save(str(final_path))
    print(f"Saved final model: {final_path}")

    # Plot training curve
    if callback.evaluations:
        plt.figure(figsize=(12, 4))

        iterations = [e['iteration'] for e in callback.evaluations]
        mean_rewards = [e['mean_reward'] for e in callback.evaluations]
        std_rewards = [e['std_reward'] for e in callback.evaluations]
        mean_lengths = [e['mean_length'] for e in callback.evaluations]

        plt.subplot(1, 2, 1)
        plt.plot(iterations, mean_rewards, linewidth=2, label='Eval Mean')
        plt.fill_between(
            iterations,
            np.array(mean_rewards) - np.array(std_rewards),
            np.array(mean_rewards) + np.array(std_rewards),
            alpha=0.3,
        )
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.title(f'{algorithm_name} (tau={tau}, ε={epsilon})')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(iterations, mean_lengths, marker='o')
        plt.xlabel('Iteration')
        plt.ylabel('Episode Length')
        plt.title(f'Episode Length (tau={tau})')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = save_path / f"training_curve_tau{tau}_{suffix}_sb3.png"
        plt.savefig(plot_path, dpi=150)
        print(f"Saved training curve: {plot_path}")

    env.close()
    eval_env.close()

    return model, callback


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RQE-PPO agent")

    # Environment
    parser.add_argument("--env", type=str, default="RiskyCartPole-medium-v0",
                        help="Environment name")

    # RQE parameters
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Risk aversion (lower = more risk-averse)")
    parser.add_argument("--epsilon", type=float, default=0.01,
                        help="Bounded rationality (entropy coefficient)")
    parser.add_argument("--risk_measure", type=str, default="entropic",
                        choices=["entropic", "cvar", "mean_variance"],
                        help="Risk measure type")

    # Training parameters
    parser.add_argument("--total_timesteps", type=int, default=None,
                        help="Total training timesteps (overrides n_iterations)")
    parser.add_argument("--n_iterations", type=int, default=500,
                        help="Number of training iterations (default: 500, each is 2048 steps)")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Steps per rollout")
    parser.add_argument("--eval_interval", type=int, default=10,
                        help="Evaluate every N iterations")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--save_dir", type=str, default="checkpoints/single_agent",
                        help="Directory to save checkpoints")

    # Algorithm variant
    parser.add_argument("--use_true_rqe", action="store_true",
                        help="Use TRUE RQE gradient with exponential weights (default: practical GAE-based)")
    parser.add_argument("--normalize_weights", action="store_true", default=True,
                        help="Normalize importance weights (only for true RQE)")
    parser.add_argument("--no_normalize_weights", action="store_false", dest="normalize_weights",
                        help="Don't normalize weights")
    parser.add_argument("--weight_clip", type=float, default=10.0,
                        help="Clip extreme weights (only for true RQE)")
    parser.add_argument("--use_clipping", action="store_true", default=True,
                        help="Use PPO clipping (only for true RQE)")
    parser.add_argument("--no_clipping", action="store_false", dest="use_clipping",
                        help="Don't use PPO clipping")

    args = parser.parse_args()

    # Compute total_timesteps from n_iterations if not specified
    if args.total_timesteps is None:
        args.total_timesteps = args.n_iterations * args.n_steps

    train(
        env_name=args.env,
        tau=args.tau,
        epsilon=args.epsilon,
        risk_measure=args.risk_measure,
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        eval_interval=args.eval_interval,
        save_dir=args.save_dir,
        seed=args.seed,
        use_true_rqe=args.use_true_rqe,
        normalize_weights=args.normalize_weights,
        weight_clip=args.weight_clip,
        use_clipping=args.use_clipping,
    )
