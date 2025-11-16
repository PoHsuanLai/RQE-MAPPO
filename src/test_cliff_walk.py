"""
Test Practical RQE vs True RQE on Cliff Walk Environment

This script trains both versions of RQE-PPO on the cliff walking environment
and compares their performance and risk-aversion behavior.
"""

import sys
sys.path.insert(0, '/Users/pohsuanlai/Documents/rqe/stable-baselines3')

import gymnasium as gym
import numpy as np
from src.envs.cliff_walk import CliffWalkEnv, register_cliff_walk
from src.algorithms.rqe_ppo_sb3 import RQE_PPO_SB3
from src.algorithms.true_rqe_ppo_sb3 import TrueRQE_PPO_SB3


def make_cliff_walk_env():
    """Create a single-agent wrapper around the multi-agent cliff walk"""
    # For simplicity, we'll flatten the multi-agent problem into single-agent
    # by having one learnable agent and one scripted agent
    return CliffWalkEnv()


def train_practical_rqe(tau=0.5, total_timesteps=50000):
    """Train Practical RQE-PPO on Cliff Walk"""
    print("\n" + "="*60)
    print("TRAINING PRACTICAL RQE-PPO")
    print(f"tau={tau} (risk aversion)")
    print("="*60)

    env = make_cliff_walk_env()

    model = RQE_PPO_SB3(
        "MlpPolicy",
        env,
        tau=tau,
        risk_measure="entropic",
        n_atoms=51,
        v_min=-10.0,  # Adjusted for cliff walk rewards
        v_max=50.0,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
    )

    print(f"Device: {model.device}")
    print(f"Distributional critic parameters: {sum(p.numel() for p in model.distributional_critic.parameters())}")

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    return model


def train_true_rqe(tau=0.5, total_timesteps=50000):
    """Train True RQE-PPO on Cliff Walk"""
    print("\n" + "="*60)
    print("TRAINING TRUE RQE-PPO")
    print(f"tau={tau} (risk aversion)")
    print("="*60)

    env = make_cliff_walk_env()

    model = TrueRQE_PPO_SB3(
        "MlpPolicy",
        env,
        tau=tau,
        risk_measure="entropic",
        n_atoms=51,
        v_min=-10.0,
        v_max=50.0,
        learning_rate=1e-4,
        critic_learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        critic_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        use_clipping=True,
        normalize_weights=True,
        weight_clip=10.0,
        verbose=1,
    )

    print(f"Device: {model.device}")
    print(f"Distributional critic parameters: {sum(p.numel() for p in model.distributional_critic.parameters())}")

    model.learn(total_timesteps=total_timesteps, progress_bar=True)

    return model


def evaluate_model(model, n_episodes=10, render=False):
    """Evaluate a trained model"""
    env = make_cliff_walk_env()
    if render:
        env = CliffWalkEnv(render_mode='human')

    episode_rewards = []
    episode_lengths = []
    cliff_hits = 0
    goal_reaches = 0

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            done = terminated or truncated

            if render:
                env.render()

            # Track metrics
            if terminated and reward < 0:  # Hit cliff
                cliff_hits += 1
            if info.get('agent1_at_goal') or info.get('agent2_at_goal'):
                goal_reaches += 1
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if render:
            print(f"Episode {episode + 1}: Reward={episode_reward:.1f}, Length={episode_length}")

    results = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'cliff_hit_rate': cliff_hits / n_episodes,
        'goal_reach_rate': goal_reaches / n_episodes,
    }

    return results


def compare_methods():
    """Compare Practical RQE vs True RQE"""
    print("\n" + "="*60)
    print("COMPARING PRACTICAL RQE vs TRUE RQE ON CLIFF WALK")
    print("="*60)

    tau_values = [0.3, 0.5, 1.0]  # Different risk aversion levels
    total_timesteps = 100000

    for tau in tau_values:
        print(f"\n{'='*60}")
        print(f"Testing with tau={tau}")
        print(f"{'='*60}")

        # Train Practical RQE
        practical_model = train_practical_rqe(tau=tau, total_timesteps=total_timesteps)

        print("\nEvaluating Practical RQE...")
        practical_results = evaluate_model(practical_model, n_episodes=20)

        # Train True RQE
        true_model = train_true_rqe(tau=tau, total_timesteps=total_timesteps)

        print("\nEvaluating True RQE...")
        true_results = evaluate_model(true_model, n_episodes=20)

        # Print comparison
        print(f"\n{'='*60}")
        print(f"RESULTS FOR tau={tau}")
        print(f"{'='*60}")
        print(f"\nPractical RQE:")
        print(f"  Mean Reward: {practical_results['mean_reward']:.2f} ± {practical_results['std_reward']:.2f}")
        print(f"  Mean Length: {practical_results['mean_length']:.1f}")
        print(f"  Cliff Hit Rate: {practical_results['cliff_hit_rate']:.1%}")
        print(f"  Goal Reach Rate: {practical_results['goal_reach_rate']:.1%}")

        print(f"\nTrue RQE:")
        print(f"  Mean Reward: {true_results['mean_reward']:.2f} ± {true_results['std_reward']:.2f}")
        print(f"  Mean Length: {true_results['mean_length']:.1f}")
        print(f"  Cliff Hit Rate: {true_results['cliff_hit_rate']:.1%}")
        print(f"  Goal Reach Rate: {true_results['goal_reach_rate']:.1%}")

        print(f"\nDifference:")
        print(f"  Reward: {true_results['mean_reward'] - practical_results['mean_reward']:+.2f}")
        print(f"  Cliff Hit Rate: {true_results['cliff_hit_rate'] - practical_results['cliff_hit_rate']:+.1%}")


def visualize_trained_agent(model, tau, method_name):
    """Visualize a trained agent's behavior"""
    print(f"\n{'='*60}")
    print(f"VISUALIZING {method_name} (tau={tau})")
    print(f"{'='*60}")

    evaluate_model(model, n_episodes=3, render=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test RQE methods on Cliff Walk")
    parser.add_argument('--mode', type=str, default='compare',
                        choices=['compare', 'practical', 'true', 'visualize'],
                        help='Mode to run')
    parser.add_argument('--tau', type=float, default=0.5,
                        help='Risk aversion parameter')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps')

    args = parser.parse_args()

    if args.mode == 'compare':
        compare_methods()
    elif args.mode == 'practical':
        model = train_practical_rqe(tau=args.tau, total_timesteps=args.timesteps)
        results = evaluate_model(model, n_episodes=20)
        print(f"\nResults: {results}")
        visualize_trained_agent(model, args.tau, "Practical RQE")
    elif args.mode == 'true':
        model = train_true_rqe(tau=args.tau, total_timesteps=args.timesteps)
        results = evaluate_model(model, n_episodes=20)
        print(f"\nResults: {results}")
        visualize_trained_agent(model, args.tau, "True RQE")
    elif args.mode == 'visualize':
        # Quick test with small training
        print("Training Practical RQE...")
        practical_model = train_practical_rqe(tau=args.tau, total_timesteps=20000)
        visualize_trained_agent(practical_model, args.tau, "Practical RQE")

        print("\nTraining True RQE...")
        true_model = train_true_rqe(tau=args.tau, total_timesteps=20000)
        visualize_trained_agent(true_model, args.tau, "True RQE")
