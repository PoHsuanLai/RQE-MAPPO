"""
Quick comparison of Practical RQE vs True RQE on Cliff Walk
"""

import sys
sys.path.insert(0, '/Users/pohsuanlai/Documents/rqe/stable-baselines3')

from src.envs.cliff_walk import CliffWalkEnv
from src.algorithms.rqe_ppo_sb3 import RQE_PPO_SB3
from src.algorithms.true_rqe_ppo_sb3 import TrueRQE_PPO_SB3
import numpy as np


def train_and_evaluate(model_class, model_name, tau, timesteps=30000):
    """Train and evaluate a model"""
    print(f"\n{'='*60}")
    print(f"{model_name} (tau={tau})")
    print(f"{'='*60}")

    env = CliffWalkEnv()

    if model_class == RQE_PPO_SB3:
        model = RQE_PPO_SB3(
            "MlpPolicy",
            env,
            tau=tau,
            risk_measure="entropic",
            n_atoms=51,
            v_min=-10.0,
            v_max=50.0,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=0,
        )
    else:  # TrueRQE
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
            critic_epochs=5,
            gamma=0.99,
            verbose=0,
        )

    print(f"Training for {timesteps} steps...")
    model.learn(total_timesteps=timesteps, progress_bar=True)

    # Evaluate
    print("\nEvaluating...")
    env = CliffWalkEnv()
    n_eval = 20
    rewards = []
    cliffs = 0
    goals = 0

    for _ in range(n_eval):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

            if terminated and reward < 0:
                cliffs += 1
                break
            if info.get('agent1_at_goal') or info.get('agent2_at_goal'):
                goals += 1
                break

        rewards.append(ep_reward)

    print(f"\nResults:")
    print(f"  Mean Reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Cliff Hit Rate: {cliffs/n_eval:.1%}")
    print(f"  Goal Reach Rate: {goals/n_eval:.1%}")

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'cliff_rate': cliffs/n_eval,
        'goal_rate': goals/n_eval
    }


if __name__ == "__main__":
    print("\n" + "="*60)
    print("COMPARING PRACTICAL RQE vs TRUE RQE")
    print("="*60)

    tau = 0.5
    timesteps = 30000

    # Train Practical RQE
    practical_results = train_and_evaluate(
        RQE_PPO_SB3,
        "PRACTICAL RQE",
        tau=tau,
        timesteps=timesteps
    )

    # Train True RQE
    true_results = train_and_evaluate(
        TrueRQE_PPO_SB3,
        "TRUE RQE",
        tau=tau,
        timesteps=timesteps
    )

    # Final comparison
    print(f"\n{'='*60}")
    print("FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"\nPractical RQE:")
    print(f"  Reward: {practical_results['mean_reward']:.2f} ± {practical_results['std_reward']:.2f}")
    print(f"  Cliff Rate: {practical_results['cliff_rate']:.1%}")
    print(f"  Goal Rate: {practical_results['goal_rate']:.1%}")

    print(f"\nTrue RQE:")
    print(f"  Reward: {true_results['mean_reward']:.2f} ± {true_results['std_reward']:.2f}")
    print(f"  Cliff Rate: {true_results['cliff_rate']:.1%}")
    print(f"  Goal Rate: {true_results['goal_rate']:.1%}")

    print(f"\nΔ (True - Practical):")
    print(f"  Reward: {true_results['mean_reward'] - practical_results['mean_reward']:+.2f}")
    print(f"  Cliff Rate: {true_results['cliff_rate'] - practical_results['cliff_rate']:+.1%}")
    print(f"  Goal Rate: {true_results['goal_rate'] - practical_results['goal_rate']:+.1%}")
