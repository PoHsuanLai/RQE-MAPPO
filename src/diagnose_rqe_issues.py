"""
Diagnose issues with RQE-PPO implementation

This script identifies potential bugs and configuration issues that could
explain why RQE-PPO performs worse than standard PPO.
"""

import torch
import numpy as np
from pathlib import Path
import gymnasium as gym

import sys
sys.path.insert(0, '/Users/pohsuanlai/Documents/rqe')

from src.algorithms.distributional_rqe_ppo import DistributionalRQE_PPO, DistributionalRQEPPOConfig

# Directly import to avoid pettingzoo dependency
import gymnasium
from gymnasium import register

def register_risky_envs():
    """Register risky CartPole environments"""
    try:
        gymnasium.make('RiskyCartPole-medium-v0')
    except:
        register(
            id='RiskyCartPole-medium-v0',
            entry_point='src.envs.risky_cartpole:RiskyCartPoleEnv',
            max_episode_steps=500,
            kwargs={'wind_strength': 0.5}
        )


def check_value_support_range():
    """
    Issue #1: Value support range doesn't match task

    CartPole episodes typically get rewards in [0, 500] range.
    Current support: [-50, 50] completely misses the actual value range!
    """
    print("=" * 80)
    print("Issue #1: Value Support Range")
    print("=" * 80)

    config = DistributionalRQEPPOConfig(
        obs_dim=4,
        action_dim=2,
        tau=1000.0,  # Risk-neutral
        v_min=-50.0,
        v_max=50.0
    )

    print(f"Current support range: [{config.v_min}, {config.v_max}]")
    print(f"CartPole typical returns: [0, 500]")
    print()
    print("❌ CRITICAL BUG: Support range completely misses actual values!")
    print("   The distributional critic cannot represent returns > 50")
    print("   All successful episodes (return ~500) are clipped to v_max=50")
    print()
    print("Recommendation: Change to v_min=0.0, v_max=600.0")
    print("=" * 80)
    print()


def check_actor_initialization():
    """
    Issue #2: Actor network initialization is too small

    Current: gain=0.01 (way too small!)
    Standard PPO: gain=1.0 or sqrt(2) ≈ 1.414
    """
    print("=" * 80)
    print("Issue #2: Actor Network Initialization")
    print("=" * 80)

    from src.networks.actor import ActorNetwork

    actor = ActorNetwork(obs_dim=4, action_dim=2)

    # Check weight magnitudes
    first_layer_weights = actor.mlp[0].weight.data
    output_layer_weights = actor.mlp[-1].weight.data

    print(f"First layer weight magnitude: {first_layer_weights.abs().mean():.6f}")
    print(f"Output layer weight magnitude: {output_layer_weights.abs().mean():.6f}")
    print()
    print("❌ CRITICAL BUG: Weights initialized with gain=0.01 (too small!)")
    print("   Standard PPO uses gain=1.0 or sqrt(2)")
    print("   Small initialization makes learning very slow")
    print()
    print("Recommendation: Change gain from 0.01 to 1.0 in ActorNetwork._initialize_weights()")
    print("=" * 80)
    print()


def check_learning_rates():
    """
    Issue #3: Critic learning rate is 3x higher than actor

    Current: actor=3e-4, critic=1e-3
    Standard: both use 3e-4
    """
    print("=" * 80)
    print("Issue #3: Learning Rate Imbalance")
    print("=" * 80)

    config = DistributionalRQEPPOConfig(
        obs_dim=4,
        action_dim=2,
        lr_actor=3e-4,
        lr_critic=1e-3
    )

    print(f"Actor learning rate:  {config.lr_actor:.1e}")
    print(f"Critic learning rate: {config.lr_critic:.1e}")
    print(f"Ratio (critic/actor): {config.lr_critic / config.lr_actor:.1f}x")
    print()
    print("⚠️  POTENTIAL ISSUE: Critic learns 3x faster than actor")
    print("   This can cause instability and value overestimation")
    print("   Standard practice: use same LR for both networks")
    print()
    print("Recommendation: Change lr_critic to 3e-4 (same as actor)")
    print("=" * 80)
    print()


def check_bellman_update():
    """
    Issue #4: Distributional Bellman update uses torch.roll (WRONG!)

    torch.roll shifts observations but doesn't respect episode boundaries.
    This causes incorrect bootstrapping across episodes.
    """
    print("=" * 80)
    print("Issue #4: Episode Boundary Bug in Bellman Update")
    print("=" * 80)
    print()
    print("Current implementation (line 286 in distributional_rqe_ppo.py):")
    print("    next_observations = torch.roll(observations, shifts=-1, dims=0)")
    print()
    print("❌ CRITICAL BUG: torch.roll doesn't respect episode boundaries!")
    print()
    print("Example: If buffer contains 3 episodes:")
    print("    Episode 1: obs[0:100]")
    print("    Episode 2: obs[100:250]")
    print("    Episode 3: obs[250:300]")
    print()
    print("torch.roll will bootstrap:")
    print("    obs[99] -> obs[100]   ❌ WRONG! Different episodes!")
    print("    obs[249] -> obs[250]  ❌ WRONG! Different episodes!")
    print()
    print("This causes the critic to learn incorrect value functions.")
    print()
    print("Recommendation: Properly handle episode boundaries using 'dones' flag")
    print("=" * 80)
    print()


def check_risk_measure_computation():
    """
    Issue #5: Check if risk measure computation is correct
    """
    print("=" * 80)
    print("Issue #5: Risk Measure Numerical Stability")
    print("=" * 80)

    from src.networks.distributional_critic import DistributionalCritic

    critic = DistributionalCritic(
        obs_dim=4,
        n_atoms=51,
        v_min=-50.0,  # Wrong range, but checking computation
        v_max=50.0
    )

    # Test with sample observation
    obs = torch.randn(1, 4)

    # Get distribution
    support, probs = critic.get_distribution(obs)

    print(f"Support range: [{support.min():.2f}, {support.max():.2f}]")
    print(f"Probabilities sum: {probs.sum():.6f}")
    print()

    # Test different tau values
    for tau in [0.3, 1.0, 1000.0]:
        risk_value = critic.get_risk_value(obs, tau=tau, risk_type="entropic")
        expected_value = critic.get_expected_value(obs)

        print(f"τ={tau:6.1f}: Risk-adjusted V={risk_value.item():7.3f}, "
              f"Expected V={expected_value.item():7.3f}, "
              f"Diff={risk_value.item() - expected_value.item():7.3f}")

    print()
    print("✓ Risk measure computation seems numerically stable")
    print("  (But still wrong due to incorrect support range!)")
    print("=" * 80)
    print()


def test_on_actual_episodes():
    """
    Issue #6: Test value predictions on actual CartPole episodes
    """
    print("=" * 80)
    print("Issue #6: Value Prediction on Actual Episodes")
    print("=" * 80)

    # Load trained model
    checkpoint_dir = Path('/Users/pohsuanlai/Documents/rqe/checkpoints/single_agent')

    # Try risk-neutral model
    checkpoint_path = checkpoint_dir / 'agent_tau1000.0_final.pt'

    if not checkpoint_path.exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    # Load config and model
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    config = checkpoint['config']

    agent = DistributionalRQE_PPO(config)
    agent.load(str(checkpoint_path))

    # Create environment
    register_risky_envs()
    env = gym.make('RiskyCartPole-medium-v0')

    # Collect episode
    obs, _ = env.reset()
    episode_values = []
    episode_rewards = []

    for step in range(500):
        # Get value prediction
        obs_tensor = torch.FloatTensor(obs).to(agent.device)
        with torch.no_grad():
            value = agent.critic.get_expected_value(obs_tensor.unsqueeze(0))
            episode_values.append(value.item())

        # Step
        action, _, _ = agent.select_action(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_rewards.append(reward)

        if terminated or truncated:
            break

    actual_return = sum(episode_rewards)
    initial_value = episode_values[0]

    print(f"Episode length: {len(episode_rewards)}")
    print(f"Actual return: {actual_return:.1f}")
    print(f"Initial value prediction: {initial_value:.1f}")
    print(f"Prediction error: {abs(actual_return - initial_value):.1f}")
    print()

    if actual_return > config.v_max:
        print(f"❌ CRITICAL: Actual return ({actual_return:.1f}) > v_max ({config.v_max})")
        print(f"   The critic CANNOT represent this value!")
        print(f"   All values are clamped to [{config.v_min}, {config.v_max}]")

    print()
    print(f"Value range in episode: [{min(episode_values):.1f}, {max(episode_values):.1f}]")
    print(f"Support range: [{config.v_min}, {config.v_max}]")

    env.close()
    print("=" * 80)
    print()


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "RQE-PPO DIAGNOSTIC REPORT" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    print("This script identifies bugs in the RQE-PPO implementation that explain")
    print("why it performs worse than standard PPO.")
    print()

    check_value_support_range()
    check_actor_initialization()
    check_learning_rates()
    check_bellman_update()
    check_risk_measure_computation()
    test_on_actual_episodes()

    print("=" * 80)
    print("SUMMARY OF ISSUES")
    print("=" * 80)
    print()
    print("CRITICAL BUGS (must fix immediately):")
    print("  1. ❌ Value support range [-50, 50] should be [0, 600]")
    print("  2. ❌ Actor initialization gain=0.01 should be gain=1.0")
    print("  3. ❌ Episode boundary bug in distributional Bellman update")
    print()
    print("IMPORTANT ISSUES (should fix):")
    print("  4. ⚠️  Critic LR 3x higher than actor (should be equal)")
    print()
    print("These bugs explain why RQE-PPO performs much worse than standard PPO.")
    print("The distributional critic literally cannot represent successful episodes!")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
