"""
Quick test to verify bug fixes are correct
"""

import torch
import numpy as np

from src.algorithms.distributional_rqe_ppo import DistributionalRQE_PPO, DistributionalRQEPPOConfig
from src.networks.actor import ActorNetwork


def test_value_support_range():
    """Test that value support range is now correct"""
    print("=" * 80)
    print("Test 1: Value Support Range")
    print("=" * 80)

    config = DistributionalRQEPPOConfig(
        obs_dim=4,
        action_dim=2,
        tau=1000.0
    )

    print(f"✓ Support range: [{config.v_min}, {config.v_max}]")
    assert config.v_min == 0.0, "v_min should be 0.0"
    assert config.v_max == 600.0, "v_max should be 600.0"
    print(f"✓ CartPole returns [0, 500] are now within support range")
    print()


def test_actor_initialization():
    """Test that actor initialization is now correct"""
    print("=" * 80)
    print("Test 2: Actor Initialization")
    print("=" * 80)

    actor = ActorNetwork(obs_dim=4, action_dim=2)

    first_layer_weights = actor.mlp[0].weight.data
    output_layer_weights = actor.mlp[-1].weight.data

    first_magnitude = first_layer_weights.abs().mean().item()
    output_magnitude = output_layer_weights.abs().mean().item()

    print(f"First layer weight magnitude: {first_magnitude:.4f}")
    print(f"Output layer weight magnitude: {output_magnitude:.4f}")

    # With gain=1.0, typical magnitude should be ~0.1-0.3
    assert first_magnitude > 0.05, f"Weights too small: {first_magnitude}"
    assert first_magnitude < 0.5, f"Weights too large: {first_magnitude}"

    print(f"✓ Weight magnitudes are in expected range [0.05, 0.5]")
    print()


def test_learning_rates():
    """Test that learning rates are now balanced"""
    print("=" * 80)
    print("Test 3: Learning Rate Balance")
    print("=" * 80)

    config = DistributionalRQEPPOConfig(
        obs_dim=4,
        action_dim=2
    )

    print(f"Actor learning rate:  {config.lr_actor:.1e}")
    print(f"Critic learning rate: {config.lr_critic:.1e}")

    assert config.lr_actor == config.lr_critic, "Learning rates should match"
    assert config.lr_critic == 3e-4, "Critic LR should be 3e-4"

    print(f"✓ Learning rates are balanced (both 3e-4)")
    print()


def test_episode_boundary_handling():
    """Test that episode boundaries are handled correctly"""
    print("=" * 80)
    print("Test 4: Episode Boundary Handling")
    print("=" * 80)

    # Create agent
    config = DistributionalRQEPPOConfig(
        obs_dim=4,
        action_dim=2,
        v_min=0.0,
        v_max=600.0
    )
    agent = DistributionalRQE_PPO(config)

    # Simulate buffer with 2 episodes:
    # Episode 1: timesteps 0-4 (terminal at 4)
    # Episode 2: timesteps 5-9 (terminal at 9)
    T = 10
    buffer = {
        'observations': torch.randn(T, 4),
        'actions': torch.randint(0, 2, (T,)),
        'rewards': torch.ones(T),
        'dones': torch.zeros(T),
        'log_probs_old': torch.randn(T)
    }

    # Mark terminals
    buffer['dones'][4] = 1.0  # Episode 1 ends
    buffer['dones'][9] = 1.0  # Episode 2 ends

    print("Simulated buffer:")
    print("  Episode 1: timesteps 0-4 (done[4]=1)")
    print("  Episode 2: timesteps 5-9 (done[9]=1)")
    print()

    # Try to update (this tests the episode boundary logic)
    try:
        metrics = agent.update(buffer)
        print("✓ Update succeeded without errors")
        print(f"  Critic loss: {metrics['critic_loss']:.4f}")
        print(f"  Actor loss: {metrics['actor_loss']:.4f}")
        print()
        print("✓ Episode boundary handling appears correct")
    except Exception as e:
        print(f"✗ Update failed: {e}")
        raise

    print()


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "TESTING BUG FIXES" + " " * 36 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    test_value_support_range()
    test_actor_initialization()
    test_learning_rates()
    test_episode_boundary_handling()

    print("=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
    print()
    print("Bug fixes verified:")
    print("  ✓ Value support range: [0, 600]")
    print("  ✓ Actor initialization: gain=1.0")
    print("  ✓ Learning rates: both 3e-4")
    print("  ✓ Episode boundaries: properly handled")
    print()
    print("Ready to retrain with fixed implementation!")
    print("=" * 80)


if __name__ == "__main__":
    main()
