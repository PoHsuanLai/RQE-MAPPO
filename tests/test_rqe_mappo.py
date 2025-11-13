"""Test RQE-MAPPO implementation"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.algorithms import RQE_MAPPO, RQEConfig


def test_rqe_mappo():
    print("Testing RQE-MAPPO...")

    config = RQEConfig(
        n_agents=3,
        obs_dim=10,
        action_dim=5,
        tau=0.5,  # Risk-averse
        epsilon=0.01,  # Bounded rationality
        risk_measure="entropic"
    )

    agent = RQE_MAPPO(config)

    # Test get_actions
    obs = torch.randn(32, 3, 10)  # [batch, n_agents, obs_dim]
    actions, log_probs, entropies = agent.get_actions(obs)
    print(f"✓ Actions shape: {actions.shape}")
    print(f"✓ Log probs shape: {log_probs.shape}")
    print(f"✓ Entropies shape: {entropies.shape}")
    assert actions.shape == (32, 3)
    assert log_probs.shape == (32, 3)
    assert entropies.shape == (32, 3)

    # Test get_values
    values = agent.get_values(obs)
    print(f"✓ Values shape: {values.shape}")
    assert values.shape == (32, 3)

    # Test update
    rewards = torch.randn(32, 3)
    dones = torch.zeros(32)
    next_obs = torch.randn(32, 3, 10)

    stats = agent.update(obs, actions, log_probs, rewards, dones, next_obs)
    print(f"✓ Training stats: {stats}")

    # Check stats have expected keys
    assert 'actor_loss' in stats
    assert 'critic_loss' in stats
    assert 'entropy' in stats
    assert 'approx_kl' in stats
    assert 'clipfrac' in stats

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_rqe_mappo()
