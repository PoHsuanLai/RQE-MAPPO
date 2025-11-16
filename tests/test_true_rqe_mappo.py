"""Test True RQE-MAPPO implementation"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.algorithms import TrueRQE_MAPPO, TrueRQEConfig


def test_true_rqe_mappo():
    print("Testing True RQE-MAPPO...")

    config = TrueRQEConfig(
        n_agents=3,
        obs_dim=10,
        action_dim=5,
        tau=0.5,  # Risk-averse
        epsilon=0.01,  # Bounded rationality
        risk_measure="entropic",
        critic_epochs=5  # Extra critic training
    )

    agent = TrueRQE_MAPPO(config)

    # Test get_actions
    obs = torch.randn(32, 3, 10)  # [batch, n_agents, obs_dim]
    actions, log_probs, entropies = agent.get_actions(obs)
    print(f"✓ Actions shape: {actions.shape}")
    print(f"✓ Log probs shape: {log_probs.shape}")
    print(f"✓ Entropies shape: {entropies.shape}")
    assert actions.shape == (32, 3)
    assert log_probs.shape == (32, 3)
    assert entropies.shape == (32, 3)

    # Test get_values (action-conditioned)
    q_values = agent.get_values(obs, actions)
    print(f"✓ Q-values shape: {q_values.shape}")
    assert q_values.shape == (32, 3)

    # Test that Q-values are action-conditioned (different actions → different Q-values)
    actions_alt = torch.randint(0, config.action_dim, (32, 3))
    q_values_alt = agent.get_values(obs, actions_alt)
    print(f"✓ Q-values are action-conditioned: {not torch.allclose(q_values, q_values_alt)}")

    # Test update
    rewards = torch.randn(32, 3)
    dones = torch.zeros(32)
    next_obs = torch.randn(32, 3, 10)

    stats = agent.update(obs, actions, log_probs, rewards, dones, next_obs)
    print(f"✓ Training stats: {stats}")

    # Check stats
    assert 'actor_loss' in stats
    assert 'critic_loss' in stats
    assert 'entropy' in stats
    assert 'approx_kl' in stats
    assert 'clipfrac' in stats

    # Test self-play population
    print(f"✓ Population size: {len(agent.policy_population)}")
    assert len(agent.policy_population) == 0  # Not updated yet (update_population_every=10)

    # Update 10 times to trigger population update
    for _ in range(10):
        agent.update(obs, actions, log_probs, rewards, dones, next_obs)

    print(f"✓ Population size after 10 updates: {len(agent.policy_population)}")
    assert len(agent.policy_population) == 1

    # Test sampling opponent
    opponent = agent.sample_opponent_from_population(agent_id=0)
    print(f"✓ Sampled opponent: {opponent is not None}")

    # Test save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as f:
        path = f.name

    agent.save(path)
    print(f"✓ Saved to {path}")

    # Create new agent and load
    new_agent = TrueRQE_MAPPO(config)
    new_agent.load(path)
    print(f"✓ Loaded from {path}")

    # Test that loaded agent produces same outputs
    new_actions, new_log_probs, _ = new_agent.get_actions(obs)
    new_q_values = new_agent.get_values(obs, actions)

    # Check policy is the same (deterministic output should match)
    det_actions1, _, _ = agent.get_actions(obs, deterministic=True)
    det_actions2, _, _ = new_agent.get_actions(obs, deterministic=True)
    assert torch.allclose(det_actions1.float(), det_actions2.float())
    print(f"✓ Loaded policy matches original")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_true_rqe_mappo()
