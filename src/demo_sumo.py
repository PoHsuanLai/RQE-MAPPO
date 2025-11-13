"""
Demo trained RQE-MAPPO agent in SUMO-GUI (watch it live!)

Usage:
    python -m src.demo_sumo --exp_name sumo_single-intersection_tau0.5_eps0.01
"""

import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path
import time

from sumo_rl.environment.env import SumoEnvironment
from src.algorithms import RQE_MAPPO, RQEConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Demo SUMO agent with GUI")

    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name to load model from")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Log directory")
    parser.add_argument("--n_episodes", type=int, default=3,
                        help="Number of episodes to run")
    parser.add_argument("--checkpoint", type=str, default="final",
                        help="Checkpoint to load (final, 100000, etc.)")
    parser.add_argument("--delay", type=float, default=0.0,
                        help="Delay between steps (seconds) for slower playback")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set SUMO_HOME
    if 'SUMO_HOME' not in os.environ:
        os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'

    print(f"\n{'='*70}")
    print("🚦 RQE-MAPPO Agent Demo in SUMO-GUI")
    print(f"{'='*70}\n")

    # Load experiment config
    exp_dir = Path(args.log_dir) / args.exp_name
    config_file = exp_dir / "config.json"

    if not config_file.exists():
        print(f"Error: Config not found at {config_file}")
        return

    with open(config_file, 'r') as f:
        config_data = json.load(f)

    print(f"Experiment: {args.exp_name}")
    print(f"Network: {config_data['net']}")
    print(f"Risk Aversion (τ): {config_data['tau']}")
    print(f"Bounded Rationality (ε): {config_data['epsilon']}\n")

    # Load checkpoint
    checkpoint_dir = exp_dir / "checkpoints"
    if args.checkpoint == "final":
        model_path = checkpoint_dir / "model_final.pt"
    else:
        model_path = checkpoint_dir / f"model_{args.checkpoint}.pt"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model: {model_path.name}")

    # Create environment WITH GUI
    import sumo_rl
    sumo_rl_path = Path(sumo_rl.__file__).parent
    nets_path = sumo_rl_path / 'nets'

    net_file = str(nets_path / config_data['net'] / f"{config_data['net']}.net.xml")
    route_file = str(nets_path / config_data['net'] / f"{config_data['net']}.rou.xml")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=True,  # GUI enabled!
        num_seconds=config_data.get('num_seconds', 1000),
        delta_time=config_data.get('delta_time', 5),
    )

    # Get dimensions
    obs = env.reset()
    agents = list(env.ts_ids)
    n_agents = len(agents)
    obs_dim = len(obs[agents[0]])
    action_dim = env.traffic_signals[agents[0]].action_space.n

    print(f"Traffic Signals: {n_agents}")
    print(f"Observation Dim: {obs_dim}")
    print(f"Action Dim: {action_dim}\n")

    # Load agent
    rqe_config = RQEConfig(
        n_agents=n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        tau=config_data['tau'],
        epsilon=config_data['epsilon'],
        hidden_dims=config_data.get('hidden_dims', [256, 256])
    )

    agent = RQE_MAPPO(rqe_config).to(device)
    agent.load(str(model_path))

    print("✅ Model loaded successfully!\n")
    print(f"{'='*70}")
    print("🎬 Starting Demo - Watch the SUMO-GUI window!")
    print(f"{'='*70}\n")
    print("The trained RQE-MAPPO agent is now controlling the traffic lights.")
    print("Watch how it manages traffic to minimize waiting times!\n")

    if args.delay > 0:
        print(f"Running with {args.delay}s delay between steps for observation.\n")

    # Run episodes
    for episode in range(args.n_episodes):
        print(f"\n{'─'*70}")
        print(f"📍 Episode {episode + 1}/{args.n_episodes}")
        print(f"{'─'*70}\n")

        obs_dict = env.reset()
        done = False
        step = 0
        episode_reward = 0
        total_waiting_time = 0

        while not done and step < 200:
            # Get observations as tensor
            obs_list = [obs_dict[agent_id] for agent_id in agents]
            obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)

            # Get action from trained agent
            with torch.no_grad():
                actions, _, _ = agent.get_actions(obs_tensor.unsqueeze(0), deterministic=True)
                actions = actions.squeeze(0)

            # Convert to dict
            actions_np = actions.cpu().numpy().astype(int)
            action_dict = {agent_id: int(actions_np[i]) for i, agent_id in enumerate(agents)}

            # Step environment
            obs_dict, reward_dict, done_dict, info = env.step(action_dict)

            # Stats
            step_reward = sum(reward_dict.values())
            episode_reward += step_reward

            waiting_time = np.mean([sum(env.traffic_signals[ts_id].get_accumulated_waiting_time_per_lane())
                                   for ts_id in agents])
            total_waiting_time += waiting_time

            if step % 20 == 0:
                print(f"  Step {step:3d} | Reward: {step_reward:7.2f} | "
                      f"Avg Waiting: {waiting_time:.1f}s | "
                      f"Action: {list(action_dict.values())}")

            done = all(done_dict.values())

            # Optional delay for observation
            if args.delay > 0:
                time.sleep(args.delay)

            step += 1

        avg_waiting = total_waiting_time / step if step > 0 else 0

        print(f"\n✓ Episode {episode + 1} Complete:")
        print(f"    Steps: {step}")
        print(f"    Total Reward: {episode_reward:.2f}")
        print(f"    Avg Waiting Time: {avg_waiting:.2f}s")

    env.close()

    print(f"\n{'='*70}")
    print("🎉 Demo Complete!")
    print(f"{'='*70}\n")
    print("Tip: Use macOS screen recording (Cmd+Shift+5) to capture a video!")
    print()


if __name__ == "__main__":
    main()
