"""
Visualize RQE-MAPPO agents in SUMO with GUI

Usage:
    python -m src.visualize_sumo --exp_name YOUR_EXP --use_gui
"""

import argparse
import os
import torch

import sumo_rl
from src.algorithms import RQE_MAPPO, RQEConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize in SUMO")

    parser.add_argument("--exp_name", type=str, default=None,
                        help="Load trained model from experiment")
    parser.add_argument("--use_gui", action="store_true",
                        help="Use SUMO-GUI for visualization")
    parser.add_argument("--n_episodes", type=int, default=1,
                        help="Number of episodes to run")
    parser.add_argument("--scenario", type=str, default="4x4grid",
                        choices=["4x4grid", "2way-single-intersection"],
                        help="SUMO scenario")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set SUMO_HOME if not set
    if 'SUMO_HOME' not in os.environ:
        os.environ['SUMO_HOME'] = '/usr/share/sumo'

    print(f"\n{'='*70}")
    print("RQE-MAPPO Visualization in SUMO")
    print(f"{'='*70}\n")
    print(f"SUMO_HOME: {os.environ.get('SUMO_HOME')}")
    print(f"Using GUI: {args.use_gui}")
    print(f"Scenario: {args.scenario}\n")

    # Create SUMO environment
    env = sumo_rl.parallel_env(
        net_file=f"nets/{args.scenario}/{args.scenario}.net.xml",
        route_file=f"nets/{args.scenario}/{args.scenario}.rou.xml",
        use_gui=args.use_gui,
        num_seconds=1000,
    )

    observations = env.reset()
    agents = env.agents

    print(f"Environment created with {len(agents)} agents")
    print(f"Agents: {agents}\n")

    # If trained model provided, load it
    if args.exp_name:
        print(f"Loading trained model from: {args.exp_name}")
        # TODO: Load trained RQE-MAPPO model
        # For now, use random actions
        use_trained = False
    else:
        use_trained = False
        print("No trained model specified, using random actions\n")

    # Run episodes
    for episode in range(args.n_episodes):
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{args.n_episodes}")
        print(f"{'='*70}\n")

        observations = env.reset()
        done = False
        step = 0
        total_reward = {agent: 0 for agent in agents}

        while not done:
            # Get actions (random for now)
            actions = {agent: env.action_space(agent).sample()
                      for agent in agents if agent in observations}

            # Step environment
            observations, rewards, dones, truncated, infos = env.step(actions)

            # Update rewards
            for agent in rewards:
                total_reward[agent] += rewards[agent]

            # Check if done
            done = all(dones.values()) or all(truncated.values())
            step += 1

            if step % 100 == 0:
                print(f"Step {step}: Total rewards = {sum(total_reward.values()):.2f}")

        print(f"\nEpisode {episode + 1} finished after {step} steps")
        print(f"Total rewards: {total_reward}")
        print(f"Sum: {sum(total_reward.values()):.2f}\n")

    env.close()

    print(f"{'='*70}")
    print("Visualization complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
