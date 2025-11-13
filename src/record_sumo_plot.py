"""
Record trained RQE-MAPPO agent in SUMO as animated plot (no GUI needed)

Usage:
    python -m src.record_sumo_plot --exp_name sumo_single-intersection_tau0.5_eps0.01
"""

import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, Circle
from PIL import Image

from sumo_rl.environment.env import SumoEnvironment
from src.algorithms import RQE_MAPPO, RQEConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Record SUMO agent as animated plot")

    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name to load model from")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Log directory")
    parser.add_argument("--output", type=str, default="agent.gif",
                        help="Output file (*.gif)")
    parser.add_argument("--n_steps", type=int, default=200,
                        help="Number of steps to record")
    parser.add_argument("--fps", type=int, default=5,
                        help="Frames per second")
    parser.add_argument("--checkpoint", type=str, default="final",
                        help="Checkpoint to load (final, 100000, etc.)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set SUMO_HOME
    if 'SUMO_HOME' not in os.environ:
        os.environ['SUMO_HOME'] = '/opt/homebrew/opt/sumo/share/sumo'

    print(f"\n{'='*70}")
    print("Recording RQE-MAPPO Agent in SUMO (Plot Animation)")
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
    print(f"Tau: {config_data['tau']}")
    print(f"Epsilon: {config_data['epsilon']}\n")

    # Load checkpoint
    checkpoint_dir = exp_dir / "checkpoints"
    if args.checkpoint == "final":
        model_path = checkpoint_dir / "model_final.pt"
    else:
        model_path = checkpoint_dir / f"model_{args.checkpoint}.pt"

    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        return

    print(f"Loading model: {model_path}")

    # Create environment WITHOUT GUI
    import sumo_rl
    sumo_rl_path = Path(sumo_rl.__file__).parent
    nets_path = sumo_rl_path / 'nets'

    net_file = str(nets_path / config_data['net'] / f"{config_data['net']}.net.xml")
    route_file = str(nets_path / config_data['net'] / f"{config_data['net']}.rou.xml")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create SUMO environment WITHOUT GUI (headless)
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=False,  # No GUI needed!
        num_seconds=config_data.get('num_seconds', 1000),
        delta_time=config_data.get('delta_time', 5),
    )

    # Get dimensions
    obs = env.reset()
    agents = list(env.ts_ids)
    n_agents = len(agents)
    obs_dim = len(obs[agents[0]])
    action_dim = env.traffic_signals[agents[0]].action_space.n

    print(f"Agents: {n_agents}")
    print(f"Obs dim: {obs_dim}")
    print(f"Action dim: {action_dim}\n")

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

    print("Model loaded successfully!\n")
    print(f"Recording {args.n_steps} steps...")
    print(f"Output: {args.output}\n")

    # Collect episode data
    obs_dict = env.reset()

    history = {
        'waiting_times': [],
        'rewards': [],
        'actions': [],
        'queue_lengths': [],
        'steps': []
    }

    print("Running simulation...")
    for step in range(args.n_steps):
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

        # Store metrics
        history['steps'].append(step)
        history['waiting_times'].append(
            np.mean([sum(env.traffic_signals[ts_id].get_accumulated_waiting_time_per_lane())
                    for ts_id in agents])
        )
        history['rewards'].append(sum(reward_dict.values()))
        history['actions'].append(list(action_dict.values()))
        history['queue_lengths'].append(
            np.mean([sum(env.traffic_signals[ts_id].get_lanes_queue())
                    for ts_id in agents])
        )

        if step % 20 == 0:
            print(f"  Step {step}/{args.n_steps}")

        if all(done_dict.values()):
            break

    env.close()

    print(f"\nSimulation complete! Recorded {len(history['steps'])} steps")
    print("\nGenerating animation...")

    # Create animation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'RQE-MAPPO Agent (τ={config_data["tau"]}, ε={config_data["epsilon"]})',
                 fontsize=14, fontweight='bold')

    # Initialize plots
    steps = history['steps']

    # Plot 1: Cumulative Reward
    ax1 = axes[0, 0]
    line1, = ax1.plot([], [], 'b-', linewidth=2)
    ax1.set_xlim(0, len(steps))
    ax1.set_ylim(min(np.cumsum(history['rewards'])), max(np.cumsum(history['rewards'])))
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Cumulative Reward')
    ax1.set_title('Cumulative Reward Over Time')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Waiting Time
    ax2 = axes[0, 1]
    line2, = ax2.plot([], [], 'r-', linewidth=2)
    ax2.set_xlim(0, len(steps))
    ax2.set_ylim(0, max(history['waiting_times']) * 1.1)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Avg Waiting Time (s)')
    ax2.set_title('Average Waiting Time')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Queue Length
    ax3 = axes[1, 0]
    line3, = ax3.plot([], [], 'g-', linewidth=2)
    ax3.set_xlim(0, len(steps))
    ax3.set_ylim(0, max(history['queue_lengths']) * 1.1)
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Avg Queue Length')
    ax3.set_title('Average Queue Length')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Traffic Light States
    ax4 = axes[1, 1]
    ax4.set_xlim(0, len(steps))
    ax4.set_ylim(-0.5, n_agents + 0.5)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Traffic Signal ID')
    ax4.set_title('Traffic Light Actions (Green=1, Red=0)')
    ax4.set_yticks(range(n_agents))
    ax4.set_yticklabels([f'TS {i}' for i in range(n_agents)])
    ax4.grid(True, alpha=0.3, axis='x')

    # For traffic lights, we'll show colored rectangles
    rectangles = []
    for i in range(n_agents):
        rect = Rectangle((0, i-0.4), 1, 0.8, facecolor='red', alpha=0.7)
        ax4.add_patch(rect)
        rectangles.append(rect)

    plt.tight_layout()

    # Animation update function
    cumulative_reward = np.cumsum(history['rewards'])

    def animate(frame):
        # Update line plots
        line1.set_data(steps[:frame+1], cumulative_reward[:frame+1])
        line2.set_data(steps[:frame+1], history['waiting_times'][:frame+1])
        line3.set_data(steps[:frame+1], history['queue_lengths'][:frame+1])

        # Update traffic light states
        if frame < len(history['actions']):
            actions = history['actions'][frame]
            for i, (rect, action) in enumerate(zip(rectangles, actions)):
                rect.set_x(frame)
                rect.set_facecolor('green' if action == 1 else 'red')

        return [line1, line2, line3] + rectangles

    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(steps),
        interval=1000//args.fps, blit=True, repeat=True
    )

    # Save as GIF
    print(f"Saving animation to {args.output}...")
    anim.save(args.output, writer='pillow', fps=args.fps)

    plt.close()

    print(f"\n✅ Animation saved: {args.output}")
    print(f"\n{'='*70}")
    print("Recording complete!")
    print(f"{'='*70}\n")

    # Print summary
    print("Episode Summary:")
    print(f"  Total Reward: {sum(history['rewards']):.2f}")
    print(f"  Avg Waiting Time: {np.mean(history['waiting_times']):.2f}s")
    print(f"  Avg Queue Length: {np.mean(history['queue_lengths']):.2f}")
    print()


if __name__ == "__main__":
    main()
