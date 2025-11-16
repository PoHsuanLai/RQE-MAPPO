"""
Record trained RQE-MAPPO agent in SUMO as GIF/video

Usage:
    python -m src.record_sumo --exp_name sumo_single-intersection_tau0.5_eps0.01 --output agent.gif
"""

import argparse
import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import subprocess

from sumo_rl.environment.env import SumoEnvironment
from src.algorithms import RQE_MAPPO, RQEConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Record SUMO agent as GIF")

    parser.add_argument("--exp_name", type=str, required=True,
                        help="Experiment name to load model from")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Log directory")
    parser.add_argument("--output", type=str, default="agent.gif",
                        help="Output file (*.gif or *.mp4)")
    parser.add_argument("--n_episodes", type=int, default=1,
                        help="Number of episodes to record")
    parser.add_argument("--fps", type=int, default=5,
                        help="Frames per second")
    parser.add_argument("--checkpoint", type=str, default="final",
                        help="Checkpoint to load (final, 100000, etc.)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set SUMO_HOME
    if 'SUMO_HOME' not in os.environ:
        os.environ['SUMO_HOME'] = '/usr/share/sumo'

    print(f"\n{'='*70}")
    print("Recording RQE-MAPPO Agent in SUMO")
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

    # Create environment
    import sumo_rl
    sumo_rl_path = Path(sumo_rl.__file__).parent
    nets_path = sumo_rl_path / 'nets'

    net_file = str(nets_path / config_data['net'] / f"{config_data['net']}.net.xml")
    route_file = str(nets_path / config_data['net'] / f"{config_data['net']}.rou.xml")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create SUMO environment with GUI and screenshot saving
    env = SumoEnvironment(
        net_file=net_file,
        route_file=route_file,
        use_gui=True,  # Enable GUI for recording
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
    print(f"Recording {args.n_episodes} episode(s)...")
    print(f"Output: {args.output}\n")
    print("Note: SUMO-GUI will open. Watch the traffic lights being controlled!")
    print("      Screenshots will be saved automatically.\n")

    # Record episodes
    frames = []
    screenshot_dir = Path("sumo_screenshots")
    screenshot_dir.mkdir(exist_ok=True)

    for episode in range(args.n_episodes):
        print(f"\nEpisode {episode + 1}/{args.n_episodes}")

        obs_dict = env.reset()
        done = False
        step = 0
        episode_reward = 0

        while not done and step < 200:  # Limit to 200 steps
            # Get observations as tensor
            obs_list = [obs_dict[agent] for agent in agents]
            obs_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32, device=device)

            # Get action from trained agent
            with torch.no_grad():
                actions, _, _ = agent.get_actions(obs_tensor.unsqueeze(0), deterministic=True)
                actions = actions.squeeze(0)

            # Convert to dict
            actions_np = actions.cpu().numpy().astype(int)
            action_dict = {agent: int(actions_np[i]) for i, agent in enumerate(agents)}

            # Step environment
            obs_dict, reward_dict, done_dict, info = env.step(action_dict)

            # Take screenshot every few steps
            if step % 5 == 0:
                screenshot_file = screenshot_dir / f"step_{episode:03d}_{step:04d}.png"

                # Use SUMO's screenshot functionality via TraCI
                try:
                    import traci
                    traci.gui.screenshot("View #0", str(screenshot_file))

                    # Load and store frame
                    img = Image.open(screenshot_file)
                    frames.append(img.copy())

                except Exception as e:
                    print(f"Warning: Could not take screenshot: {e}")

            episode_reward += sum(reward_dict.values())
            done = all(done_dict.values())
            step += 1

            if step % 20 == 0:
                print(f"  Step {step}: Reward={episode_reward:.2f}")

        print(f"Episode {episode + 1} complete: {step} steps, Reward={episode_reward:.2f}")

    env.close()

    # Save as GIF or video
    if len(frames) > 0:
        output_path = Path(args.output)

        if output_path.suffix == '.gif':
            print(f"\nSaving GIF to {args.output}...")
            frames[0].save(
                args.output,
                save_all=True,
                append_images=frames[1:],
                duration=1000//args.fps,
                loop=0
            )
            print(f"✅ GIF saved: {args.output}")

        elif output_path.suffix == '.mp4':
            print(f"\nConverting to MP4...")
            # Save frames as temp images
            temp_dir = screenshot_dir / "temp"
            temp_dir.mkdir(exist_ok=True)

            for i, frame in enumerate(frames):
                frame.save(temp_dir / f"frame_{i:04d}.png")

            # Use ffmpeg to create video
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", str(args.fps),
                "-i", str(temp_dir / "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                args.output
            ])

            print(f"✅ Video saved: {args.output}")

        # Cleanup
        import shutil
        shutil.rmtree(screenshot_dir)

    else:
        print("\n⚠️  No frames captured. Make sure SUMO-GUI is visible and TraCI screenshots are working.")

    print(f"\n{'='*70}")
    print("Recording complete!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
