#!/usr/bin/env python3
"""
Render trained Boxing models and save as GIF

Supports RQE-MAPPO, True RQE-MAPPO, and Deep RQE models
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from PIL import Image
import imageio
from pettingzoo.atari import boxing_v2

# Add src to path (both as 'src' and as absolute path for different import styles)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))


# ==================== CNN Network Definitions ====================

class CNNQNetwork(nn.Module):
    """CNN Q-Network for Deep RQE (game-theoretic Q-values)"""

    def __init__(self, my_action_dim: int, opponent_action_dim: int,
                 input_channels: int = 3, features_dim: int = 512):
        super().__init__()

        self.my_action_dim = my_action_dim
        self.opponent_action_dim = opponent_action_dim

        # CNN feature extractor (Nature DQN architecture)
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size (for Atari 210x160)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 210, 160)
            cnn_out_size = self.cnn(dummy).shape[1]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size, features_dim),
            nn.ReLU(),
        )

        # Q-value head
        output_dim = my_action_dim * opponent_action_dim
        self.q_head = nn.Linear(features_dim, output_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: [batch, channels, height, width] - visual observations

        Returns:
            Q-values: [batch, my_action_dim, opponent_action_dim]
        """
        batch_size = obs.shape[0]

        # Extract features
        features = self.cnn(obs)
        features = self.fc(features)

        # Compute Q-values
        q_flat = self.q_head(features)
        q_matrix = q_flat.view(batch_size, self.my_action_dim, self.opponent_action_dim)

        return q_matrix


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render trained Boxing models as GIFs"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file (.pt)"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["rqe_mappo", "true_rqe_mappo", "deep_rqe"],
        help="Type of model"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=3,
        help="Number of episodes to render"
    )
    parser.add_argument(
        "--output_gif",
        type=str,
        default="results/boxing_render.gif",
        help="Output path for GIF"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Frames per second for GIF"
    )
    parser.add_argument(
        "--max_cycles",
        type=int,
        default=1800,
        help="Maximum cycles per episode (default: 1800 for Boxing)"
    )

    return parser.parse_args()


class CNNActor(nn.Module):
    """CNN-based actor for visual observations (must match training)"""

    def __init__(self, action_dim: int):
        super().__init__()

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 210, 160)
            cnn_out_size = self.cnn(dummy).shape[1]

        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.cnn(obs)
        logits = self.fc(features)
        return torch.clamp(logits, min=-10.0, max=10.0)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        logits = self.forward(obs)
        if deterministic:
            actions = logits.argmax(dim=-1)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()
        else:
            dist = Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            entropies = dist.entropy()
        return actions, log_probs, entropies


def load_rqe_mappo_checkpoint(checkpoint_path):
    """Load RQE-MAPPO checkpoint"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config
    config = checkpoint['config']

    # Create CNN actors (same architecture as training)
    actors = []
    for i, actor_state in enumerate(checkpoint['actors']):
        actor = CNNActor(action_dim=config.action_dim)
        actor.load_state_dict(actor_state)
        actor.to(device)
        actor.eval()
        actors.append(actor)

    # Create a simple namespace to hold actors and device
    class AgentHolder:
        pass
    agents = AgentHolder()
    agents.actors = actors
    agents.device = device

    print(f"Loaded RQE-MAPPO checkpoint from {checkpoint_path}")
    print(f"Config: tau={config.tau}, epsilon={config.epsilon}, risk_measure={config.risk_measure}")

    return agents, config


def load_true_rqe_mappo_checkpoint(checkpoint_path):
    """Load True RQE-MAPPO checkpoint"""
    from algorithms.true_rqe_mappo import TrueRQE_MAPPO

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Create agents from saved config
    config = checkpoint['config']
    config.device = device
    agents = TrueRQE_MAPPO(config)

    # Load actor weights
    for i, actor in enumerate(agents.actors):
        actor.load_state_dict(checkpoint['actors'][i])
        actor.to(device)
        actor.eval()

    print(f"Loaded True RQE-MAPPO checkpoint from {checkpoint_path}")
    print(f"Config: tau={config.tau}, eps={config.eps}, risk_type={config.risk_type}")

    return agents, config


def load_deep_rqe_checkpoint(checkpoint_path):
    """Load Deep RQE Q-learning checkpoint"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load checkpoint (our own trusted checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract config and Q-networks
    config = checkpoint.get('config', None)
    q_networks_state = checkpoint.get('q_networks', [])

    # Create Q-networks (game-theoretic Q-networks)
    action_dim = 18  # Boxing action space
    q_networks = []

    for i, state_dict in enumerate(q_networks_state):
        network = CNNQNetwork(
            my_action_dim=action_dim,
            opponent_action_dim=action_dim,
            input_channels=3,
            features_dim=512
        )
        network.load_state_dict(state_dict)
        network.to(device)
        network.eval()
        q_networks.append(network)

    print(f"Loaded Deep RQE checkpoint from {checkpoint_path}")
    print(f"Episode: {checkpoint.get('episode', 'N/A')}")
    if config:
        print(f"Config: tau={config.tau if hasattr(config, 'tau') else 'N/A'}")

    return q_networks, config, device


def preprocess_obs(obs):
    """Preprocess observation for CNN"""
    # obs is (210, 160, 3) from environment
    # Need to convert to (3, 210, 160) and normalize
    obs = np.transpose(obs, (2, 0, 1))  # HWC -> CHW
    obs = obs.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return obs


def evaluate_boxing(agents, env, num_episodes, max_cycles, model_type, device=None):
    """Evaluate agents on Boxing and collect frames"""

    episode_rewards = []
    all_frames = []

    for ep in range(num_episodes):
        env.reset()
        frames = []
        episode_reward = {0: 0, 1: 0}

        for agent in env.agent_iter(max_iter=max_cycles):
            observation, reward, termination, truncation, info = env.last()

            # Render frame
            frame = env.render()
            if frame is not None and agent == env.agents[0]:  # Only save once per cycle
                frames.append(frame)

            if termination or truncation:
                action = None
            else:
                # Get agent index
                agent_idx = 0 if agent == env.agents[0] else 1

                # Preprocess observation
                obs_processed = preprocess_obs(observation)
                obs_tensor = torch.FloatTensor(obs_processed).unsqueeze(0)

                if device:
                    obs_tensor = obs_tensor.to(device)
                elif hasattr(agents, 'device'):
                    obs_tensor = obs_tensor.to(agents.device)

                # Get action based on model type
                with torch.no_grad():
                    if model_type in ["rqe_mappo", "true_rqe_mappo"]:
                        # MAPPO style: get action for single agent
                        actor = agents.actors[agent_idx]
                        action_tensor, _, _ = actor.get_action(obs_tensor, deterministic=True)
                        action = action_tensor.item()
                    else:  # deep_rqe
                        # Deep RQE Q-learning: Q[batch, my_action, opp_action]
                        # For deterministic policy: take best response (maximin)
                        q_network = agents[agent_idx]
                        q_matrix = q_network(obs_tensor)  # [1, my_actions, opp_actions]

                        # Maximin strategy: for each of my actions, compute worst-case value
                        # Then pick the action with best worst-case
                        q_my_actions = q_matrix[0]  # [my_actions, opp_actions]
                        worst_case_values = q_my_actions.min(dim=1)[0]  # [my_actions]
                        action = worst_case_values.argmax().item()

            # Track reward
            if reward > 0:
                episode_reward[agent_idx] += reward

            env.step(action)

        # Episode finished
        total_reward = sum(episode_reward.values())
        episode_rewards.append(total_reward)

        if ep == 0:  # Save first episode frames for GIF
            all_frames = frames

        print(f"Episode {ep+1}/{num_episodes} - Agent 0: {episode_reward[0]:.0f}, "
              f"Agent 1: {episode_reward[1]:.0f}, Total: {total_reward:.0f}")

    return episode_rewards, all_frames


def save_gif(frames, output_path, fps=15):
    """Save frames as GIF"""
    if not frames:
        print("No frames to save!")
        return

    # Convert frames to uint8 if needed
    images = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            images.append(frame)
        else:
            images.append(frame)

    # Save as GIF
    imageio.mimsave(output_path, images, fps=fps)
    print(f"\nGIF saved to: {output_path}")
    print(f"Total frames: {len(frames)}")


def main():
    args = parse_args()

    print("=" * 70)
    print("Rendering Boxing Episode")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model type: {args.model_type}")
    print(f"Episodes: {args.num_episodes}")
    print("=" * 70)

    # Create Boxing environment with rendering
    env = boxing_v2.env(render_mode="rgb_array")

    # Load model based on type
    if args.model_type == "rqe_mappo":
        agents, config = load_rqe_mappo_checkpoint(args.checkpoint)
        device = agents.device
    elif args.model_type == "true_rqe_mappo":
        agents, config = load_true_rqe_mappo_checkpoint(args.checkpoint)
        device = agents.device
    elif args.model_type == "deep_rqe":
        agents, config, device = load_deep_rqe_checkpoint(args.checkpoint)
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

    # Evaluate and collect frames
    episode_rewards, frames = evaluate_boxing(
        agents, env, args.num_episodes, args.max_cycles, args.model_type, device
    )

    # Print statistics
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Mean Total Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")

    # Save GIF
    if frames:
        save_gif(frames, args.output_gif, fps=args.fps)
    else:
        print("\nWarning: No frames captured for GIF!")

    print("=" * 70)

    env.close()


if __name__ == "__main__":
    main()
