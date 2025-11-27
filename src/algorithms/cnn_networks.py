"""
Example CNN networks for Deep RQE algorithms

These demonstrate how to create custom networks for visual observations.
Users can create their own networks following the same interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CNNQNetwork(nn.Module):
    """
    CNN Q-Network for visual observations (e.g., Atari)

    Interface required by DeepRQE_QLearning:
    - __init__(my_action_dim, opponent_action_dim, **kwargs)
    - forward(obs) -> [batch, my_action_dim, opponent_action_dim]
    """
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
        # You can also pass this as a parameter
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


class CNNActor(nn.Module):
    """
    CNN Actor for visual observations (e.g., Atari)

    Interface required by DeepRQE_MAPPO:
    - __init__(action_dim, **kwargs)
    - forward(obs) -> distribution
    """
    def __init__(self, action_dim: int, input_channels: int = 3,
                 features_dim: int = 512, action_type: str = "discrete"):
        super().__init__()

        self.action_dim = action_dim
        self.action_type = action_type

        # CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 210, 160)
            cnn_out_size = self.cnn(dummy).shape[1]

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_size, features_dim),
            nn.ReLU(),
        )

        # Action head
        if action_type == "discrete":
            self.action_head = nn.Linear(features_dim, action_dim)
        else:
            self.mean_head = nn.Linear(features_dim, action_dim)
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs: torch.Tensor) -> torch.distributions.Distribution:
        """
        Args:
            obs: [batch, channels, height, width]

        Returns:
            Distribution over actions
        """
        features = self.cnn(obs)
        features = self.fc(features)

        if self.action_type == "discrete":
            logits = self.action_head(features)
            return torch.distributions.Categorical(logits=logits)
        else:
            mean = self.mean_head(features)
            std = torch.exp(self.log_std)
            return torch.distributions.Normal(mean, std)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        """Sample action and compute log probability"""
        dist = self.forward(obs)

        if deterministic:
            if self.action_type == "discrete":
                action = dist.probs.argmax(dim=-1)
            else:
                action = dist.mean
        else:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        return action, log_prob
