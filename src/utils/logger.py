"""
Logger for training metrics

Uses tensorboard for visualization
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """
    Logger for training metrics

    Logs to:
    - Tensorboard
    - JSON file
    - Console
    """

    def __init__(
        self,
        log_dir: str,
        exp_name: str = "experiment",
        use_tensorboard: bool = True
    ):
        """
        Args:
            log_dir: Directory to save logs
            exp_name: Experiment name
            use_tensorboard: Whether to use tensorboard
        """
        self.log_dir = Path(log_dir) / exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.writer = SummaryWriter(str(self.log_dir))

        # Store metrics for JSON export
        self.metrics = []

        print(f"Logging to: {self.log_dir}")

    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int
    ):
        """Log a scalar value"""
        if self.use_tensorboard:
            self.writer.add_scalar(tag, value, step)

    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: int
    ):
        """Log multiple scalar values"""
        if self.use_tensorboard:
            self.writer.add_scalars(tag, values, step)

    def log_metrics(
        self,
        metrics: Dict[str, Any],
        step: int,
        prefix: str = ""
    ):
        """
        Log a dictionary of metrics

        Args:
            metrics: Dictionary of metrics
            step: Training step
            prefix: Prefix for metric names (e.g., "train/", "eval/")
        """
        # Add step to metrics
        metrics_with_step = {"step": step, **metrics}
        self.metrics.append(metrics_with_step)

        # Log to tensorboard
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self.log_scalar(f"{prefix}{key}", value, step)

    def log_episode(
        self,
        episode: int,
        episode_reward: float,
        episode_length: int,
        **kwargs
    ):
        """Log episode statistics"""
        metrics = {
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            **kwargs
        }

        self.log_metrics(metrics, episode, prefix="episode/")

        # Print to console
        print(
            f"Episode {episode:4d} | "
            f"Reward: {episode_reward:8.2f} | "
            f"Length: {episode_length:4d}"
        )

    def save_config(self, config: Dict[str, Any]):
        """Save configuration to JSON"""
        config_path = self.log_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2, default=str)

    def save_metrics(self):
        """Save all metrics to JSON"""
        metrics_path = self.log_dir / "metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)

    def close(self):
        """Close logger"""
        self.save_metrics()
        if self.use_tensorboard:
            self.writer.close()


if __name__ == "__main__":
    # Test logger
    print("Testing Logger...")

    logger = Logger(
        log_dir="logs/test",
        exp_name="test_experiment"
    )

    # Log some metrics
    for step in range(10):
        logger.log_metrics({
            "loss": 1.0 / (step + 1),
            "accuracy": step * 0.1
        }, step, prefix="train/")

    # Log episode
    logger.log_episode(
        episode=1,
        episode_reward=100.5,
        episode_length=250
    )

    # Save config
    logger.save_config({
        "learning_rate": 0.001,
        "batch_size": 32
    })

    logger.close()

    print("\n✓ All tests passed!")
    print(f"Check logs at: {logger.log_dir}")
