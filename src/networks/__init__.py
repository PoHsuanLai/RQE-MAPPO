"""
Network architectures for different observation types
"""

from .fc_networks import FCActor, FCDistributionalCritic
from .cnn_networks import CNNActor, CNNDistributionalCritic

__all__ = [
    "FCActor",
    "FCDistributionalCritic",
    "CNNActor",
    "CNNDistributionalCritic",
]
