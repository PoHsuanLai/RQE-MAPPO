from .risk_measures import get_risk_measure, EntropicRisk, CVaR, MeanVariance
from .rqe_exact import ExactRQE

# Core RQE Solvers (game-theoretic equilibrium solvers)
from .rqe_solver import RQESolver, RQEConfig as RQESolverConfig, solve_rqe
from .markov_rqe_solver import (
    MarkovRQESolver, MarkovGameConfig, MarkovGame,
    PenaltyType, RegularizerType, create_random_markov_game
)

# Multi-agent implementations with self-play (no Ray dependency)
from .rqe_mappo import RQE_MAPPO, RQEConfig
from .true_rqe_mappo import TrueRQE_MAPPO, TrueRQEConfig

# Game-theoretic approaches
from .psro_rqe import PSRO_RQE, PSRORQEConfig

# RLlib implementations (optional - only import if Ray is available)
try:
    from .rqe_ppo_rllib import RQEPPO, RQEPPOConfig
    from .true_rqe_ppo_rllib import TrueRQEPPO, TrueRQEPPOConfig
    _has_rllib = True
except ImportError:
    _has_rllib = False

__all__ = [
    "get_risk_measure", "EntropicRisk", "CVaR", "MeanVariance",
    "ExactRQE",
    # Core RQE Solvers
    "RQESolver", "RQESolverConfig", "solve_rqe",
    "MarkovRQESolver", "MarkovGameConfig", "MarkovGame",
    "PenaltyType", "RegularizerType", "create_random_markov_game",
    # Multi-agent implementations
    "RQE_MAPPO", "RQEConfig",
    "TrueRQE_MAPPO", "TrueRQEConfig",
    "PSRO_RQE", "PSRORQEConfig",
]

if _has_rllib:
    __all__.extend(["RQEPPO", "RQEPPOConfig", "TrueRQEPPO", "TrueRQEPPOConfig"])
