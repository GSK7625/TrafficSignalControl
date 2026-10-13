"""Traffic signal controllers package."""

from src.controllers.base import (
    BaseTrafficController,
    ControlledLink,
    IntersectionObservation,
)
from src.controllers.fixed_time import FixedTimeController
from src.controllers.max_pressure import MaxPressureController, compute_phase_pressure


def __getattr__(name: str):
    if name == "QLearningController":
        from src.controllers.q_learning import QLearningController
        return QLearningController
    if name == "DQNController":
        from src.controllers.dqn import DQNController
        return DQNController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseTrafficController",
    "ControlledLink",
    "IntersectionObservation",
    "FixedTimeController",
    "MaxPressureController",
    "QLearningController",
    "DQNController",
    "compute_phase_pressure",
]
