"""Reinforcement Learning module for traffic signal control."""

from src.rl.discretization import StateDiscretizer
from src.rl.rewards import (
    BaseRewardFunction,
    NegativePressureReward,
)


def __getattr__(name: str):
    if name == "TrafficSignalRLEnv":
        from src.rl.environment import TrafficSignalRLEnv
        return TrafficSignalRLEnv
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TrafficSignalRLEnv",
    "StateDiscretizer",
    "BaseRewardFunction",
    "NegativePressureReward",
]
