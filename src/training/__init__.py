"""Training package for Reinforcement Learning controllers."""

from typing import Any

__all__ = ["train_q_learning_agent"]


def __getattr__(name: str) -> Any:
    """Lazy-load train_q_learning_agent to avoid premature import warnings when running CLI."""
    if name == "train_q_learning_agent":
        from src.training.train_q_learning import train_q_learning_agent

        return train_q_learning_agent
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

