"""Package huấn luyện cho các bộ điều khiển học tăng cường (RL Controllers)."""

from typing import Any

__all__ = ["train_q_learning_agent"]


def __getattr__(name: str) -> Any:
    """Tải lười (lazy-load) train_q_learning_agent để tránh cảnh báo import sớm khi chạy CLI."""
    if name == "train_q_learning_agent":
        import src.training.train_q_learning as tq
        return tq.train_q_learning_agent
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

