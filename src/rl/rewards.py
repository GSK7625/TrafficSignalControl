"""Reward functions for reinforcement learning traffic signal control."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence


class BaseRewardFunction(ABC):
    """Abstract base class for traffic signal control reward functions."""

    @abstractmethod
    def reset(self) -> None:
        """Resets internal accumulator state at the start of a decision interval or episode."""
        pass

    @abstractmethod
    def record_step(
        self,
        in_lane_queues: Dict[str, float],
        waiting_times: Optional[Dict[str, float]] = None,
    ) -> None:
        """Records traffic metrics during a single 1-second simulation step."""
        pass

    @abstractmethod
    def calculate_reward(self) -> float:
        """Computes and returns the scalar reward for the completed decision interval."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, any]:
        """Returns reward function configuration parameters."""
        pass


class NegativeAverageQueueReward(BaseRewardFunction):
    """Primary reward function: negative average queue length across the decision interval.

    Formulation:
        reward = - (1 / M) * sum_{m=1}^{M} [ sum_{l in InLanes} queue(l, m) ] * scale

    Where:
        - M: Number of simulation steps within the decision interval (typically delta_time / step_len).
        - InLanes: All incoming controlled lanes for the intersection.
        - queue(l, m): Number of halting vehicles (speed < 0.1 m/s) on lane l at step m.
        - scale: Optional scaling factor (default: 1.0).

    Properties:
        - Maximizing this reward directly minimizes intersection queue accumulation.
        - Robust against variations in decision interval length or premature episode truncation.
    """

    def __init__(self, scale: float = 1.0):
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        self.scale = float(scale)
        self._interval_queue_sums: List[float] = []

    def reset(self) -> None:
        """Clears the step queue records for the current decision interval."""
        self._interval_queue_sums.clear()

    def record_step(
        self,
        in_lane_queues: Dict[str, float],
        waiting_times: Optional[Dict[str, float]] = None,
    ) -> None:
        """Sums queue lengths across all incoming lanes in the current simulation step."""
        step_total_queue = sum(in_lane_queues.values()) if in_lane_queues else 0.0
        self._interval_queue_sums.append(float(step_total_queue))

    def calculate_reward(self) -> float:
        """Calculates negative average queue length across the recorded steps."""
        if not self._interval_queue_sums:
            return 0.0
        avg_queue = sum(self._interval_queue_sums) / len(self._interval_queue_sums)
        reward = -avg_queue * self.scale
        return float(reward)

    def get_config(self) -> Dict[str, any]:
        return {
            "name": "NegativeAverageQueueReward",
            "scale": self.scale,
        }
