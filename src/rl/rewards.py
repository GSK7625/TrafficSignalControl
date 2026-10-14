"""Reward functions for reinforcement learning traffic signal control."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional


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
        pending_vehicles: int = 0,
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
    """Primary reward function: negative average queue length across the decision interval,
    incorporating pending uninserted vehicles to prevent starvation / entry bottleneck blindness.

    Formulation:
        reward = - (1 / M) * sum_{m=1}^{M} [ sum_{l in InLanes} queue(l, m) + pending_weight * pending(m) ] * scale

    Where:
        - M: Number of simulation steps within the decision interval.
        - InLanes: All incoming controlled lanes for the intersection.
        - queue(l, m): Number of halting vehicles (speed < 0.1 m/s) on lane l at step m.
        - pending(m): Number of vehicles waiting to enter the network at step m.
        - pending_weight: Penalty weighting factor for uninserted vehicles (default: 1.0).
        - scale: Optional scaling factor (default: 1.0).
    """

    def __init__(self, scale: float = 1.0, pending_weight: float = 1.0):
        if scale <= 0:
            raise ValueError(f"scale must be positive, got {scale}")
        if pending_weight < 0:
            raise ValueError(f"pending_weight must be non-negative, got {pending_weight}")
        self.scale = float(scale)
        self.pending_weight = float(pending_weight)
        self._interval_queue_sums: List[float] = []

    def reset(self) -> None:
        """Clears the step queue records for the current decision interval."""
        self._interval_queue_sums.clear()

    def record_step(
        self,
        in_lane_queues: Dict[str, float],
        waiting_times: Optional[Dict[str, float]] = None,
        pending_vehicles: int = 0,
    ) -> None:
        """Sums queue lengths and pending uninserted vehicles in the current simulation step."""
        step_lane_queue = sum(in_lane_queues.values()) if in_lane_queues else 0.0
        step_total_penalty = step_lane_queue + (self.pending_weight * float(pending_vehicles))
        self._interval_queue_sums.append(float(step_total_penalty))

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
            "pending_weight": self.pending_weight,
        }
