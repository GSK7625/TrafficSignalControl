"""Reward functions for reinforcement learning traffic signal control.

Directly aligns multi-objective reward signals with 5 core traffic metrics:
  1. Average Waiting Time: delay penalty
  2. Average Travel Time / Time Loss: speed loss penalty
  3. Mean & Max Queue Length: quadratic queue penalty to resolve bottlenecks
  4. Throughput: positive reward for completed vehicles
  5. Lost Time / Switching: clearance interval penalty to avoid excessive phase flickering
"""

import math
from typing import Any, Dict, Optional


class MetricsAlignedReward:
    """Multi-objective reward function aligned directly with 5 core traffic metrics.

    Calculates reward for a decision interval:
        R = scale * (
            w_completed * completed_vehicles
            - w_queue * mean(sum(queue^2))
            - w_waiting * mean(waiting_time)
            - w_time_loss * mean(time_loss)
            - (w_switch if phase_switched else 0.0)
        )
    """

    def __init__(
        self,
        scale: float = 1.0,
        w_completed: float = 2.0,
        w_queue: float = 0.05,
        w_waiting: float = 0.02,
        w_time_loss: float = 0.01,
        w_switch: float = 0.5,
    ) -> None:
        """Initializes MetricsAlignedReward with metric weightings."""
        for name, val in (
            ("scale", scale),
            ("w_completed", w_completed),
            ("w_queue", w_queue),
            ("w_waiting", w_waiting),
            ("w_time_loss", w_time_loss),
            ("w_switch", w_switch),
        ):
            if not math.isfinite(val) or val < 0:
                raise ValueError(f"{name} must be non-negative and finite, got {val}")

        self.scale = float(scale)
        self.w_completed = float(w_completed)
        self.w_queue = float(w_queue)
        self.w_waiting = float(w_waiting)
        self.w_time_loss = float(w_time_loss)
        self.w_switch = float(w_switch)

        self._completed_sum = 0.0
        self._queue_sq_sum = 0.0
        self._waiting_sum = 0.0
        self._time_loss_sum = 0.0
        self._has_switch = False
        self._step_count = 0

    def reset_interval(self) -> None:
        """Resets transient metrics accumulated during decision interval."""
        self._completed_sum = 0.0
        self._queue_sq_sum = 0.0
        self._waiting_sum = 0.0
        self._time_loss_sum = 0.0
        self._has_switch = False
        self._step_count = 0

    def reset(self) -> None:
        """Resets the reward function state for a new episode."""
        self.reset_interval()

    def record_step(
        self,
        in_lane_queues: Optional[Dict[str, float]] = None,
        waiting_time: float = 0.0,
        completed_vehicles: int = 0,
        time_loss: float = 0.0,
        is_switch: bool = False,
        **kwargs: Any,
    ) -> None:
        """Records traffic measurements for a single simulation step.

        Args:
            in_lane_queues: Dict mapping lane ID to queue length / halted count.
            waiting_time: Sum of waiting time of active vehicles at this step.
            completed_vehicles: Number of vehicles that completed trips at this step.
            time_loss: Sum of time loss of active vehicles at this step.
            is_switch: True if a phase switch occurred during this step.
        """
        if in_lane_queues:
            self._queue_sq_sum += sum(float(q) ** 2 for q in in_lane_queues.values())
        self._waiting_sum += float(waiting_time or 0.0)
        self._completed_sum += float(completed_vehicles or 0)
        self._time_loss_sum += float(time_loss or 0.0)
        if is_switch:
            self._has_switch = True
        self._step_count += 1

    def calculate_reward(self) -> float:
        """Calculates normalized reward for the decision interval.

        Returns:
            Computed scalar reward value.
        """
        if self._step_count == 0:
            return 0.0

        avg_queue_sq = self._queue_sq_sum / self._step_count
        avg_waiting = self._waiting_sum / self._step_count
        avg_time_loss = self._time_loss_sum / self._step_count

        reward = (
            self.w_completed * self._completed_sum
            - self.w_queue * avg_queue_sq
            - self.w_waiting * avg_waiting
            - self.w_time_loss * avg_time_loss
            - (self.w_switch if self._has_switch else 0.0)
        )
        return float(reward * self.scale)

    def get_config(self) -> Dict[str, Any]:
        """Returns reward function configuration parameters."""
        return {
            "name": "MetricsAlignedReward",
            "scale": self.scale,
            "w_completed": self.w_completed,
            "w_queue": self.w_queue,
            "w_waiting": self.w_waiting,
            "w_time_loss": self.w_time_loss,
            "w_switch": self.w_switch,
        }
