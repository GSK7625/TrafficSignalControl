"""Reward functions for reinforcement learning traffic signal control."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRewardFunction(ABC):
    @abstractmethod
    def reset(self) -> None:
        pass

    @abstractmethod
    def record_step(
        self,
        in_lane_queues: Dict[str, float],
        out_lane_queues: Optional[Dict[str, float]] = None,
        waiting_times: Optional[Dict[str, float]] = None,
        pending_vehicles: int = 0,
        controlled_links: Optional[List[Any]] = None,
    ) -> None:
        pass

    @abstractmethod
    def calculate_reward(self) -> float:
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        pass


class NegativePressureReward(BaseRewardFunction):
    """Pressure-based reward function (PressLight formulation)."""

    def __init__(self, scale: float = 1.0, pending_weight: float = 0.0):
        self.scale = float(scale)
        self.pending_weight = float(pending_weight)
        self._interval_pressures: List[float] = []

    def reset(self) -> None:
        self._interval_pressures.clear()

    def record_step(
        self,
        in_lane_queues: Dict[str, float],
        out_lane_queues: Optional[Dict[str, float]] = None,
        waiting_times: Optional[Dict[str, float]] = None,
        pending_vehicles: int = 0,
        controlled_links: Optional[List[Any]] = None,
    ) -> None:
        if not in_lane_queues:
            self._interval_pressures.append(self.pending_weight * float(pending_vehicles))
            return

        out_q = out_lane_queues or {}

        if controlled_links:
            outs_per_in: Dict[str, List[str]] = {}
            for link in controlled_links:
                if link.from_lane and link.to_lane:
                    outs_per_in.setdefault(link.from_lane, []).append(link.to_lane)

            pressure = 0.0
            for from_l, q_in in in_lane_queues.items():
                outs = outs_per_in.get(from_l, [])
                q_out_bar = sum(out_q.get(to_l, 0.0) for to_l in outs) / len(outs) if outs else 0.0
                pressure += abs(q_in - q_out_bar)
        elif out_q:
            mean_out = sum(out_q.values()) / max(1, len(in_lane_queues))
            pressure = sum(abs(q_in - mean_out) for q_in in in_lane_queues.values())
        else:
            pressure = sum(in_lane_queues.values())

        self._interval_pressures.append(pressure + self.pending_weight * float(pending_vehicles))

    def calculate_reward(self) -> float:
        if not self._interval_pressures:
            return 0.0
        return -float(sum(self._interval_pressures) / len(self._interval_pressures)) * self.scale

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "NegativePressureReward",
            "scale": self.scale,
            "pending_weight": self.pending_weight,
        }
