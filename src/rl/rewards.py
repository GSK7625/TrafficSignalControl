"""Reward functions for reinforcement learning traffic signal control."""

import math
from typing import Any, Dict, List, Optional, Set


class NegativePressureReward:
    """Pressure-based reward function (PressLight formulation).

    Calculates negative average pressure across incoming lanes:
        pressure(l) = abs(queue(l) - mean_{m in Out(l)}(queue(m)))

    Parameters:
        scale: Scaling multiplier for the negative pressure component.
        pending_weight: Weight for unentered pending vehicles.
        topology: Pre-configured mapping of incoming lane to outgoing lanes.
    """

    def __init__(
        self,
        scale: float = 1.0,
        pending_weight: float = 0.0,
        topology: Optional[Dict[str, Set[str]]] = None,
    ):
        for key, value in (("scale", scale), ("pending_weight", pending_weight)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be non-negative and finite")

        self.scale = float(scale)
        self.pending_weight = float(pending_weight)
        self.topology: Dict[str, Set[str]] = dict(topology or {})

        self._pressure_sum = 0.0
        self._pressure_count = 0

    def set_topology_from_links(self, controlled_links: List[Any]) -> None:
        """Configures lane topology from SUMO controlled link definitions."""
        topo: Dict[str, Set[str]] = {}
        for link in controlled_links:
            if link.from_lane and link.to_lane:
                topo.setdefault(link.from_lane, set()).add(link.to_lane)
        self.topology = topo

    def reset_interval(self) -> None:
        """Resets transient metrics for a decision interval."""
        self._pressure_sum = 0.0
        self._pressure_count = 0

    def reset(self) -> None:
        """Resets the reward function state for a new episode."""
        self.reset_interval()

    def record_step(
        self,
        in_lane_queues: Dict[str, float],
        out_lane_queues: Optional[Dict[str, float]] = None,
        pending_vehicles: int = 0,
        controlled_links: Optional[List[Any]] = None,
    ) -> None:
        """Records queues for one simulation step and computes PressLight pressure."""
        if controlled_links and not self.topology:
            self.set_topology_from_links(controlled_links)

        out_q = out_lane_queues or {}
        pressure = 0.0
        for from_l, q_in in in_lane_queues.items():
            outs = self.topology.get(from_l, set())
            q_out_bar = sum(out_q.get(to_l, 0.0) for to_l in outs) / len(outs) if outs else 0.0
            pressure += abs(q_in - q_out_bar)

        self._pressure_sum += pressure + self.pending_weight * float(pending_vehicles)
        self._pressure_count += 1

    def calculate_reward(self) -> float:
        """Calculates normalized reward for the decision interval."""
        if not self._pressure_count:
            return 0.0
        return -(self._pressure_sum / self._pressure_count) * self.scale

    def get_config(self) -> Dict[str, Any]:
        return {
            "name": "NegativePressureReward",
            "scale": self.scale,
            "pending_weight": self.pending_weight,
        }
