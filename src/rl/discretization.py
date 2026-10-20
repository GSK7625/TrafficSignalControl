"""State discretization for Tabular Q-Learning in traffic signal control."""

from __future__ import annotations

import bisect
import math
from typing import TYPE_CHECKING, Any, Dict, Set, Tuple

if TYPE_CHECKING:
    from src.controllers.base import IntersectionObservation


class StateDiscretizer:
    """Discretizes continuous intersection observations into discrete state tuples.

    Uses aggregate waiting queue representation: (current_phase, serving_bin, waiting_bin, time_bin).
    """

    def __init__(
        self,
        min_green: float = 5.0,
        max_effective_green: float = 25.0,
        serving_queue_thresholds: Tuple[float, ...] = (1.0, 5.0),
        waiting_queue_thresholds: Tuple[float, ...] = (1.0, 6.0, 15.0),
        group_by: str = "aggregate",
    ):
        self.min_green = float(min_green)
        self.max_effective_green = float(max_effective_green)
        self.serving_thresholds = tuple(float(x) for x in serving_queue_thresholds)
        self.waiting_thresholds = tuple(float(x) for x in waiting_queue_thresholds)
        self.group_by = str(group_by).lower()

        if not (math.isfinite(self.min_green) and self.min_green >= 0):
            raise ValueError("min_green must be non-negative and finite")
        if not (math.isfinite(self.max_effective_green) and self.max_effective_green >= self.min_green):
            raise ValueError("max_effective_green must be finite and >= min_green")
        for thresholds in (self.serving_thresholds, self.waiting_thresholds):
            if any(not math.isfinite(x) or x < 0 for x in thresholds) or any(a >= b for a, b in zip(thresholds, thresholds[1:])):
                raise ValueError("Queue thresholds must be finite, non-negative and strictly increasing")
        if self.group_by not in ("aggregate", "branch"):
            raise ValueError(f"group_by only supports 'aggregate', got {group_by}")

    def _bin_value(self, val: float, thresholds: Tuple[float, ...]) -> int:
        return bisect.bisect_right(thresholds, val)

    def _get_green_lanes(self, obs: IntersectionObservation) -> Set[str]:
        p = obs.current_phase
        if not obs.green_phases or not (0 <= p < len(obs.green_phases)):
            return set()
        phase_str = obs.green_phases[p]
        return {
            link.from_lane
            for link in obs.controlled_links
            if link.from_lane and 0 <= link.signal_index < len(phase_str) and phase_str[link.signal_index] in "Ggs"
        }

    def discretize(self, obs: IntersectionObservation) -> Tuple[int, int, int, int]:
        green_lanes = self._get_green_lanes(obs)
        serving_q = sum(q for lane, q in obs.in_lane_queues.items() if lane in green_lanes)
        serving_bin = self._bin_value(serving_q, self.serving_thresholds)

        waiting_q = sum(q for lane, q in obs.in_lane_queues.items() if lane not in green_lanes)
        waiting_bin = self._bin_value(waiting_q, self.waiting_thresholds)

        time_bin = 0 if obs.is_in_transition else self._bin_value(obs.time_in_phase, (self.min_green, self.max_effective_green))
        return (obs.current_phase, serving_bin, waiting_bin, time_bin)

    def get_config(self) -> Dict[str, Any]:
        return {
            "min_green": self.min_green,
            "max_effective_green": self.max_effective_green,
            "serving_thresholds": list(self.serving_thresholds),
            "waiting_thresholds": list(self.waiting_thresholds),
            "group_by": self.group_by,
        }

    @classmethod
    def from_config(cls, config: Dict[str, Any], min_green: float = 5.0) -> "StateDiscretizer":
        """Restores checkpoint thresholds, defaulting to aggregate mode."""
        return cls(
            min_green=config.get("min_green", min_green),
            max_effective_green=config.get("max_effective_green", 25.0),
            serving_queue_thresholds=tuple(config.get("serving_thresholds", (1.0, 5.0))),
            waiting_queue_thresholds=tuple(config.get("waiting_thresholds", (1.0, 6.0, 15.0))),
            group_by=config.get("group_by", "aggregate"),
        )
