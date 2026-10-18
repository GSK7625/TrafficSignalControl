"""State discretization for Tabular Q-Learning in traffic signal control."""

from typing import Any, Dict, Set, Tuple
from src.controllers.base import IntersectionObservation


class StateDiscretizer:
    """Discretizes continuous intersection observations into discrete state tuples."""

    def __init__(
        self,
        min_green: float = 5.0,
        max_effective_green: float = 25.0,
        serving_queue_thresholds: Tuple[float, ...] = (1.0, 5.0),
        waiting_queue_thresholds: Tuple[float, ...] = (1.0, 6.0, 15.0),
    ):
        self.min_green = float(min_green)
        self.max_effective_green = float(max_effective_green)
        self.serving_thresholds = tuple(float(x) for x in serving_queue_thresholds)
        self.waiting_thresholds = tuple(float(x) for x in waiting_queue_thresholds)

    def _get_green_lanes(self, obs: IntersectionObservation) -> Set[str]:
        if not obs.green_phases or obs.current_phase >= len(obs.green_phases):
            return set()
        phase_str = obs.green_phases[obs.current_phase]
        return {
            link.from_lane
            for link in obs.controlled_links
            if link.from_lane and 0 <= link.signal_index < len(phase_str) and phase_str[link.signal_index] in ("G", "g")
        }

    def _bin_value(self, val: float, thresholds: Tuple[float, ...]) -> int:
        for i, th in enumerate(thresholds):
            if val < th:
                return i
        return len(thresholds)

    def discretize(self, obs: IntersectionObservation) -> Tuple[int, int, int, int]:
        green_lanes = self._get_green_lanes(obs)
        serving_q = sum(q for lane, q in obs.in_lane_queues.items() if lane in green_lanes)
        waiting_q = sum(q for lane, q in obs.in_lane_queues.items() if lane not in green_lanes)

        serving_bin = self._bin_value(serving_q, self.serving_thresholds)
        waiting_bin = self._bin_value(waiting_q, self.waiting_thresholds)

        if obs.is_in_transition or obs.time_in_phase < self.min_green:
            time_bin = 0
        elif obs.time_in_phase < self.max_effective_green:
            time_bin = 1
        else:
            time_bin = 2

        return (obs.current_phase, serving_bin, waiting_bin, time_bin)

    def get_config(self) -> Dict[str, Any]:
        return {
            "min_green": self.min_green,
            "max_effective_green": self.max_effective_green,
            "serving_thresholds": list(self.serving_thresholds),
            "waiting_thresholds": list(self.waiting_thresholds),
        }
