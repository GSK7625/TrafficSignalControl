"""State discretization module for Tabular Q-Learning in traffic signal control."""

from typing import Any, Dict, List, Set, Tuple
from src.controllers.base import IntersectionObservation


class StateDiscretizer:
    """Discretizes continuous intersection observations into compact discrete state tuples.

    Discretization Strategy:
    1. Active Green Phase: Discrete integer index p in [0, K-1].
    2. Serving Queue Bin: Queue on incoming lanes currently possessing Green right-of-way.
       Categorized into:
       - 0: None (queue < 1.0)
       - 1: Low (1.0 <= queue < 5.0)
       - 2: High (queue >= 5.0)
    3. Waiting Queue Bin: Queue on incoming lanes facing Red (backpressure waiting to be served).
       Categorized into:
       - 0: None (queue < 1.0)
       - 1: Low (1.0 <= queue < 6.0)
       - 2: Moderate (6.0 <= queue < 15.0)
       - 3: Severe (queue >= 15.0)
    4. Elapsed Green Time Bin:
       - 0: Below min_green (cannot safely switch)
       - 1: Standard green window [min_green, max_effective_green)
       - 2: Saturated / Max green window (>= max_effective_green)

    Resulting State Space Size:
    - Cologne1: 8 phases * 3 serving * 4 waiting * 3 time = 288 states.
    - Ingolstadt1: 3 phases * 3 serving * 4 waiting * 3 time = 108 states.
    """

    def __init__(
        self,
        min_green: float = 5.0,
        max_effective_green: float = 25.0,
        serving_queue_thresholds: Tuple[float, float] = (1.0, 5.0),
        waiting_queue_thresholds: Tuple[float, float, float] = (1.0, 6.0, 15.0),
    ):
        self.min_green = float(min_green)
        self.max_effective_green = float(max_effective_green)
        self.serving_thresholds = tuple(float(x) for x in serving_queue_thresholds)
        self.waiting_thresholds = tuple(float(x) for x in waiting_queue_thresholds)

    def _get_green_lanes(self, obs: IntersectionObservation) -> Set[str]:
        """Identifies incoming lanes that have an active green signal in the current phase."""
        if not obs.green_phases or obs.current_phase >= len(obs.green_phases):
            return set()

        phase_str = obs.green_phases[obs.current_phase]
        str_len = len(phase_str)
        green_lanes = set()

        for link in obs.controlled_links:
            idx = link.signal_index
            if 0 <= idx < str_len and phase_str[idx] in ("G", "g"):
                if link.from_lane:
                    green_lanes.add(link.from_lane)

        return green_lanes

    def _bin_value(self, val: float, thresholds: Tuple[float, ...]) -> int:
        """Assigns a numeric value to an integer bin index based on monotonic thresholds."""
        for i, th in enumerate(thresholds):
            if val < th:
                return i
        return len(thresholds)

    def discretize(self, obs: IntersectionObservation) -> Tuple[int, int, int, int]:
        """Maps an IntersectionObservation to a 4-tuple discrete state representation.

        Returns:
            Tuple of (current_phase, serving_q_bin, waiting_q_bin, time_bin).
        """
        phase_idx = obs.current_phase

        green_lanes = self._get_green_lanes(obs)
        serving_q = 0.0
        waiting_q = 0.0

        for lane_id, q in obs.in_lane_queues.items():
            if lane_id in green_lanes:
                serving_q += q
            else:
                waiting_q += q

        serving_bin = self._bin_value(serving_q, self.serving_thresholds)
        waiting_bin = self._bin_value(waiting_q, self.waiting_thresholds)

        if obs.is_in_transition or obs.time_in_phase < self.min_green:
            time_bin = 0
        elif obs.time_in_phase < self.max_effective_green:
            time_bin = 1
        else:
            time_bin = 2

        return (phase_idx, serving_bin, waiting_bin, time_bin)

    def get_config(self) -> Dict[str, Any]:
        """Returns discretization configuration dictionary."""
        return {
            "min_green": self.min_green,
            "max_effective_green": self.max_effective_green,
            "serving_thresholds": list(self.serving_thresholds),
            "waiting_thresholds": list(self.waiting_thresholds),
        }
