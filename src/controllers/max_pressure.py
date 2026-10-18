"""Max-Pressure traffic signal controller."""

from typing import Any, Dict, List, Optional, Set
from src.controllers.base import BaseTrafficController, ControlledLink, IntersectionObservation


def compute_phase_pressure(
    phase_state: str,
    controlled_links: List[ControlledLink],
    in_lane_metrics: Dict[str, float],
    out_lane_metrics: Dict[str, float],
) -> float:
    """Computes traffic pressure for a green phase state (Inflow - Outflow)."""
    if not phase_state or not controlled_links:
        return 0.0

    total_outs: Dict[str, Set[str]] = {}
    active_ins: Dict[str, Set[str]] = {}
    active_outs: Set[str] = set()

    for link in controlled_links:
        if link.from_lane and link.to_lane:
            total_outs.setdefault(link.from_lane, set()).add(link.to_lane)
            if 0 <= link.signal_index < len(phase_state) and phase_state[link.signal_index] in ("G", "g", "s"):
                active_ins.setdefault(link.from_lane, set()).add(link.to_lane)
                active_outs.add(link.to_lane)

    inflow = sum(
        (len(outs) / len(total_outs[lane])) * in_lane_metrics.get(lane, 0.0)
        for lane, outs in active_ins.items()
        if lane in total_outs and total_outs[lane]
    )
    outflow = sum(out_lane_metrics.get(lane, 0.0) for lane in active_outs)
    return float(inflow - outflow)


class MaxPressureController(BaseTrafficController):
    """Max-Pressure traffic signal controller."""

    def __init__(
        self,
        min_green: float = 5.0,
        max_green: Optional[float] = 60.0,
        pressure_metric: str = "queue",
    ):
        super().__init__(name="MaxPressure")
        self.min_green = float(min_green) if min_green is not None else 5.0
        self.max_green = float(max_green) if max_green is not None else None
        self.pressure_metric = pressure_metric
        self.reset()

    def reset(self) -> None:
        """Resets controller internal states."""
        self._target_phases: Dict[str, int] = {}

    def select_action(
        self, observations: Dict[str, IntersectionObservation]
    ) -> Dict[str, int]:
        """Calculates pressure across candidate phases and chooses argmax phase."""
        actions: Dict[str, int] = {}

        for tls_id, obs in observations.items():
            if not obs.green_phases:
                actions[tls_id] = 0
                continue

            # Hold current phase during transitions or until min_green is satisfied
            if obs.is_in_transition:
                actions[tls_id] = self._target_phases.get(tls_id, obs.current_phase)
                continue
            if obs.time_in_phase < self.min_green:
                actions[tls_id] = obs.current_phase
                continue

            in_q = obs.in_lane_queues if self.pressure_metric == "queue" else obs.in_lane_vehicles
            out_q = obs.out_lane_queues if self.pressure_metric == "queue" else obs.out_lane_vehicles

            pressures = [
                compute_phase_pressure(p, obs.controlled_links, in_q, out_q)
                for p in obs.green_phases
            ]
            max_p = max(pressures)
            curr = obs.current_phase if 0 <= obs.current_phase < len(pressures) else 0

            # Switch away if max_green exceeded; otherwise hold current phase on ties
            if self.max_green is not None and obs.time_in_phase >= self.max_green and len(pressures) > 1:
                other_indices = [i for i in range(len(pressures)) if i != curr]
                chosen = max(other_indices, key=lambda i: pressures[i])
            elif abs(pressures[curr] - max_p) <= 1e-6:
                chosen = curr
            else:
                chosen = pressures.index(max_p)

            self._target_phases[tls_id] = chosen
            actions[tls_id] = chosen

        return actions

    def get_config(self) -> Dict[str, Any]:
        """Returns controller configuration dictionary."""
        return {
            "name": self.name,
            "min_green": self.min_green,
            "max_green": self.max_green,
            "pressure_metric": self.pressure_metric,
        }
