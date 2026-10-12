"""Max-Pressure traffic signal controller."""

from typing import Any, Dict, List, Optional, Set

from src.controllers.base import BaseTrafficController, ControlledLink, IntersectionObservation


def compute_phase_pressure(
    phase_state: str,
    controlled_links: List[ControlledLink],
    in_lane_metrics: Dict[str, float],
    out_lane_metrics: Dict[str, float],
) -> float:
    """Computes traffic pressure for a candidate green phase state.

    Mathematical Formulation:
        Pressure(p) = sum_{i in In(p)} [ (|J(i, p)| / |Out(i)|) * q_in(i) ]
                      - sum_{j in Out(p)} q_out(j)

    Where:
        - In(p): Set of distinct incoming lanes with at least one green movement in phase p.
        - J(i, p): Set of distinct outgoing lanes connected to lane i that are served in phase p.
        - Out(i): Total set of distinct outgoing lanes connected to lane i across all movements.
        - Out(p): Set of distinct outgoing lanes receiving traffic from any green movement in phase p.
        - q_in(i): Queue length or vehicle count on incoming lane i.
        - q_out(j): Queue length or vehicle count on downstream outgoing lane j.

    Key Properties:
        1. Prevents double-counting: an incoming lane with multiple outgoing movements
           contributes at most 1.0 * q_in(i) when all movements are green.
        2. Symmetrical backpressure: downstream congestion exerts backpressure (-q_out(j)),
           deterring the selection of phases that spill back into already congested exits.
        3. Boundary-safe: exit lanes leaving the network or untracked have q_out = 0.
        4. Handles multi-link connections and varying signal state lengths.
    """
    if not phase_state or not controlled_links:
        return 0.0

    # 1. Map total outgoing connectivity per incoming lane across all junction links
    total_out_per_in: Dict[str, Set[str]] = {}
    for link in controlled_links:
        if link.from_lane:
            if link.from_lane not in total_out_per_in:
                total_out_per_in[link.from_lane] = set()
            if link.to_lane:
                total_out_per_in[link.from_lane].add(link.to_lane)

    # 2. Identify links served in this phase (signal character in 'G' or 'g')
    active_in_lanes: Dict[str, Set[str]] = {}
    active_out_lanes: Set[str] = set()

    state_len = len(phase_state)
    for link in controlled_links:
        idx = link.signal_index
        if 0 <= idx < state_len and phase_state[idx] in ("G", "g"):
            from_l = link.from_lane
            to_l = link.to_lane

            if from_l:
                if from_l not in active_in_lanes:
                    active_in_lanes[from_l] = set()
                if to_l:
                    active_in_lanes[from_l].add(to_l)

            if to_l:
                active_out_lanes.add(to_l)

    # 3. Calculate incoming queue contribution
    inflow_pressure = 0.0
    for from_l, served_outs in active_in_lanes.items():
        q_in = in_lane_metrics.get(from_l, 0.0)
        total_outs = total_out_per_in.get(from_l, set())
        ratio = (len(served_outs) / len(total_outs)) if total_outs else 1.0
        inflow_pressure += ratio * q_in

    # 4. Calculate downstream backpressure
    outflow_backpressure = 0.0
    for to_l in active_out_lanes:
        q_out = out_lane_metrics.get(to_l, 0.0)
        outflow_backpressure += q_out

    return float(inflow_pressure - outflow_backpressure)


class MaxPressureController(BaseTrafficController):
    """Max-Pressure traffic signal controller.

    Dynamically selects the green phase with maximum aggregate queue pressure,
    subject to minimum green time constraints and safe yellow/all-red clearances.
    """

    def __init__(
        self,
        min_green: float = 5.0,
        pressure_metric: str = "queue",
    ):
        """Initializes MaxPressureController.

        Args:
            min_green: Minimum duration in seconds a green phase must be held before switching.
            pressure_metric: Metric to compute pressure ('queue' for halting vehicles, 'vehicle_count' for all vehicles).
        """
        super().__init__(name="MaxPressure")
        if min_green < 0:
            raise ValueError(f"min_green must be non-negative, got {min_green}")
        if pressure_metric not in ("queue", "vehicle_count"):
            raise ValueError(f"Unknown pressure_metric '{pressure_metric}', must be 'queue' or 'vehicle_count'")

        self.min_green = float(min_green)
        self.pressure_metric = pressure_metric
        self._target_phases: Dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        """Resets controller internal states and history."""
        self.history.clear()
        self._target_phases.clear()

    def select_action(
        self, observations: Dict[str, IntersectionObservation]
    ) -> Dict[str, int]:
        """Calculates pressure across candidate phases and chooses argmax phase per TLS."""
        actions: Dict[str, int] = {}

        for tls_id, obs in observations.items():
            green_phases = obs.green_phases
            if not green_phases:
                actions[tls_id] = 0
                continue

            current_phase = obs.current_phase
            target_phase = self._target_phases.get(tls_id, current_phase)

            # If intersection is in clearance transition (YELLOW or ALL_RED), maintain target phase
            if obs.is_in_transition:
                actions[tls_id] = target_phase
                continue

            # If min_green constraint is not yet satisfied, hold current green phase
            if obs.time_in_phase < self.min_green:
                actions[tls_id] = current_phase
                continue

            # Select metrics dictionary based on configuration
            in_metrics = obs.in_lane_queues if self.pressure_metric == "queue" else obs.in_lane_vehicles
            out_metrics = obs.out_lane_queues if self.pressure_metric == "queue" else obs.out_lane_vehicles

            # Calculate pressure for every selectable green phase
            pressures: Dict[int, float] = {}
            for p_idx, p_state in enumerate(green_phases):
                pressures[p_idx] = compute_phase_pressure(
                    phase_state=p_state,
                    controlled_links=obs.controlled_links,
                    in_lane_metrics=in_metrics,
                    out_lane_metrics=out_metrics,
                )

            max_p = max(pressures.values())
            # Find all phases that achieve maximum pressure (within floating tolerance)
            max_candidates = [
                idx for idx, p_val in pressures.items() if abs(p_val - max_p) <= 1e-6
            ]

            # Tie-breaking rule:
            # If current phase is in max_candidates, keep current phase to prevent signal chatter.
            # Otherwise, pick lowest index candidate.
            if current_phase in max_candidates:
                chosen_phase = current_phase
            else:
                chosen_phase = min(max_candidates)

            self._target_phases[tls_id] = chosen_phase
            actions[tls_id] = chosen_phase

            # Log decision audit trail
            self.log_decision(
                tls_id=tls_id,
                sim_time=obs.sim_time,
                step=obs.step,
                action=chosen_phase,
                details={
                    "pressures": pressures,
                    "max_pressure": max_p,
                    "current_phase": current_phase,
                    "candidates": max_candidates,
                    "switched": (chosen_phase != current_phase),
                },
            )

        return actions

    def get_config(self) -> Dict[str, Any]:
        """Returns controller configuration dictionary."""
        return {
            "name": self.name,
            "min_green": self.min_green,
            "pressure_metric": self.pressure_metric,
        }
