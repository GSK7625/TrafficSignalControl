"""Max Pressure traffic signal controller module (Varaiya 2013).

Calculates network movement pressure W(p) = sum_{(u,d) in M(p)} (x_u - x_d)
and actuates traffic lights dynamically while respecting safety constraints.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.controllers.base import BaseController, register_controller

__all__ = ["MaxPressureController"]


@register_controller("max_pressure")
class MaxPressureController(BaseController):
    """Max Pressure traffic signal controller with theoretical formulation and safety bounds.

    Attributes:
        green_phases: List of green phase signal state strings.
        movements: Sequence of controlled movement link objects or dicts.
        min_green: Minimum green phase duration in seconds (default: 20.0s).
        max_green: Maximum green phase duration before forced switch in seconds (default: 45.0s).
        yellow_time: Clearance yellow duration in seconds (default: 3.0s).
        tls_id: Traffic light identifier.
    """

    def __init__(
        self,
        green_phases: Sequence[Any],
        movements: Sequence[Any],
        min_green: float = 20.0,
        max_green: float = 45.0,
        yellow_time: float = 3.0,
        all_red_time: float = 1.0,
        max_red: float = 90.0,
        tls_id: Optional[str] = None,
        lane_vehicle_fn: Optional[Callable[[str], int]] = None,
    ) -> None:
        if min_green <= 0:
            raise ValueError(f"min_green must be positive, got {min_green}.")
        if max_green < min_green:
            raise ValueError(
                f"max_green ({max_green}) must be greater than or equal to min_green ({min_green})."
            )

        self.min_green: float = float(min_green)
        self.max_green: float = float(max_green)
        self.movements: List[Any] = list(movements) if movements is not None else []

        super().__init__(
            green_phases=green_phases,
            yellow_time=yellow_time,
            all_red_time=all_red_time,
            max_red=max_red,
            tls_id=tls_id,
            lane_vehicle_fn=lane_vehicle_fn,
        )

        # Precompute active movements per phase for performance
        self._phase_movements: List[List[Any]] = []
        for p_state in self.green_phases:
            active_m: List[Any] = []
            for m in self.movements:
                idx = getattr(m, "link_index", None) if not isinstance(m, dict) else m.get("link_index")
                if idx is not None and 0 <= idx < len(p_state):
                    if p_state[idx] in ("G", "g"):
                        active_m.append(m)
            self._phase_movements.append(active_m)

    def calculate_phase_pressures(self) -> Dict[int, float]:
        """Calculates pressure W(p) = sum_{(u,d) in M(p)} (x_u - x_d) for each green phase."""
        pressures: Dict[int, float] = {}
        lane_counts: Dict[str, int] = {}

        def get_count(lane: str) -> int:
            if not lane:
                return 0
            if lane not in lane_counts:
                lane_counts[lane] = self._get_vehicle_count(lane)
            return lane_counts[lane]

        for p_idx, moves in enumerate(self._phase_movements):
            pressure = 0.0
            for m in moves:
                u_lane = getattr(m, "from_lane", "") if not isinstance(m, dict) else m.get("from_lane", "")
                d_lane = getattr(m, "to_lane", "") if not isinstance(m, dict) else m.get("to_lane", "")
                pressure += float(get_count(u_lane) - get_count(d_lane))
            pressures[p_idx] = pressure

        return pressures

    def select_next_phase(self, force_switch: Optional[bool] = None) -> int:
        """Selects the next green phase according to Max Pressure rules and Starvation Watchdog."""
        if force_switch is None:
            force_switch = (self._time_in_phase >= self.max_green - 1e-6)

        pressures = self.calculate_phase_pressures()
        num_phases = len(self.green_phases)
        curr_p = self._current_phase_idx

        if num_phases <= 1:
            return curr_p

        alt_phases = [(curr_p + i) % num_phases for i in range(1, num_phases)]

        # Starvation Watchdog: Check if any competing phase reached max_red
        starved = [
            p for p in alt_phases
            if self.get_phase_red_time(p) >= self.max_red - 1e-6
        ]
        if starved:
            return max(starved, key=lambda p: (self.get_phase_red_time(p), pressures.get(p, 0.0)))

        if force_switch:
            return max(alt_phases, key=lambda p: pressures.get(p, 0.0))

        curr_pressure = pressures.get(curr_p, 0.0)
        max_p = max(pressures, key=lambda p: pressures.get(p, 0.0))
        max_val = pressures[max_p]

        if max_val <= 0.0 or max_val <= curr_pressure:
            return curr_p

        return max(alt_phases, key=lambda p: pressures.get(p, 0.0))

    def step(self, step_duration: float = 1.0) -> Tuple[str, bool]:
        """Advances controller by step_duration seconds."""
        if step_duration <= 0:
            raise ValueError(f"step_duration must be positive, got {step_duration}.")

        tolerance = 1e-6

        if self._is_yellow:
            if self._time_in_phase >= self.yellow_time - tolerance:
                self._is_yellow = False
                self._time_in_phase = 0.0
                if self.all_red_time > 0:
                    self._is_all_red = True
                else:
                    self._is_all_red = False
                    self._current_phase_idx = self._next_phase_idx
        elif self._is_all_red:
            if self._time_in_phase >= self.all_red_time - tolerance:
                self._is_all_red = False
                self._current_phase_idx = self._next_phase_idx
                self._time_in_phase = 0.0
        else:
            if self._time_in_phase >= self.min_green - tolerance:
                has_starvation = any(
                    self.get_phase_red_time(p) >= self.max_red - tolerance
                    for p in range(len(self.green_phases))
                    if p != self._current_phase_idx
                )
                force_switch = (self._time_in_phase >= self.max_green - tolerance) or has_starvation
                next_p = self.select_next_phase(force_switch=force_switch)
                if next_p != self._current_phase_idx:
                    self._next_phase_idx = next_p
                    self._time_in_phase = 0.0
                    if self.yellow_time > 0:
                        self._is_yellow = True
                    elif self.all_red_time > 0:
                        self._is_all_red = True
                    else:
                        self._current_phase_idx = next_p

        self._update_red_timers(step_duration)

        current_state = self.get_current_state()
        phase_changed = (self._last_state != current_state)
        self._last_state = current_state

        self._time_in_phase += step_duration
        self._total_time += step_duration
        self._step_count += 1

        return current_state, phase_changed
