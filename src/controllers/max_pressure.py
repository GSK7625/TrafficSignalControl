"""Max Pressure traffic signal controller module (Varaiya 2013).

Calculates network movement pressure W(p) = sum_{(u,d) in M(p)} (x_u - x_d)
and actuates traffic lights dynamically while respecting safety constraints:
- min_green: minimum green time to prevent rapid phase flickering
- yellow_time: clearance interval between opposing green phases
- max_green: maximum green time to prevent movement starvation
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from src.controllers.fixed_time import build_yellow_state


class MaxPressureController:
    """Max Pressure traffic signal controller with theoretical formulation and safety bounds.

    Attributes:
        green_phases: List of green phase signal state strings.
        movements: Sequence of controlled movement link objects or dicts.
        min_green: Minimum green phase duration in seconds (default: 10.0s).
        max_green: Maximum green phase duration before forced switch in seconds (default: 60.0s).
        yellow_time: Clearance yellow duration in seconds (default: 3.0s).
        tls_id: Traffic light identifier.
    """

    def __init__(
        self,
        green_phases: Sequence[Any],
        movements: Sequence[Any],
        min_green: float = 10.0,
        max_green: float = 60.0,
        yellow_time: float = 3.0,
        tls_id: Optional[str] = None,
        lane_vehicle_fn: Optional[Callable[[str], int]] = None,
    ) -> None:
        if not green_phases:
            raise ValueError("green_phases must contain at least one phase.")
        if min_green <= 0:
            raise ValueError(f"min_green must be positive, got {min_green}.")
        if max_green < min_green:
            raise ValueError(
                f"max_green ({max_green}) must be greater than or equal to min_green ({min_green})."
            )
        if yellow_time < 0:
            raise ValueError(f"yellow_time must be non-negative, got {yellow_time}.")

        self.green_phases: List[str] = [
            p.state if hasattr(p, "state") else str(p) for p in green_phases
        ]
        self.movements: List[Any] = list(movements) if movements is not None else []
        self.min_green: float = float(min_green)
        self.max_green: float = float(max_green)
        self.yellow_time: float = float(yellow_time)
        self.tls_id: Optional[str] = tls_id
        self._lane_vehicle_fn: Optional[Callable[[str], int]] = lane_vehicle_fn

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

        # Internal state
        self._current_phase_idx: int = 0
        self._next_phase_idx: int = 0
        self._is_yellow: bool = False
        self._time_in_phase: float = 0.0
        self._total_time: float = 0.0
        self._step_count: int = 0
        self._last_state: Optional[str] = None

    @staticmethod
    def build_yellow_state(
        curr_state: str,
        next_state: Optional[str] = None,
        only_changing: bool = False,
    ) -> str:
        """Expose build_yellow_state as a static method for convenience."""
        return build_yellow_state(curr_state, next_state, only_changing=only_changing)

    @property
    def num_phases(self) -> int:
        """Total number of green phases."""
        return len(self.green_phases)

    @property
    def current_phase(self) -> int:
        """Current green phase index."""
        return self._current_phase_idx

    @property
    def is_yellow(self) -> bool:
        """True if controller is currently in yellow clearance interval."""
        return self._is_yellow

    @property
    def time_in_phase(self) -> float:
        """Elapsed time in current interval (green or yellow)."""
        return self._time_in_phase

    @property
    def total_time(self) -> float:
        """Total accumulated simulation time in seconds."""
        return self._total_time

    @property
    def step_count(self) -> int:
        """Total number of control steps taken."""
        return self._step_count

    def get_current_phase(self) -> int:
        """Returns the current green phase index."""
        return self._current_phase_idx

    def is_in_yellow(self) -> bool:
        """Returns True if controller is currently in yellow clearance interval."""
        return self._is_yellow

    def get_current_state(self) -> str:
        """Returns the active signal state string for the current step."""
        curr_green = self.green_phases[self._current_phase_idx]
        if self._is_yellow:
            next_green = self.green_phases[self._next_phase_idx]
            return build_yellow_state(curr_green, next_green)
        return curr_green

    def _get_vehicle_count(self, lane_id: str) -> int:
        """Retrieves vehicle count on a given lane."""
        if not lane_id:
            return 0
        if self._lane_vehicle_fn is not None:
            try:
                return int(self._lane_vehicle_fn(lane_id))
            except Exception:
                return 0
        try:
            import traci
            if traci.isLoaded():
                return int(traci.lane.getLastStepVehicleNumber(lane_id))
        except (ImportError, AttributeError, Exception):
            pass
        return 0

    def calculate_phase_pressures(self) -> Dict[int, float]:
        """Calculates pressure W(p) = sum_{(u,d) in M(p)} (x_u - x_d) for each green phase.

        Returns:
            Dictionary mapping phase index to calculated pressure value.
        """
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
                u_count = get_count(u_lane)
                d_count = get_count(d_lane)
                pressure += float(u_count - d_count)
            pressures[p_idx] = pressure

        return pressures

    def select_next_phase(self, force_switch: Optional[bool] = None) -> int:
        """Selects the next green phase according to Max Pressure rules.

        If force_switch is None, automatically forces switch if time_in_phase >= max_green.
        If force_switch is True, ignores current phase and selects the alternative phase
        with the highest pressure.
        If not force_switch, stays on current phase if max pressure is non-positive or
        less than/equal to current phase pressure.

        Args:
            force_switch: Explicitly force switching away from current phase.

        Returns:
            Chosen green phase index.
        """
        if force_switch is None:
            force_switch = (self._time_in_phase >= self.max_green - 1e-6)

        pressures = self.calculate_phase_pressures()
        num_phases = len(self.green_phases)
        curr_p = self._current_phase_idx

        if num_phases <= 1:
            return curr_p

        # Cyclic alternative phases starting from (curr_p + 1) % num_phases
        alt_phases = [(curr_p + i) % num_phases for i in range(1, num_phases)]

        if force_switch:
            # Must transition to another phase with highest pressure among eligible alternatives
            return max(alt_phases, key=lambda p: pressures.get(p, 0.0))

        curr_pressure = pressures.get(curr_p, 0.0)
        max_p = max(pressures, key=lambda p: pressures.get(p, 0.0))
        max_val = pressures[max_p]

        # If max pressure is non-positive or equal to current phase pressure, stay
        if max_val <= 0.0 or max_val <= curr_pressure:
            return curr_p

        # Select the best alternative phase
        return max(alt_phases, key=lambda p: pressures.get(p, 0.0))

    def step(self, step_duration: float = 1.0) -> Tuple[str, bool]:
        """Advances controller by step_duration seconds.

        Args:
            step_duration: Control step duration in seconds (default: 1.0s).

        Returns:
            Tuple[str, bool]: (current_state, phase_changed)
        """
        if step_duration <= 0:
            raise ValueError(f"step_duration must be positive, got {step_duration}.")

        tolerance = 1e-6

        if self._is_yellow:
            # Check if yellow clearance interval has completed
            if self._time_in_phase >= self.yellow_time - tolerance:
                self._is_yellow = False
                self._current_phase_idx = self._next_phase_idx
                self._time_in_phase = 0.0
        else:
            # In green phase: check if min_green requirement is met
            if self._time_in_phase >= self.min_green - tolerance:
                force_switch = (self._time_in_phase >= self.max_green - tolerance)
                next_p = self.select_next_phase(force_switch=force_switch)
                if next_p != self._current_phase_idx:
                    self._next_phase_idx = next_p
                    if self.yellow_time > 0:
                        self._is_yellow = True
                        self._time_in_phase = 0.0
                    else:
                        self._current_phase_idx = next_p
                        self._time_in_phase = 0.0

        current_state = self.get_current_state()
        phase_changed = (self._last_state != current_state)
        self._last_state = current_state

        self._time_in_phase += step_duration
        self._total_time += step_duration
        self._step_count += 1

        return current_state, phase_changed

    def reset(self) -> None:
        """Resets the controller to initial state."""
        self._current_phase_idx = 0
        self._next_phase_idx = 0
        self._is_yellow = False
        self._time_in_phase = 0.0
        self._total_time = 0.0
        self._step_count = 0
        self._last_state = None
