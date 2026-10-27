"""Fixed-time traffic signal controller module."""

from typing import Any, List, Optional, Sequence, Tuple


def build_yellow_state(
    curr_state: str,
    next_state: Optional[str] = None,
    only_changing: bool = False,
) -> str:
    """Builds a safe yellow transition state for clearance before the next green phase.

    By default, all active green signals ('G' or 'g') in curr_state transition to
    yellow ('y') to ensure safe clearance of the intersection.
    If only_changing=True and next_state is provided, only signals that lose right-of-way
    (i.e., green in curr_state but not green in next_state) transition to yellow.
    Red signals ('r', 'R') and special signals ('s') remain unchanged.

    Args:
        curr_state: Current traffic signal state string.
        next_state: Optional upcoming traffic signal state string.
        only_changing: If True and next_state is provided, only signals changing
            from green to non-green will turn yellow.

    Returns:
        The computed yellow transition state string.
    """
    yellow_chars: List[str] = []
    for i, char in enumerate(curr_state):
        if char in ("G", "g"):
            if only_changing and next_state is not None:
                if i < len(next_state) and next_state[i] in ("G", "g"):
                    yellow_chars.append(char)
                else:
                    yellow_chars.append("y")
            else:
                yellow_chars.append("y")
        else:
            yellow_chars.append(char)
    return "".join(yellow_chars)


class FixedTimeController:
    """Fixed-time traffic signal controller with configurable cycle and safe yellow clearance.

    Operates on a fixed cycle duration per green phase with an intermediate yellow clearance
    phase automatically generated before switching to the next green phase. Loops infinitely.

    Attributes:
        green_phases: List of green phase state strings.
        green_time: Duration of each green phase in seconds (default: 30.0s).
        yellow_time: Duration of yellow clearance phase in seconds (default: 3.0s).
        tls_id: Identifier of the traffic light signal.
    """

    def __init__(
        self,
        green_phases: Sequence[Any],
        green_time: float = 30.0,
        yellow_time: float = 3.0,
        tls_id: Optional[str] = None,
    ) -> None:
        if not green_phases:
            raise ValueError("green_phases must contain at least one phase.")
        if green_time <= 0:
            raise ValueError(f"green_time must be positive, got {green_time}.")
        if yellow_time < 0:
            raise ValueError(f"yellow_time must be non-negative, got {yellow_time}.")

        self.green_phases: List[str] = [
            p.state if hasattr(p, "state") else str(p)
            for p in green_phases
        ]
        self.green_time: float = float(green_time)
        self.yellow_time: float = float(yellow_time)
        self.tls_id: Optional[str] = tls_id

        # Internal state
        self._current_phase_idx: int = 0
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
        """Number of green phases."""
        return len(self.green_phases)

    @property
    def current_phase(self) -> int:
        """Current phase index."""
        return self._current_phase_idx

    @property
    def is_yellow(self) -> bool:
        """True if controller is in yellow clearance interval."""
        return self._is_yellow

    @property
    def time_in_phase(self) -> float:
        """Elapsed time in current phase/interval."""
        return self._time_in_phase

    @property
    def total_time(self) -> float:
        """Total accumulated simulation time."""
        return self._total_time

    @property
    def step_count(self) -> int:
        """Total number of steps taken."""
        return self._step_count

    def get_current_phase(self) -> int:
        """Returns the current green phase index."""
        return self._current_phase_idx

    def is_in_yellow(self) -> bool:
        """Returns True if the controller is currently in yellow clearance."""
        return self._is_yellow

    def get_current_state(self) -> str:
        """Returns the active signal state string for the current step."""
        curr_green = self.green_phases[self._current_phase_idx]
        if self._is_yellow:
            next_idx = (self._current_phase_idx + 1) % len(self.green_phases)
            next_green = self.green_phases[next_idx]
            return build_yellow_state(curr_green, next_green)
        return curr_green

    def reset(self) -> None:
        """Resets the controller to initial state (phase 0, green, 0 elapsed time)."""
        self._current_phase_idx = 0
        self._is_yellow = False
        self._time_in_phase = 0.0
        self._total_time = 0.0
        self._step_count = 0
        self._last_state = None

    def step(self, step_duration: float = 1.0) -> Tuple[str, bool]:
        """Advances the controller by step_duration seconds.

        Args:
            step_duration: Time step duration in seconds (default: 1.0).

        Returns:
            Tuple[str, bool]: (current_state, phase_changed)
                - current_state: The traffic signal state string to apply for this step.
                - phase_changed: True if this step marks a transition to a new signal state.
        """
        if step_duration <= 0:
            raise ValueError(f"step_duration must be positive, got {step_duration}.")

        tolerance = 1e-6

        # Check if the previous interval has completed and transition is due
        if not self._is_yellow:
            if self._time_in_phase >= self.green_time - tolerance:
                if self.yellow_time > 0:
                    self._is_yellow = True
                    self._time_in_phase = 0.0
                else:
                    self._current_phase_idx = (self._current_phase_idx + 1) % len(self.green_phases)
                    self._time_in_phase = 0.0
        else:
            if self._time_in_phase >= self.yellow_time - tolerance:
                self._is_yellow = False
                self._time_in_phase = 0.0
                self._current_phase_idx = (self._current_phase_idx + 1) % len(self.green_phases)

        # Get state for this step
        current_state = self.get_current_state()
        phase_changed = (self._last_state != current_state)
        self._last_state = current_state

        # Advance elapsed time for current step
        self._time_in_phase += step_duration
        self._total_time += step_duration
        self._step_count += 1

        return current_state, phase_changed
