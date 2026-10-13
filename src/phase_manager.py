"""Safe Traffic Light Phase Manager.

Ensures minimum green times, enforces safe yellow clearance intervals,
and prevents abrupt phase transitions across traffic control algorithms.
"""

from enum import Enum
from typing import List, Optional, Union
import warnings


class PhaseState(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ALL_RED = "ALL_RED"


def build_yellow_state(current_state: str, target_state: str) -> str:
    """Builds an intermediate yellow clearance state between current and target signal states.

    Rules:
    - Any link that is currently Green ('G', 'g') and will become Red ('r', 's', 'u')
      must switch to Yellow ('y').
    - Any link that remains Green retains its green state ('G', 'g').
    - All other links remain as they are or switch to Red.
    """
    if len(current_state) != len(target_state):
        raise ValueError(
            f"State lengths do not match: current({len(current_state)}) vs target({len(target_state)})"
        )

    yellow_chars = []
    for c_curr, c_targ in zip(current_state, target_state):
        if c_curr in ("G", "g") and c_targ in ("r", "s", "u", "O", "o"):
            yellow_chars.append("y")
        elif c_curr in ("G", "g") and c_targ in ("G", "g"):
            yellow_chars.append(c_curr)
        elif c_curr == "y":
            yellow_chars.append("y")
        else:
            yellow_chars.append(c_targ if c_targ in ("r", "s", "u", "O", "o") else "r")

    return "".join(yellow_chars)


class SafePhaseManager:
    """Manages signal phase transitions safely for an individual traffic light intersection."""

    def __init__(
        self,
        tls_id: str,
        green_phases: List[str],
        yellow_duration: float = 3.0,
        all_red_duration: float = 0.0,
        min_green: float = 5.0,
        initial_phase: int = 0,
    ):
        """Initializes SafePhaseManager.

        Args:
            tls_id: Traffic light intersection ID in SUMO.
            green_phases: List of valid green phase signal strings.
            yellow_duration: Clearance yellow time in seconds.
            all_red_duration: Optional clearance all-red time in seconds (default: 0.0).
            min_green: Minimum required green duration before a switch can occur.
            initial_phase: Starting green phase index.
        """
        if not green_phases:
            raise ValueError(f"green_phases cannot be empty for TLS '{tls_id}'")
        if yellow_duration <= 0:
            raise ValueError("yellow_duration must be positive")
        if all_red_duration < 0:
            raise ValueError("all_red_duration must be non-negative")
        if min_green < 0:
            raise ValueError("min_green must be non-negative")
        if initial_phase < 0 or initial_phase >= len(green_phases):
            raise IndexError(f"initial_phase {initial_phase} out of bounds [0, {len(green_phases)-1}]")

        self.tls_id = tls_id
        self.green_phases = green_phases
        self.yellow_duration = yellow_duration
        self.all_red_duration = all_red_duration
        self.min_green = min_green

        self._current_phase: int = initial_phase
        self._target_phase: int = initial_phase
        self._state_type: PhaseState = PhaseState.GREEN
        self._current_signal_string: str = self.green_phases[initial_phase]
        self._time_in_phase: float = 0.0
        self._yellow_time_remaining: float = 0.0
        self._all_red_time_remaining: float = 0.0

        # Phase usage tracking
        self.phase_switch_count: int = 0
        self.phase_durations: dict[int, float] = {i: 0.0 for i in range(len(green_phases))}
        self.total_transition_time: float = 0.0
        self.actual_yellow_time: float = 0.0

    def get_current_phase(self) -> int:
        """Returns the current active green phase index."""
        return self._current_phase

    def get_target_phase(self) -> int:
        """Returns the target phase index (relevant during yellow transition)."""
        return self._target_phase

    def get_current_state(self) -> str:
        """Returns the current signal string (e.g. 'GGGrrr' or 'yyyrrr')."""
        return self._current_signal_string

    def is_in_transition(self) -> bool:
        """Returns True if the signal is currently in yellow or all-red transition."""
        return self._state_type in (PhaseState.YELLOW, PhaseState.ALL_RED)

    def time_in_current_phase(self) -> float:
        """Returns elapsed time in current green phase (or yellow transition)."""
        return self._time_in_phase

    def sync_external_phase(
        self,
        phase_index: Optional[int] = None,
        signal_state: Optional[str] = None,
        step_duration: float = 1.0,
    ) -> None:
        """Synchronizes phase tracking with an external controller (e.g. native SUMO program).

        Maps external signal states to internal green phase indices [0, len(green_phases)-1]
        and appropriately classifies yellow/all-red clearance intervals without polluting
        green phase metrics or setting out-of-range phase indices.
        """
        if signal_state is not None:
            self._current_signal_string = signal_state
            if signal_state in self.green_phases:
                green_idx = self.green_phases.index(signal_state)
                if green_idx != self._current_phase:
                    self._current_phase = green_idx
                    self._target_phase = green_idx
                    self._time_in_phase = 0.0
                    self.phase_switch_count += 1
                elif self._state_type != PhaseState.GREEN:
                    self._time_in_phase = 0.0
                self._state_type = PhaseState.GREEN
                self._time_in_phase += step_duration
                self.phase_durations[green_idx] = (
                    self.phase_durations.get(green_idx, 0.0) + step_duration
                )
            else:
                self.total_transition_time += step_duration
                if any(c in signal_state for c in ("y", "Y")):
                    self._state_type = PhaseState.YELLOW
                    self.actual_yellow_time += step_duration
                else:
                    self._state_type = PhaseState.ALL_RED
                self._time_in_phase += step_duration
        elif phase_index is not None:
            if 0 <= phase_index < len(self.green_phases):
                if phase_index != self._current_phase:
                    self._current_phase = phase_index
                    self._target_phase = phase_index
                    self._time_in_phase = 0.0
                    self.phase_switch_count += 1
                elif self._state_type != PhaseState.GREEN:
                    self._time_in_phase = 0.0
                self._state_type = PhaseState.GREEN
                self._current_signal_string = self.green_phases[phase_index]
                self._time_in_phase += step_duration
                self.phase_durations[phase_index] = (
                    self.phase_durations.get(phase_index, 0.0) + step_duration
                )
            else:
                self._state_type = PhaseState.YELLOW
                self.total_transition_time += step_duration
                self.actual_yellow_time += step_duration
                self._time_in_phase += step_duration

    def request_phase(self, target_phase_idx: int) -> bool:
        """Requests switching to a new green phase index.

        Returns:
            True if request was accepted (either already in that phase, or yellow transition initiated).
            False if rejected (due to min_green not met, or currently in transition).
        """
        if target_phase_idx < 0 or target_phase_idx >= len(self.green_phases):
            raise IndexError(
                f"Target phase index {target_phase_idx} out of range [0, {len(self.green_phases)-1}]"
            )

        # If already in the target phase and not transitioning, accept immediately
        if target_phase_idx == self._current_phase and self._state_type == PhaseState.GREEN:
            return True

        # If currently transitioning, cannot accept new change
        if self._state_type in (PhaseState.YELLOW, PhaseState.ALL_RED):
            return False

        # If min green constraint not met, reject change
        if self._time_in_phase < self.min_green:
            return False

        # Initiate yellow transition
        self._target_phase = target_phase_idx
        current_state = self.green_phases[self._current_phase]
        target_state = self.green_phases[target_phase_idx]

        if self.all_red_duration > 1e-6:
            # When an ALL-RED interval is configured, ALL links currently showing Green
            # must transition through Yellow, because in the subsequent all-red state
            # every link will become Red.
            all_red_target = "r" * len(current_state)
            self._current_signal_string = build_yellow_state(current_state, all_red_target)
        else:
            self._current_signal_string = build_yellow_state(current_state, target_state)

        self._state_type = PhaseState.YELLOW
        self._yellow_time_remaining = self.yellow_duration
        self._all_red_time_remaining = 0.0
        self._time_in_phase = 0.0
        self.phase_switch_count += 1
        return True

    def update(self, step_duration: float = 1.0, traci_module=None) -> None:
        """Advances internal timing by step_duration and updates SUMO TLS if traci_module is provided."""
        if self._state_type in (PhaseState.YELLOW, PhaseState.ALL_RED):
            self.total_transition_time += step_duration
            if any(c in self._current_signal_string for c in ("y", "Y")):
                self.actual_yellow_time += step_duration

        if self._state_type == PhaseState.YELLOW:
            self._time_in_phase += step_duration
            self._yellow_time_remaining -= step_duration

            if self._yellow_time_remaining <= 1e-6:
                if self.all_red_duration > 1e-6:
                    # Switch to intermediate ALL_RED clearance
                    self._state_type = PhaseState.ALL_RED
                    self._current_signal_string = "r" * len(self.green_phases[self._target_phase])
                    self._all_red_time_remaining = self.all_red_duration
                    self._time_in_phase = 0.0
                    self._yellow_time_remaining = 0.0
                else:
                    # Transition complete: switch directly to target green phase
                    self._current_phase = self._target_phase
                    self._current_signal_string = self.green_phases[self._current_phase]
                    self._state_type = PhaseState.GREEN
                    self._time_in_phase = 0.0
                    self._yellow_time_remaining = 0.0
        elif self._state_type == PhaseState.ALL_RED:
            self._time_in_phase += step_duration
            self._all_red_time_remaining -= step_duration

            if self._all_red_time_remaining <= 1e-6:
                # All-red complete: switch to target green phase
                self._current_phase = self._target_phase
                self._current_signal_string = self.green_phases[self._current_phase]
                self._state_type = PhaseState.GREEN
                self._time_in_phase = 0.0
                self._all_red_time_remaining = 0.0
        else:
            self._time_in_phase += step_duration
            self.phase_durations[self._current_phase] = (
                self.phase_durations.get(self._current_phase, 0.0) + step_duration
            )

        if traci_module is not None:
            self.apply_to_traci(traci_module)

    def apply_to_traci(self, traci_module) -> None:
        """Sets the active signal string on the SUMO traffic light via TraCI."""
        try:
            traci_module.trafficlight.setRedYellowGreenState(
                self.tls_id, self._current_signal_string
            )
        except Exception as e:
            raise RuntimeError(f"Failed to set state for TLS '{self.tls_id}': {e}") from e
