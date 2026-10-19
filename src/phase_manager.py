"""Traffic Light Phase Manager (SafePhaseManager).

Quản lý chuyển pha an toàn: đảm bảo min_green, thời gian đèn vàng dọn nút (yellow clearance),
và toàn đỏ (all-red) nếu có.
"""

from enum import Enum
import math
from typing import List, Optional


class PhaseState(Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    ALL_RED = "ALL_RED"


def build_yellow_state(current_state: str, target_state: str) -> str:
    """Tạo trạng thái đèn vàng chuyển tiếp an toàn giữa 2 pha."""
    if len(current_state) != len(target_state):
        raise ValueError(
            f"State lengths do not match: current({len(current_state)}) vs target({len(target_state)})"
        )
    res = []
    for c, t in zip(current_state, target_state):
        if c in "Ggs" and (t in "ruOo" or (c in "Gg" and t == "s")):
            res.append("y")
        elif c in "Gg" and t in "Gg":
            res.append("g" if "g" in (c, t) else "G")
        elif c in "sy":
            res.append(c)
        else:
            res.append("r")
    return "".join(res)


class SafePhaseManager:
    """Quản lý chu kỳ chuyển pha cho từng nút giao thông."""

    def __init__(
        self,
        tls_id: str,
        green_phases: List[str],
        yellow_duration: float = 3.0,
        all_red_duration: float = 0.0,
        min_green: float = 5.0,
        max_green: Optional[float] = None,
        initial_phase: int = 0,
    ):
        if not green_phases:
            raise ValueError("green_phases cannot be empty")
        if yellow_duration <= 0:
            raise ValueError("yellow_duration must be positive")
        if all_red_duration < 0:
            raise ValueError("all_red_duration must be non-negative")
        if not math.isfinite(min_green) or min_green < 0:
            raise ValueError(f"min_green must be non-negative and finite, got {min_green}")
        if max_green is not None:
            if not math.isfinite(max_green):
                raise ValueError(f"max_green must be finite, got {max_green}")
            if max_green <= 0:
                raise ValueError(f"max_green must be positive, got {max_green}")
            if max_green < min_green:
                raise ValueError(f"max_green must be >= min_green ({max_green} < {min_green})")
        if not (0 <= initial_phase < len(green_phases)):
            raise IndexError("initial_phase out of bounds")

        self.tls_id = tls_id
        self.green_phases = green_phases
        self.yellow_duration = yellow_duration
        self.all_red_duration = all_red_duration
        self.min_green = min_green
        self.max_green = max_green

        self._current_phase: int = initial_phase
        self._target_phase: int = initial_phase
        self._state_type: PhaseState = PhaseState.GREEN
        self._current_signal_string: str = green_phases[initial_phase]
        self._time_in_phase: float = 0.0
        self._yellow_time_remaining: float = 0.0
        self._all_red_time_remaining: float = 0.0

        # Thống kê hoạt động
        self.phase_switch_count: int = 0
        self.phase_durations: dict[int, float] = {i: 0.0 for i in range(len(green_phases))}
        self.total_transition_time: float = 0.0
        self.actual_yellow_time: float = 0.0

    def get_current_phase(self) -> int:
        return self._current_phase

    def get_target_phase(self) -> int:
        return self._target_phase

    def get_current_state(self) -> str:
        return self._current_signal_string

    def is_in_transition(self) -> bool:
        return self._state_type != PhaseState.GREEN

    def time_in_current_phase(self) -> float:
        return self._time_in_phase

    def is_max_green_exceeded(self) -> bool:
        return (
            self.max_green is not None
            and self._state_type == PhaseState.GREEN
            and self._time_in_phase >= self.max_green
        )

    def valid_actions(self) -> List[int]:
        """The single source of timing rules for controller and environment masks."""
        if self.is_in_transition():
            return [self._target_phase]
        if self._time_in_phase < self.min_green:
            return [self._current_phase]
        actions = list(range(len(self.green_phases)))
        if self.is_max_green_exceeded() and len(actions) > 1:
            actions.remove(self._current_phase)
        return actions

    def request_phase(self, target_phase_idx: int) -> bool:
        if not (0 <= target_phase_idx < len(self.green_phases)):
            raise IndexError(f"Target phase index {target_phase_idx} out of range")

        if self.is_in_transition():
            return False

        if target_phase_idx == self._current_phase:
            if self.is_max_green_exceeded() and len(self.green_phases) > 1:
                return False
            return True

        if self._time_in_phase < self.min_green:
            return False

        self._target_phase = target_phase_idx
        curr_state = self.green_phases[self._current_phase]
        targ_state = (
            "r" * len(curr_state)
            if self.all_red_duration > 1e-6
            else self.green_phases[target_phase_idx]
        )

        self._current_signal_string = build_yellow_state(curr_state, targ_state)
        self._state_type = PhaseState.YELLOW
        self._yellow_time_remaining = self.yellow_duration
        self._all_red_time_remaining = 0.0
        self._time_in_phase = 0.0
        self.phase_switch_count += 1
        return True

    def update(self, step_duration: float = 1.0, traci_module=None) -> None:
        self._time_in_phase += step_duration

        if self._state_type == PhaseState.YELLOW:
            self.total_transition_time += step_duration
            self.actual_yellow_time += step_duration
            self._yellow_time_remaining -= step_duration
            if self._yellow_time_remaining <= 1e-6:
                if self.all_red_duration > 1e-6:
                    self._state_type = PhaseState.ALL_RED
                    self._current_signal_string = "r" * len(self.green_phases[self._target_phase])
                    self._all_red_time_remaining = self.all_red_duration
                    self._time_in_phase = 0.0
                else:
                    self._activate_green(self._target_phase)
        elif self._state_type == PhaseState.ALL_RED:
            self.total_transition_time += step_duration
            self._all_red_time_remaining -= step_duration
            if self._all_red_time_remaining <= 1e-6:
                self._activate_green(self._target_phase)
        else:
            self.phase_durations[self._current_phase] = (
                self.phase_durations.get(self._current_phase, 0.0) + step_duration
            )

        if traci_module is not None:
            self.apply_to_traci(traci_module)

    def _activate_green(self, phase_idx: int) -> None:
        self._current_phase = phase_idx
        self._current_signal_string = self.green_phases[phase_idx]
        self._state_type = PhaseState.GREEN
        self._time_in_phase = 0.0

    def sync_external_phase(
        self,
        phase_index: Optional[int] = None,
        signal_state: Optional[str] = None,
        step_duration: float = 1.0,
    ) -> None:
        if signal_state is not None:
            self._current_signal_string = signal_state
            if signal_state in self.green_phases:
                idx = self.green_phases.index(signal_state)
            else:
                self.total_transition_time += step_duration
                if any(c in signal_state for c in ("y", "Y")):
                    self._state_type = PhaseState.YELLOW
                    self.actual_yellow_time += step_duration
                else:
                    self._state_type = PhaseState.ALL_RED
                self._time_in_phase += step_duration
                return
        elif phase_index is not None and 0 <= phase_index < len(self.green_phases):
            idx = phase_index
            self._current_signal_string = self.green_phases[idx]
        else:
            self._state_type = PhaseState.YELLOW
            self.total_transition_time += step_duration
            self.actual_yellow_time += step_duration
            self._time_in_phase += step_duration
            return

        if idx != self._current_phase:
            self._current_phase = self._target_phase = idx
            self._time_in_phase = 0.0
            self.phase_switch_count += 1
        elif self._state_type != PhaseState.GREEN:
            self._time_in_phase = 0.0

        self._state_type = PhaseState.GREEN
        self._time_in_phase += step_duration
        self.phase_durations[idx] = self.phase_durations.get(idx, 0.0) + step_duration

    def apply_to_traci(self, traci_module) -> None:
        try:
            traci_module.trafficlight.setRedYellowGreenState(
                self.tls_id, self._current_signal_string
            )
        except Exception as e:
            raise RuntimeError(f"Failed to set state for TLS '{self.tls_id}': {e}") from e
