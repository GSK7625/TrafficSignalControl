"""Fixed-time traffic signal controller module."""

from typing import Any, Optional, Sequence, Tuple

from src.controllers.base import BaseController, build_yellow_state, register_controller

__all__ = ["FixedTimeController", "build_yellow_state"]


@register_controller("fixed_time")
class FixedTimeController(BaseController):
    """Fixed-time traffic signal controller with configurable cycle and yellow clearance."""

    def __init__(
        self,
        green_phases: Sequence[Any],
        green_time: float = 30.0,
        yellow_time: float = 3.0,
        all_red_time: float = 1.0,
        max_red: float = 90.0,
        tls_id: Optional[str] = None,
    ) -> None:
        if green_time <= 0:
            raise ValueError(f"green_time must be positive, got {green_time}.")

        self.green_time: float = float(green_time)
        super().__init__(
            green_phases=green_phases,
            yellow_time=yellow_time,
            all_red_time=all_red_time,
            max_red=max_red,
            tls_id=tls_id,
        )

    def step(self, step_duration: float = 1.0) -> Tuple[str, bool]:
        """Advances controller by step_duration seconds and returns (state, changed)."""
        if step_duration <= 0:
            raise ValueError(f"step_duration must be positive, got {step_duration}.")

        # Advance phase when interval duration expires
        if self._is_yellow:
            limit = self.yellow_time
        elif self._is_all_red:
            limit = self.all_red_time
        else:
            limit = self.green_time

        if self._time_in_phase >= limit - 1e-6:
            self._time_in_phase = 0.0
            if self._is_yellow:
                self._is_yellow = False
                if self.all_red_time > 0:
                    self._is_all_red = True
                else:
                    self._is_all_red = False
                    self._current_phase_idx = (self._current_phase_idx + 1) % len(self.green_phases)
            elif self._is_all_red:
                self._is_all_red = False
                self._current_phase_idx = (self._current_phase_idx + 1) % len(self.green_phases)
            else:
                self._next_phase_idx = (self._current_phase_idx + 1) % len(self.green_phases)
                if self.yellow_time > 0:
                    self._is_yellow = True
                elif self.all_red_time > 0:
                    self._is_all_red = True
                else:
                    self._current_phase_idx = self._next_phase_idx

        self._update_red_timers(step_duration)

        state = self.get_current_state()
        changed = (self._last_state != state)
        self._last_state = state

        self._time_in_phase += step_duration
        self._total_time += step_duration
        self._step_count += 1
        return state, changed
