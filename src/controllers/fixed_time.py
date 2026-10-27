from typing import Any, Optional, Sequence, Tuple
from src.controllers.base import BaseController, build_yellow_state, register_controller

__all__ = ["FixedTimeController", "build_yellow_state"]

@register_controller("fixed_time")
class FixedTimeController(BaseController):
    """Bộ điều khiển tín hiệu giao thông chu kỳ cố định với thời gian đèn vàng và đỏ toàn phần cấu hình được."""

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
        """Tiến bộ điều khiển thêm step_duration giây và trả về (trạng_thái_đèn, cờ_đổi_pha)."""
        if step_duration <= 0:
            raise ValueError(f"step_duration must be positive, got {step_duration}.")

        if not self._handle_clearance():
            if self._time_in_phase >= self.green_time - 1e-6:
                next_p = (self._current_phase_idx + 1) % len(self.green_phases)
                self._trigger_switch(next_p)

        return self._advance_timers(step_duration)
