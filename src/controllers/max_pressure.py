"""Module bộ điều khiển tín hiệu đèn giao thông Max Pressure (Varaiya 2013).

Tính toán áp suất mạng lưới W(p) = sum_{(u,d) in M(p)} (x_u - x_d)
và điều khiển đèn giao thông thích ứng linh hoạt theo thời gian thực kết hợp các ràng buộc an toàn.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.controllers.base import BaseController, get_movement_attr, register_controller

__all__ = ["MaxPressureController"]


@register_controller("max_pressure")
class MaxPressureController(BaseController):
    """Bộ điều khiển tín hiệu giao thông Max Pressure theo lý thuyết và ràng buộc an toàn.

    Thuộc tính:
        green_phases: Danh sách chuỗi trạng thái các pha đèn xanh.
        movements: Danh sách các luồng chuyển động (link/movement).
        min_green: Thời gian xanh tối thiểu tính bằng giây (mặc định: 20.0s).
        max_green: Thời gian xanh tối đa trước khi ép chuyển pha (mặc định: 45.0s).
        yellow_time: Thời gian đèn vàng giải phóng nút giao (mặc định: 3.0s).
        all_red_time: Thời gian đỏ toàn phần giải phóng nút giao (mặc định: 1.0s).
        max_red: Thời gian chờ đèn đỏ tối đa để chống nghẽn đói pha (mặc định: 90.0s).
        tls_id: Mã định danh cụm đèn tín hiệu.
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

        # Tính toán trước các luồng chuyển động hoạt động trong từng pha xanh để tối ưu tốc độ
        self._phase_movements: List[List[Any]] = []
        for p_state in self.green_phases:
            active_m: List[Any] = []
            for m in self.movements:
                idx = get_movement_attr(m, "link_index")
                if idx is not None and 0 <= idx < len(p_state) and p_state[idx] in ("G", "g"):
                    active_m.append(m)
            self._phase_movements.append(active_m)

    def calculate_phase_pressures(self) -> Dict[int, float]:
        """Tính toán áp suất W(p) = sum_{(u,d) in M(p)} (x_u - x_d) cho từng pha đèn xanh."""
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
                u_lane = get_movement_attr(m, "from_lane", "")
                d_lane = get_movement_attr(m, "to_lane", "")
                pressure += float(get_count(u_lane) - get_count(d_lane))
            pressures[p_idx] = pressure

        return pressures

    def select_next_phase(self, force_switch: Optional[bool] = None) -> int:
        """Lựa chọn pha xanh kế tiếp theo thuật toán Max Pressure và cơ chế giám sát nghẽn đói pha (Starvation Watchdog)."""
        if force_switch is None:
            force_switch = (self._time_in_phase >= self.max_green - 1e-6)

        pressures = self.calculate_phase_pressures()
        num_phases = len(self.green_phases)
        curr_p = self._current_phase_idx

        if num_phases <= 1:
            return curr_p

        alt_phases = [(curr_p + i) % num_phases for i in range(1, num_phases)]

        # Giám sát nghẽn đói pha: Kiểm tra nếu có pha đối thủ nào đã vượt ngưỡng max_red
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
        """Tiến bộ điều khiển thêm step_duration giây và trả về (trạng_thái_đèn, cờ_đổi_pha)."""
        if step_duration <= 0:
            raise ValueError(f"step_duration must be positive, got {step_duration}.")

        if not self._handle_clearance():
            if self._time_in_phase >= self.min_green - 1e-6:
                has_starvation = any(
                    self.get_phase_red_time(p) >= self.max_red - 1e-6
                    for p in range(len(self.green_phases))
                    if p != self._current_phase_idx
                )
                force_switch = (self._time_in_phase >= self.max_green - 1e-6) or has_starvation
                next_p = self.select_next_phase(force_switch=force_switch)
                self._trigger_switch(next_p)

        return self._advance_timers(step_duration)
