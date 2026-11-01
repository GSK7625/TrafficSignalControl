"""Bộ điều khiển đèn giao thông thích ứng Max-Pressure (Varaiya 2013).

Tính toán áp suất mạng lưới:
    W(p) = sum_{m in M(p)} (x_u - x_d)
và lựa chọn pha xanh có áp suất lớn nhất kết hợp các ràng buộc an toàn:
    - min_green (20.0s): Ngăn chặn đổi pha quá nhanh.
    - max_green (45.0s): Ngăn chặn một pha chiếm dụng nút giao quá lâu.
    - max_red (90.0s): Cơ chế chống đói pha (Starvation Watchdog).
    - yellow_time (3.0s) & all_red_time (1.0s): Dọn nút giao an toàn.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from src.controllers.fixed_time import build_all_red_state, build_yellow_state


class MaxPressureController:
    """Bộ điều khiển Max-Pressure thích ứng dựa trên áp suất chênh lệch làn vào/ra."""

    def __init__(
        self,
        green_phases: List[str],
        movements: List[Dict[str, Any]],
        min_green: float = 20.0,
        max_green: float = 45.0,
        yellow_time: float = 3.0,
        all_red_time: float = 1.0,
        max_red: float = 90.0,
        tls_id: Optional[str] = None,
        vehicle_count_fn: Optional[Callable[[str], int]] = None,
    ) -> None:
        """Khởi tạo bộ điều khiển Max-Pressure.

        Tham số:
            green_phases: Danh sách chuỗi trạng thái các pha xanh hợp lệ.
            movements: Danh sách luồng chuyển động (link_index, from_lane, to_lane, dir).
            min_green: Thời gian xanh tối thiểu (mặc định 20.0s).
            max_green: Thời gian xanh tối đa trước khi ép đổi pha (mặc định 45.0s).
            yellow_time: Thời gian dọn đèn vàng (mặc định 3.0s).
            all_red_time: Thời gian dọn đỏ toàn phần (mặc định 1.0s).
            max_red: Ngưỡng thời gian chờ đỏ tối đa chống đói pha (mặc định 90.0s).
            tls_id: Mã cụm đèn tín hiệu.
            vehicle_count_fn: Hàm tùy biến lấy số lượng xe trên làn (nếu None, dùng TraCI).
        """
        if not green_phases:
            raise ValueError("green_phases phải chứa ít nhất một pha xanh.")
        if min_green <= 0:
            raise ValueError(f"min_green phải dương, nhận được: {min_green}")
        if max_green < min_green:
            raise ValueError(f"max_green ({max_green}) phải >= min_green ({min_green})")

        self.green_phases: List[str] = list(green_phases)
        self.movements: List[Dict[str, Any]] = list(movements)
        self.min_green: float = float(min_green)
        self.max_green: float = float(max_green)
        self.yellow_time: float = float(yellow_time)
        self.all_red_time: float = float(all_red_time)
        self.max_red: float = float(max_red)
        self.tls_id: Optional[str] = tls_id
        self._vehicle_count_fn: Optional[Callable[[str], int]] = vehicle_count_fn

        # Trạng thái điều khiển
        self.current_phase_idx: int = 0
        self.next_phase_idx: int = 0
        self.time_in_phase: float = 0.0
        self.is_yellow: bool = False
        self.is_all_red: bool = False
        self.step_count: int = 0

        # Bộ đếm thời gian chờ đèn đỏ cho từng pha để giám sát chống đói pha
        self.phase_red_timers: List[float] = [0.0] * len(self.green_phases)

        # Tiền xử lý: gom các chuyển động hoạt động trong từng pha xanh
        self._phase_movements: List[List[Dict[str, Any]]] = []
        for p_state in self.green_phases:
            active_m = []
            for m in self.movements:
                idx = m.get("link_index", -1)
                if 0 <= idx < len(p_state) and p_state[idx] in ("G", "g", "s"):
                    active_m.append(m)
            self._phase_movements.append(active_m)

    @property
    def num_phases(self) -> int:
        return len(self.green_phases)

    def _get_lane_vehicle_count(self, lane_id: str) -> int:
        """Lấy số lượng xe trên một làn đường qua hàm tùy biến hoặc TraCI."""
        if not lane_id:
            return 0
        if self._vehicle_count_fn is not None:
            try:
                return int(self._vehicle_count_fn(lane_id))
            except Exception:
                return 0
        try:
            import traci
            if traci.isLoaded():
                return int(traci.lane.getLastStepVehicleNumber(lane_id))
        except Exception:
            pass
        return 0

    def calculate_phase_pressures(self) -> Dict[int, float]:
        """Tính toán áp suất W(p) = sum_{m in M(p)} (x_u / deg - x_d) cho từng pha xanh.

        Tránh đếm trùng xe trên làn vào bằng cách phân bổ đều số xe cho các hướng rẽ cùng bật.
        """
        # Cache số lượng xe trên từng làn trong bước này để tránh gọi TraCI nhiều lần
        lane_counts: Dict[str, int] = {}

        def get_count(lane: str) -> int:
            if lane not in lane_counts:
                lane_counts[lane] = self._get_lane_vehicle_count(lane)
            return lane_counts[lane]

        pressures: Dict[int, float] = {}

        for p_idx, moves in enumerate(self._phase_movements):
            # Đếm số chuyển động hoạt động của mỗi làn vào u trong pha này
            u_degrees: Dict[str, int] = {}
            for m in moves:
                u = m.get("from_lane", "")
                if u:
                    u_degrees[u] = u_degrees.get(u, 0) + 1

            # Tính tổng áp suất
            total_pressure = 0.0
            for m in moves:
                u = m.get("from_lane", "")
                d = m.get("to_lane", "")
                u_cnt = get_count(u)
                d_cnt = get_count(d)
                deg = u_degrees.get(u, 1)

                # Áp suất chuyển động: x_u / deg - x_d
                w_m = (u_cnt / float(deg)) - float(d_cnt)
                total_pressure += w_m

            pressures[p_idx] = round(total_pressure, 2)

        return pressures

    def select_next_phase(self) -> Tuple[int, str, Dict[int, float]]:
        """Lựa chọn pha tiếp theo dựa trên áp suất và các ràng buộc an toàn.

        Trả về:
            Tuple (pha_được_chọn, lý_do_lựa_chọn, bảng_áp_suất_các_pha)
        """
        pressures = self.calculate_phase_pressures()
        curr_p = self.current_phase_idx
        num_p = self.num_phases

        if num_p <= 1:
            return curr_p, "single_phase", pressures

        alt_phases = [p for p in range(num_p) if p != curr_p]

        # 1. Ràng buộc an toàn 1: Chống đói pha (Starvation Watchdog)
        starved_phases = [
            p for p in alt_phases
            if self.phase_red_timers[p] >= self.max_red - 1e-6
        ]
        if starved_phases:
            # Chọn pha bị đói có thời gian đỏ lâu nhất, nếu bằng nhau xét áp suất
            chosen = max(
                starved_phases,
                key=lambda p: (self.phase_red_timers[p], pressures.get(p, 0.0), -p),
            )
            return chosen, "starvation_override", pressures

        # 2. Ràng buộc an toàn 2: Cưỡng bức đổi pha khi chạm trần max_green
        force_switch = (self.time_in_phase >= self.max_green - 1e-6)
        if force_switch:
            # Chọn pha đối thủ có áp suất lớn nhất
            chosen = max(
                alt_phases,
                key=lambda p: (pressures.get(p, 0.0), self.phase_red_timers[p], -p),
            )
            return chosen, "max_green_forced", pressures

        # 3. Lựa chọn thông thường dựa trên Max-Pressure
        # Quy tắc hòa (Tie-breaking):
        # - Nếu pha hiện tại cũng đạt áp suất cao nhất -> giữ nguyên pha hiện tại (tránh lãng phí 4s dọn nút)
        # - Nếu đổi pha -> chọn pha đối thủ có áp suất cao nhất, ưu tiên pha chờ đỏ lâu hơn
        max_pressure_val = max(pressures.values())
        curr_pressure = pressures.get(curr_p, 0.0)

        # Nếu pha hiện tại có áp suất bằng max hoặc áp suất max <= 0 -> giữ nguyên
        if max_pressure_val <= 0.0 or curr_pressure >= max_pressure_val:
            return curr_p, "keep_current_optimal", pressures

        # Ngược lại, chọn pha đối thủ có áp suất cao nhất
        best_alt = max(
            alt_phases,
            key=lambda p: (pressures.get(p, 0.0), self.phase_red_timers[p], -p),
        )

        return best_alt, "max_pressure_switch", pressures

    def get_current_state(self) -> str:
        """Lấy chuỗi trạng thái đèn hiện tại (xanh, vàng hoặc đỏ toàn phần)."""
        curr_green = self.green_phases[self.current_phase_idx]
        if self.is_yellow:
            next_green = self.green_phases[self.next_phase_idx]
            return build_yellow_state(curr_green, next_green)
        if self.is_all_red:
            return build_all_red_state(curr_green)
        return curr_green

    def step(self, step_duration: float = 1.0) -> Tuple[str, bool, Optional[str], Optional[Dict[str, Any]]]:
        """Tiến bộ điều khiển thêm một bước thời gian step_duration giây.

        Trả về:
            Tuple (trạng_thái_đèn, cờ_đổi_pha, tên_sự_kiện, thông_tin_quyết_định)
        """
        event_name: Optional[str] = None
        phase_changed: bool = False
        decision_info: Optional[Dict[str, Any]] = None

        # Cập nhật bộ đếm thời gian chờ đèn đỏ của các pha
        for p in range(self.num_phases):
            if p == self.current_phase_idx and not (self.is_yellow or self.is_all_red):
                self.phase_red_timers[p] = 0.0
            else:
                self.phase_red_timers[p] += step_duration

        # 1. Đang trong pha đèn vàng dọn nút giao
        if self.is_yellow:
            if self.time_in_phase >= self.yellow_time - 1e-6:
                self.is_yellow = False
                self.time_in_phase = 0.0
                if self.all_red_time > 0:
                    self.is_all_red = True
                    event_name = "enter_all_red"
                else:
                    self.current_phase_idx = self.next_phase_idx
                    event_name = "enter_green"
                phase_changed = True

        # 2. Đang trong pha đỏ toàn phần dọn nút giao
        elif self.is_all_red:
            if self.time_in_phase >= self.all_red_time - 1e-6:
                self.is_all_red = False
                self.current_phase_idx = self.next_phase_idx
                self.time_in_phase = 0.0
                event_name = "enter_green"
                phase_changed = True

        # 3. Đang trong pha xanh: ra quyết định thích ứng khi đã đủ min_green
        else:
            if self.time_in_phase >= self.min_green - 1e-6:
                target_p, reason, pressures = self.select_next_phase()
                decision_info = {
                    "pressures": pressures,
                    "target_phase": target_p,
                    "current_phase": self.current_phase_idx,
                    "reason": reason,
                    "time_in_phase": self.time_in_phase,
                }

                if target_p != self.current_phase_idx:
                    self.next_phase_idx = target_p
                    self.time_in_phase = 0.0
                    if self.yellow_time > 0:
                        self.is_yellow = True
                        event_name = "enter_yellow"
                    elif self.all_red_time > 0:
                        self.is_all_red = True
                        event_name = "enter_all_red"
                    else:
                        self.current_phase_idx = target_p
                        event_name = "enter_green"
                    phase_changed = True

        current_state = self.get_current_state()
        self.time_in_phase += step_duration
        self.step_count += 1

        return current_state, phase_changed, event_name, decision_info

    def reset(self) -> None:
        """Đặt lại bộ điều khiển về trạng thái ban đầu."""
        self.current_phase_idx = 0
        self.next_phase_idx = 0
        self.time_in_phase = 0.0
        self.is_yellow = False
        self.is_all_red = False
        self.step_count = 0
        self.phase_red_timers = [0.0] * len(self.green_phases)
