"""Bộ điều khiển chu kỳ cố định (Fixed-Time Controller).

Điều khiển đèn tín hiệu tuần tự theo danh sách các pha xanh:
  - Giữ mỗi pha xanh đúng 30 giây (green_time).
  - Chuyển tiếp đèn vàng đúng 3 giây (yellow_time).
  - Chuyển tiếp đỏ toàn phần đúng 1 giây (all_red_time).
  - Không sử dụng bất kỳ thông tin hàng đợi hay lưu lượng xe nào để thay đổi thời gian xanh.
"""

from typing import Any, Dict, List, Optional, Tuple


def build_yellow_state(
    curr_state: str,
    next_state: Optional[str] = None,
    only_changing: bool = True,
) -> str:
    """Tạo chuỗi trạng thái đèn vàng an toàn: các đèn xanh sắp đỏ chuyển thành 'y'."""
    res: List[str] = []
    for idx, char in enumerate(curr_state):
        if char in ("G", "g", "s"):
            if (
                only_changing
                and next_state is not None
                and idx < len(next_state)
                and next_state[idx] in ("G", "g", "s")
            ):
                res.append(char)  # Giữ nguyên nếu pha sau vẫn xanh
            else:
                res.append("y")
        else:
            res.append("r")
    return "".join(res)


def build_all_red_state(curr_state: str) -> str:
    """Tạo chuỗi trạng thái đỏ toàn phần: tất cả các đèn hiển thị 'r'."""
    return "r" * len(curr_state)


class FixedTimeController:
    """Bộ điều khiển đèn giao thông Fixed-Time tuần tự tĩnh."""

    def __init__(
        self,
        green_phases: List[str],
        green_time: float = 30.0,
        yellow_time: float = 3.0,
        all_red_time: float = 1.0,
        tls_id: Optional[str] = None,
    ) -> None:
        """Khởi tạo bộ điều khiển Fixed-Time.

        Tham số:
            green_phases: Danh sách các chuỗi trạng thái đèn xanh hợp lệ.
            green_time: Thời gian giữ xanh cố định cho mỗi pha (mặc định 30.0s).
            yellow_time: Thời gian dọn nút giao đèn vàng (mặc định 3.0s).
            all_red_time: Thời gian đỏ toàn phần dọn nút giao (mặc định 1.0s).
            tls_id: Mã định danh cụm đèn tín hiệu (tùy chọn).
        """
        if not green_phases:
            raise ValueError("green_phases phải chứa ít nhất một pha xanh.")
        if green_time <= 0:
            raise ValueError(f"green_time phải dương, nhận được: {green_time}")

        self.green_phases: List[str] = list(green_phases)
        self.green_time: float = float(green_time)
        self.yellow_time: float = float(yellow_time)
        self.all_red_time: float = float(all_red_time)
        self.tls_id: Optional[str] = tls_id

        # Trạng thái vận hành nội tại
        self.current_phase_idx: int = 0
        self.next_phase_idx: int = 0
        self.time_in_phase: float = 0.0
        self.is_yellow: bool = False
        self.is_all_red: bool = False
        self.step_count: int = 0

    @property
    def num_phases(self) -> int:
        """Trả về số lượng pha xanh trong chu kỳ."""
        return len(self.green_phases)

    @property
    def cycle_time(self) -> float:
        """Tổng thời gian của một chu kỳ đầy đủ (giây)."""
        return self.num_phases * (self.green_time + self.yellow_time + self.all_red_time)

    def _build_yellow_state(self, curr_state: str, next_state: str) -> str:
        """Tạo chuỗi trạng thái đèn vàng an toàn: các đèn xanh sắp đỏ chuyển thành 'y'."""
        res: List[str] = []
        for idx, char in enumerate(curr_state):
            if char in ("G", "g", "s"):
                if idx < len(next_state) and next_state[idx] in ("G", "g", "s"):
                    res.append(char)  # Giữ nguyên nếu pha sau vẫn xanh
                else:
                    res.append("y")
            else:
                res.append("r")
        return "".join(res)

    def _build_all_red_state(self, curr_state: str) -> str:
        """Tạo chuỗi trạng thái đỏ toàn phần: tất cả các đèn hiển thị 'r'."""
        return "r" * len(curr_state)

    def get_current_state(self) -> str:
        """Lấy chuỗi trạng thái đèn hiện tại (xanh, vàng hoặc đỏ toàn phần)."""
        curr_green = self.green_phases[self.current_phase_idx]
        if self.is_yellow:
            next_green = self.green_phases[self.next_phase_idx]
            return self._build_yellow_state(curr_green, next_green)
        if self.is_all_red:
            return self._build_all_red_state(curr_green)
        return curr_green

    def step(self, step_duration: float = 1.0) -> Tuple[str, bool, Optional[str]]:
        """Tiến bộ điều khiển thêm một bước thời gian step_duration giây.

        Trả về:
            Tuple (trạng_thái_đèn, cờ_có_sự_kiện_đổi_pha, tên_sự_kiện)
        """
        event_name: Optional[str] = None
        phase_changed: bool = False

        # 1. Đang trong pha đèn vàng
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

        # 2. Đang trong pha đỏ toàn phần
        elif self.is_all_red:
            if self.time_in_phase >= self.all_red_time - 1e-6:
                self.is_all_red = False
                self.current_phase_idx = self.next_phase_idx
                self.time_in_phase = 0.0
                event_name = "enter_green"
                phase_changed = True

        # 3. Đang trong pha xanh bình thường: giữ đúng 30 giây
        else:
            if self.time_in_phase >= self.green_time - 1e-6:
                self.next_phase_idx = (self.current_phase_idx + 1) % self.num_phases
                self.time_in_phase = 0.0
                if self.yellow_time > 0:
                    self.is_yellow = True
                    event_name = "enter_yellow"
                elif self.all_red_time > 0:
                    self.is_all_red = True
                    event_name = "enter_all_red"
                else:
                    self.current_phase_idx = self.next_phase_idx
                    event_name = "enter_green"
                phase_changed = True

        # Lấy trạng thái tín hiệu tại bước này
        current_state = self.get_current_state()

        # Tăng thời gian đếm bước
        self.time_in_phase += step_duration
        self.step_count += 1

        return current_state, phase_changed, event_name

    def reset(self) -> None:
        """Đặt lại bộ điều khiển về trạng thái bắt đầu (Pha xanh 0)."""
        self.current_phase_idx = 0
        self.next_phase_idx = 0
        self.time_in_phase = 0.0
        self.is_yellow = False
        self.is_all_red = False
        self.step_count = 0
