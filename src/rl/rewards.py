"""Các hàm phần thưởng (Reward) cho điều khiển tín hiệu đèn giao thông bằng học tăng cường.

Căn chỉnh trực tiếp tín hiệu phần thưởng đa mục tiêu với 5 chỉ số giao thông cốt lõi:
  1. Thời gian chờ trung bình (Average Waiting Time): phạt trễ
  2. Thời gian di chuyển trung bình / Tổn thất thời gian: phạt tổn thất vận tốc
  3. Độ dài hàng đợi trung bình & cực đại: phạt bậc hai hàng đợi để xử lý nút thắt cổ chai
  4. Thông lượng (Throughput): thưởng dương cho xe đã hoàn thành lộ trình
  5. Thời gian tổn thất do đổi pha: phạt chuyển pha quá nhiều để tránh nhấp nháy đèn liên tục
"""

import math
from typing import Any, Dict, Optional


class MetricsAlignedReward:
    """Hàm phần thưởng đa mục tiêu căn chỉnh trực tiếp với 5 chỉ số giao thông cốt lõi.

    Tính toán phần thưởng cho một khoảng quyết định (decision interval):
        R = scale * (
            w_completed * completed_vehicles
            - w_queue * mean(sum(queue^2))
            - w_waiting * mean(waiting_time)
            - w_time_loss * mean(time_loss)
            - (w_switch if phase_switched else 0.0)
        )
    """

    def __init__(
        self,
        scale: float = 1.0,
        w_completed: float = 2.0,
        w_queue: float = 0.05,
        w_waiting: float = 0.02,
        w_time_loss: float = 0.01,
        w_switch: float = 0.5,
    ) -> None:
        """Khởi tạo MetricsAlignedReward với trọng số các chỉ số."""
        for name, val in (
            ("scale", scale),
            ("w_completed", w_completed),
            ("w_queue", w_queue),
            ("w_waiting", w_waiting),
            ("w_time_loss", w_time_loss),
            ("w_switch", w_switch),
        ):
            if not math.isfinite(val) or val < 0:
                raise ValueError(f"{name} must be non-negative and finite, got {val}")

        self.scale = float(scale)
        self.w_completed = float(w_completed)
        self.w_queue = float(w_queue)
        self.w_waiting = float(w_waiting)
        self.w_time_loss = float(w_time_loss)
        self.w_switch = float(w_switch)

        self._completed_sum = 0.0
        self._queue_sq_sum = 0.0
        self._waiting_sum = 0.0
        self._time_loss_sum = 0.0
        self._has_switch = False
        self._step_count = 0

    def reset_interval(self) -> None:
        """Đặt lại các chỉ số tạm thời tích lũy trong khoảng quyết định."""
        self._completed_sum = 0.0
        self._queue_sq_sum = 0.0
        self._waiting_sum = 0.0
        self._time_loss_sum = 0.0
        self._has_switch = False
        self._step_count = 0

    def reset(self) -> None:
        """Đặt lại trạng thái của hàm phần thưởng cho episode mới."""
        self.reset_interval()

    def record_step(
        self,
        in_lane_queues: Optional[Dict[str, float]] = None,
        waiting_time: float = 0.0,
        completed_vehicles: int = 0,
        time_loss: float = 0.0,
        is_switch: bool = False,
        **kwargs: Any,
    ) -> None:
        """Ghi nhận các số liệu đo lường giao thông cho một bước mô phỏng đơn lẻ.

        Tham số:
            in_lane_queues: Dict ánh xạ mã làn đường tới độ dài hàng đợi / số xe dừng.
            waiting_time: Tổng thời gian chờ của các phương tiện đang hoạt động ở bước này.
            completed_vehicles: Số lượng phương tiện đã hoàn thành hành trình tại bước này.
            time_loss: Tổng tổn thất thời gian của các xe đang hoạt động ở bước này.
            is_switch: True nếu có sự kiện đổi pha đèn diễn ra trong bước này.
        """
        if in_lane_queues:
            self._queue_sq_sum += sum(float(q) ** 2 for q in in_lane_queues.values())
        self._waiting_sum += float(waiting_time or 0.0)
        self._completed_sum += float(completed_vehicles or 0)
        self._time_loss_sum += float(time_loss or 0.0)
        if is_switch:
            self._has_switch = True
        self._step_count += 1

    def calculate_reward(self) -> float:
        """Tính toán giá trị phần thưởng chuẩn hóa cho khoảng quyết định.

        Trả về:
            Giá trị phần thưởng vô hướng đã tính toán.
        """
        if self._step_count == 0:
            return 0.0

        avg_queue_sq = self._queue_sq_sum / self._step_count
        avg_waiting = self._waiting_sum / self._step_count
        avg_time_loss = self._time_loss_sum / self._step_count

        reward = (
            self.w_completed * self._completed_sum
            - self.w_queue * avg_queue_sq
            - self.w_waiting * avg_waiting
            - self.w_time_loss * avg_time_loss
            - (self.w_switch if self._has_switch else 0.0)
        )
        return float(reward * self.scale)

    def get_config(self) -> Dict[str, Any]:
        """Trả về các tham số cấu hình của hàm phần thưởng."""
        return {
            "name": "MetricsAlignedReward",
            "scale": self.scale,
            "w_completed": self.w_completed,
            "w_queue": self.w_queue,
            "w_waiting": self.w_waiting,
            "w_time_loss": self.w_time_loss,
            "w_switch": self.w_switch,
        }
