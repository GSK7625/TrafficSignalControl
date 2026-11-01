"""Module thu thập, tính toán và tổng hợp 5 chỉ số đánh giá giao thông SUMO.

Được thiết kế đơn giản, minh bạch và đúng chuẩn lý thuyết:
  1. Average Travel Time (s)
  2. Average Waiting Time (s)
  3. Average Time Loss (s)
  4. Mean & Max Queue Length (xe)
  5. Throughput (xe hoàn thành)
"""

import os
from typing import Any, Dict, List, Optional, Sequence
import xml.etree.ElementTree as ET


class MetricsCollector:
    """Bộ thu thập và tổng hợp chỉ số giao thông dùng chung cho mọi thuật toán."""

    def __init__(self, incoming_lanes: Sequence[str], total_demand: int = 0) -> None:
        """Khởi tạo bộ thu thập chỉ số.

        Tham số:
            incoming_lanes: Danh sách các làn vào của nút giao (mỗi làn chỉ tính 1 lần).
            total_demand: Tổng số chuyến xe dự kiến trong file kịch bản.
        """
        # Loại bỏ trùng lặp và sắp xếp cố định danh sách làn vào
        self.incoming_lanes: List[str] = sorted(list(set(incoming_lanes)))
        self.total_demand: int = int(total_demand)

        # Lịch sử ghi nhận từng bước
        self.time_history: List[float] = []
        self.queue_history: List[int] = []
        self.running_history: List[int] = []

        # Bộ đếm sự kiện tích lũy qua TraCI
        self.total_inserted: int = 0
        self.total_arrived_traci: int = 0
        self.total_teleported: int = 0
        self.total_collisions: int = 0
        self.total_emergency_stops: int = 0

    def record_step(
        self,
        sim_time: float,
        step_queue: int,
        running_count: int,
        inserted_step: int = 0,
        arrived_step: int = 0,
        teleported_step: int = 0,
        collisions_step: int = 0,
        emergency_step: int = 0,
    ) -> None:
        """Ghi nhận dữ liệu quan sát tại một bước mô phỏng đơn lẻ."""
        self.time_history.append(float(sim_time))
        self.queue_history.append(int(step_queue))
        self.running_history.append(int(running_count))

        self.total_inserted += int(inserted_step)
        self.total_arrived_traci += int(arrived_step)
        self.total_teleported += int(teleported_step)
        self.total_collisions += int(collisions_step)
        self.total_emergency_stops += int(emergency_step)

    def parse_tripinfo(
        self,
        tripinfo_path: str,
        begin_time: float,
        end_time: float,
    ) -> Dict[str, Any]:
        """Phân tích file XML --tripinfo-output của SUMO cho các xe hoàn thành trong cửa sổ đánh giá.

        Tham số:
            tripinfo_path: Đường dẫn tới file tripinfo.xml.
            begin_time: Mốc thời gian bắt đầu đánh giá (s).
            end_time: Mốc thời gian kết thúc đánh giá (s).

        Trả về:
            Dictionary chứa các chỉ số hành trình của tập xe hoàn thành.
        """
        if not os.path.isfile(tripinfo_path):
            raise FileNotFoundError(f"Không tìm thấy file tripinfo: {tripinfo_path}")

        tree = ET.parse(tripinfo_path)
        root = tree.getroot()

        completed_trips: List[Dict[str, float]] = []

        for trip in root.findall(".//tripinfo"):
            try:
                arrival = float(trip.get("arrival", "0.0"))
                # Chỉ tính xe hoàn thành trong cửa sổ đánh giá [begin_time, end_time]
                if begin_time <= arrival <= end_time:
                    completed_trips.append({
                        "id": trip.get("id", ""),
                        "depart": float(trip.get("depart", "0.0")),
                        "arrival": arrival,
                        "duration": float(trip.get("duration", "0.0")),
                        "waitingTime": float(trip.get("waitingTime", "0.0")),
                        "timeLoss": float(trip.get("timeLoss", "0.0")),
                    })
            except (ValueError, TypeError):
                continue

        count = len(completed_trips)

        # Xử lý nghiêm ngặt: nếu không có xe hoàn thành, trả về None, KHÔNG ghi 0.0 giả
        if count == 0:
            return {
                "throughput": 0,
                "average_travel_time": None,
                "average_waiting_time": None,
                "average_time_loss": None,
                "completed_trips": [],
            }

        total_travel = sum(t["duration"] for t in completed_trips)
        total_waiting = sum(t["waitingTime"] for t in completed_trips)
        total_loss = sum(t["timeLoss"] for t in completed_trips)

        return {
            "throughput": count,
            "average_travel_time": round(total_travel / count, 2),
            "average_waiting_time": round(total_waiting / count, 2),
            "average_time_loss": round(total_loss / count, 2),
            "completed_trips": completed_trips,
        }

    def get_summary(
        self,
        tripinfo_path: Optional[str] = None,
        begin_time: float = 0.0,
        end_time: float = 3600.0,
    ) -> Dict[str, Any]:
        """Tổng hợp toàn bộ 5 metrics cốt lõi cùng các thống kê phương tiện chi tiết."""
        trip_metrics: Dict[str, Any] = {
            "throughput": 0,
            "average_travel_time": None,
            "average_waiting_time": None,
            "average_time_loss": None,
            "completed_trips": [],
        }

        if tripinfo_path and os.path.isfile(tripinfo_path):
            trip_metrics = self.parse_tripinfo(tripinfo_path, begin_time, end_time)

        # Tính hàng đợi trung bình và cực đại từ lịch sử bước
        total_steps = len(self.queue_history)
        if total_steps > 0:
            mean_q = round(sum(self.queue_history) / total_steps, 2)
            max_q = max(self.queue_history)
        else:
            mean_q = None
            max_q = None

        throughput = trip_metrics["throughput"]
        final_running = self.running_history[-1] if self.running_history else 0
        waiting_to_enter = max(0, self.total_demand - self.total_inserted) if self.total_demand > 0 else 0
        completion_rate = (
            round((throughput / self.total_demand) * 100.0, 2) if self.total_demand > 0 else None
        )

        return {
            # 5 Metrics cốt lõi
            "average_travel_time": trip_metrics["average_travel_time"],
            "average_waiting_time": trip_metrics["average_waiting_time"],
            "average_time_loss": trip_metrics["average_time_loss"],
            "mean_queue_length": mean_q,
            "max_queue_length": max_q,
            "throughput": throughput,
            # Thống kê phương tiện
            "total_demand": self.total_demand,
            "inserted_vehicles": self.total_inserted,
            "completed_vehicles": throughput,
            "running_vehicles_at_end": final_running,
            "waiting_to_enter": waiting_to_enter,
            "completion_rate_pct": completion_rate,
            # Sự kiện bất thường
            "teleport_count": self.total_teleported,
            "collision_count": self.total_collisions,
            "emergency_stops": self.total_emergency_stops,
            # Dữ liệu chuỗi thời gian
            "total_simulation_steps": total_steps,
            "queue_history": self.queue_history,
        }
