"""Module thu thập và tính toán các chỉ số đánh giá điều khiển tín hiệu đèn giao thông SUMO.

Theo dõi 5 chỉ số giao thông tiêu chuẩn:
  1. Thời gian chờ trung bình (Average Waiting Time - s) [qua đầu ra SUMO tripinfo]
  2. Thời gian di chuyển trung bình (Average Travel Time - s) [qua đầu ra SUMO tripinfo]
  3. Độ dài hàng đợi trung bình & cực đại (Mean & Max Queue Length - xe) [qua mẫu đo từng bước TraCI]
  4. Tổn thất thời gian trung bình (Average Time Loss - s) [qua đầu ra SUMO tripinfo]
  5. Thông lượng (Throughput - số xe hoàn thành) [qua đầu ra SUMO tripinfo]
"""

from dataclasses import dataclass
import os
from typing import Any, Dict, List, Optional, Sequence
import xml.etree.ElementTree as ET


@dataclass
class StepRecord:
    """Ảnh chụp nhanh (snapshot) điều kiện giao thông tại một bước mô phỏng đơn lẻ."""
    step_time: float
    queue_length: float
    halted_vehicles: Optional[int] = None
    running_vehicles: Optional[int] = None


@dataclass
class TripRecord:
    """Dữ liệu hành trình chi tiết của một phương tiện đã hoàn thành lộ trình."""
    vehicle_id: str
    depart: float
    arrival: float
    duration: float
    waiting_time: float
    time_loss: float
    route_length: float


class MetricsCollector:
    """Thu thập các chỉ số thời gian thực từ TraCI và phân tích file XML TripInfo sau mô phỏng.

    Cung cấp bản tổng hợp hợp nhất của 5 chỉ số hiệu năng giao thông cốt lõi.
    """

    def __init__(self, monitored_lanes: Optional[Sequence[str]] = None) -> None:
        """Khởi tạo MetricsCollector.

        Tham số:
            monitored_lanes: Danh sách tùy chọn các mã làn đường cần theo dõi trong mô phỏng.
        """
        self.monitored_lanes: List[str] = list(monitored_lanes) if monitored_lanes else []
        self.step_records: List[StepRecord] = []
        self.trip_records: List[TripRecord] = []
        self.tripinfo_metrics: Dict[str, Any] = {}

    @property
    def total_steps(self) -> int:
        """Trả về tổng số bước mô phỏng đã ghi nhận."""
        return len(self.step_records)

    @property
    def queue_history(self) -> List[float]:
        """Trả về lịch sử độ dài hàng đợi tức thời được ghi nhận tại mỗi bước."""
        return [record.queue_length for record in self.step_records]

    @property
    def mean_queue_length(self) -> float:
        """Tính toán độ dài hàng đợi trung bình trên tất cả các bước mô phỏng đã ghi."""
        if not self.step_records:
            return 0.0
        return float(sum(r.queue_length for r in self.step_records) / len(self.step_records))

    @property
    def max_queue_length(self) -> int:
        """Trả về độ dài hàng đợi đỉnh (cực đại) quan sát được qua tất cả các bước."""
        if not self.step_records:
            return 0
        return int(max(r.queue_length for r in self.step_records))

    def record_step(
        self,
        step_time: float,
        queue_length: float,
        halted_vehicles: Optional[int] = None,
        running_vehicles: Optional[int] = None,
    ) -> None:
        """Ghi nhận các chỉ số giao thông cho một bước mô phỏng đơn lẻ.

        Tham số:
            step_time: Dấu thời gian mô phỏng tính bằng giây.
            queue_length: Độ dài hàng đợi / số lượng xe dừng tại bước này.
            halted_vehicles: Số lượng xe dừng chờ cụ thể (tùy chọn).
            running_vehicles: Số lượng xe đang di chuyển trên mạng lưới (tùy chọn).
        """
        record = StepRecord(
            step_time=float(step_time),
            queue_length=float(queue_length),
            halted_vehicles=halted_vehicles,
            running_vehicles=running_vehicles,
        )
        self.step_records.append(record)

    def parse_tripinfo(self, xml_path: str) -> Dict[str, Any]:
        """Phân tích file XML --tripinfo-output của SUMO và tính toán các chỉ số hành trình.

        Tính toán:
          - throughput: số lượng phương tiện đã đến đích/hoàn thành hành trình
          - average_travel_time: thời gian hành trình trung bình (s)
          - average_waiting_time: thời gian chờ trung bình với tốc độ < 0.1 m/s (s)
          - average_time_loss: tổn thất thời gian trung bình do đèn/ùn tắc (s)
          - total_travel_time, total_waiting_time, total_time_loss (s)

        Tham số:
            xml_path: Đường dẫn tuyệt đối hoặc tương đối tới file XML tripinfo của SUMO.

        Trả về:
            Dictionary chứa các chỉ số hành trình đã tính toán.

        Ngoại lệ:
            FileNotFoundError: Nếu file XML không tồn tại.
            ValueError: Nếu file không đúng định dạng XML.
        """
        if not os.path.isfile(xml_path):
            raise FileNotFoundError(f"Tripinfo file not found: {xml_path}")

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError as err:
            raise ValueError(f"Failed to parse XML from {xml_path}: {err}") from err
        except Exception as err:
            raise ValueError(f"Error reading tripinfo file {xml_path}: {err}") from err

        total_duration = 0.0
        total_waiting = 0.0
        total_loss = 0.0
        throughput = 0
        self.trip_records.clear()

        # Duyệt qua tất cả các phần tử tripinfo trong file XML
        for elem in root.findall(".//tripinfo"):
            try:
                veh_id = elem.get("id", f"veh_{throughput}")
                duration = float(elem.get("duration", "0.0"))
                waiting_time = float(elem.get("waitingTime", "0.0"))
                time_loss = float(elem.get("timeLoss", "0.0"))
                depart = float(elem.get("depart", "0.0"))
                arrival = float(elem.get("arrival", "0.0"))
                route_length = float(elem.get("routeLength", "0.0"))

                record = TripRecord(
                    vehicle_id=veh_id,
                    depart=depart,
                    arrival=arrival,
                    duration=duration,
                    waiting_time=waiting_time,
                    time_loss=time_loss,
                    route_length=route_length,
                )
                self.trip_records.append(record)

                total_duration += duration
                total_waiting += waiting_time
                total_loss += time_loss
                throughput += 1
            except (ValueError, TypeError):
                continue

        avg_travel = (total_duration / throughput) if throughput > 0 else 0.0
        avg_waiting = (total_waiting / throughput) if throughput > 0 else 0.0
        avg_loss = (total_loss / throughput) if throughput > 0 else 0.0

        metrics = {
            "throughput": throughput,
            "average_travel_time": avg_travel,
            "average_waiting_time": avg_waiting,
            "average_time_loss": avg_loss,
            "total_travel_time": total_duration,
            "total_waiting_time": total_waiting,
            "total_time_loss": total_loss,
        }

        self.tripinfo_metrics = metrics
        return metrics

    def get_summary(self, tripinfo_path: Optional[str] = None) -> Dict[str, Any]:
        """Trả về tổng hợp hợp nhất của 5 chỉ số hiệu năng giao thông tiêu chuẩn:

        1. average_waiting_time (s)
        2. average_travel_time (s)
        3. mean_queue_length (xe) & max_queue_length (xe)
        4. average_time_loss (s)
        5. throughput (số lượng xe hoàn thành)

        Tham số:
            tripinfo_path: Đường dẫn tùy chọn tới file XML tripinfo để phân tích trước khi tổng hợp.

        Trả về:
            Dictionary chứa 5 chỉ số cốt lõi kèm siêu dữ liệu giám sát mô phỏng.
        """
        if tripinfo_path is not None:
            self.parse_tripinfo(tripinfo_path)

        avg_waiting = float(self.tripinfo_metrics.get("average_waiting_time", 0.0))
        avg_travel = float(self.tripinfo_metrics.get("average_travel_time", 0.0))
        avg_loss = float(self.tripinfo_metrics.get("average_time_loss", 0.0))
        throughput = int(self.tripinfo_metrics.get("throughput", 0))

        return {
            "average_waiting_time": avg_waiting,
            "average_travel_time": avg_travel,
            "mean_queue_length": self.mean_queue_length,
            "max_queue_length": self.max_queue_length,
            "average_time_loss": avg_loss,
            "throughput": throughput,
            "total_steps": self.total_steps,
            "queue_history": self.queue_history,
            "tripinfo_metrics": dict(self.tripinfo_metrics),
        }

    def reset(self) -> None:
        """Đặt lại tất cả các chỉ số bước và chỉ số tripinfo đã thu thập về trạng thái rỗng."""
        self.step_records.clear()
        self.trip_records.clear()
        self.tripinfo_metrics.clear()

