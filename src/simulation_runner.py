"""Module thực thi mô phỏng SUMO thông qua TraCI/libsumo với điều khiển tín hiệu đèn giao thông."""

from dataclasses import dataclass
import os
from typing import Any, Callable, List, Optional, Sequence, Tuple
import sumolib
import traci

try:
    import libsumo
    HAS_LIBSUMO = True
except ImportError:
    libsumo = None
    HAS_LIBSUMO = False

from src.controllers import (
    FixedTimeController,
    MaxPressureController,
    QLearningController,
    build_default_controller,
)
from src.data_inspector import inspect_scenario, ScenarioInfo


@dataclass
class SimulationResult:
    """Container chứa dữ liệu kết quả của một lượt chạy mô phỏng."""
    total_steps: int
    sim_time_start: float
    sim_time_end: float
    queue_history: List[int]
    controller_step_count: int
    mean_queue_length: float
    max_queue_length: int
    tripinfo_path: Optional[str] = None
    completed: bool = True
    tls_id: Optional[str] = None
    controller_type: Optional[str] = None
    backend: Optional[str] = None


class SimulationRunner:
    """Điều phối mô phỏng SUMO thông qua TraCI/libsumo kết hợp điều khiển tín hiệu đèn giao thông.

    Điều phối vòng lặp mô phỏng theo từng giây:
      1. Cập nhật pha tín hiệu đèn thông qua bộ điều khiển (controller)
      2. Đo độ dài hàng đợi (số xe dừng với v < 0.1 m/s) trên các làn đi vào
      3. Tiến một bước mô phỏng (conn.simulationStep())
    Xử lý an toàn việc giải phóng kết nối TraCI/libsumo trong khối try...finally.
    """

    def __init__(
        self,
        cfg_path: str,
        controller: Optional[Any] = None,
        tls_id: Optional[str] = None,
        step_duration: float = 1.0,
        tripinfo_output: Optional[str] = None,
        incoming_lanes: Optional[Sequence[str]] = None,
        additional_args: Optional[Sequence[str]] = None,
        controller_type: str = "fixed_time",
        use_libsumo: bool = False,
    ) -> None:
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f"SUMO configuration file not found: {cfg_path}")

        if controller_type not in ("fixed_time", "max_pressure", "q_learning"):
            raise ValueError(f"Unsupported controller_type: {controller_type}. Must be 'fixed_time', 'max_pressure', or 'q_learning'.")

        self.cfg_path = os.path.abspath(cfg_path)
        self.step_duration = float(step_duration)
        self.tripinfo_output = os.path.abspath(tripinfo_output) if tripinfo_output else None
        self.additional_args = list(additional_args) if additional_args else []
        self.use_libsumo = bool(use_libsumo)
        self._conn: Optional[Any] = None
        self._backend: str = "traci"

        # Tải siêu dữ liệu kịch bản nếu thiếu controller, tls_id hoặc incoming_lanes
        scenario_info: Optional[ScenarioInfo] = None
        if controller is None or tls_id is None or incoming_lanes is None:
            try:
                scenario_info = inspect_scenario(
                    scenario_name=os.path.splitext(os.path.basename(cfg_path))[0],
                    cfg_path=self.cfg_path,
                )
            except Exception:
                scenario_info = None

        # Xác định mã định danh TLS ID
        if tls_id:
            self.tls_id = tls_id
        elif controller and getattr(controller, "tls_id", None):
            self.tls_id = controller.tls_id
        elif scenario_info and scenario_info.tls_ids:
            self.tls_id = scenario_info.tls_ids[0]
        else:
            self.tls_id = ""

        # Xác định bộ điều khiển (controller) và loại controller
        if controller is not None:
            self.controller = controller
            if isinstance(controller, MaxPressureController):
                self.controller_type = "max_pressure"
            elif isinstance(controller, QLearningController):
                self.controller_type = "q_learning"
            elif isinstance(controller, FixedTimeController):
                self.controller_type = "fixed_time"
            else:
                self.controller_type = controller_type
        else:
            self.controller_type = controller_type
            if scenario_info is None:
                raise ValueError(f"Cannot initialize {self.controller_type} controller without scenario metadata.")
            green_phases = scenario_info.get_green_phases(self.tls_id)
            if not green_phases:
                raise ValueError(f"No green phases found for TLS '{self.tls_id}'.")

            movements = scenario_info.get_movements(self.tls_id) if self.controller_type != "fixed_time" else None
            self.controller = build_default_controller(
                controller_type=self.controller_type,
                green_phases=green_phases,
                movements=movements,
                tls_id=self.tls_id,
            )

        # Xác định các làn đường đi vào nút giao (incoming lanes)
        if incoming_lanes is not None:
            self.incoming_lanes: List[str] = sorted(list(set(incoming_lanes)))
        elif scenario_info and self.tls_id in scenario_info.controlled_lanes_per_tls:
            self.incoming_lanes = sorted(list(set(scenario_info.controlled_lanes_per_tls[self.tls_id])))
        else:
            self.incoming_lanes = []

    def build_sumo_cmd(self, gui: bool = False) -> List[str]:
        """Xây dựng danh sách đối số dòng lệnh để khởi động SUMO.

        Tham số:
            gui: True nếu muốn mở giao diện đồ họa 'sumo-gui', False để chạy ngầm 'sumo'.

        Trả về:
            Danh sách đối số dòng lệnh cho traci.start().
        """
        binary_name = "sumo-gui" if gui else "sumo"
        try:
            sumo_bin = sumolib.checkBinary(binary_name)
        except Exception:
            sumo_bin = binary_name

        cmd = [
            sumo_bin,
            "-c",
            self.cfg_path,
            "--no-warnings",
            "true",
        ]

        if self.tripinfo_output:
            cmd.extend(["--tripinfo-output", self.tripinfo_output])

        # Nối các tham số bổ sung (tránh trùng lặp cờ)
        if self.additional_args:
            cmd.extend(self.additional_args)

        return cmd

    @property
    def conn(self) -> Any:
        """Trả về đối tượng kết nối mô phỏng đang hoạt động (libsumo hoặc traci)."""
        if self._conn is not None:
            return self._conn
        if self.use_libsumo and HAS_LIBSUMO:
            return libsumo
        return traci

    @property
    def backend(self) -> str:
        """Tên backend đang hoạt động: 'libsumo' hoặc 'traci'."""
        return self._backend

    def start(self, gui: bool = False) -> None:
        """Khởi động tiến trình SUMO và kết nối libsumo hoặc TraCI.

        Tham số:
            gui: Nếu True, khởi chạy sumo-gui (buộc dùng traci do libsumo không hỗ trợ GUI).
                 Nếu False, khởi chạy sumo ở chế độ headless với libsumo (nếu được yêu cầu và có sẵn).
        """
        # Đảm bảo đóng mọi kết nối đang treo trước khi khởi động lượt mới
        self.close()

        if self.use_libsumo and not gui:
            if HAS_LIBSUMO:
                self._conn = libsumo
                self._backend = "libsumo"
            else:
                self._conn = traci
                self._backend = "traci"
        else:
            self._conn = traci
            self._backend = "traci"

        cmd = self.build_sumo_cmd(gui=gui)
        self.conn.start(cmd)

        # Nếu chưa biết các làn đi vào trước đó, truy vấn động từ mạng lưới
        if not self.incoming_lanes and self.tls_id:
            try:
                lanes = self.conn.trafficlight.getControlledLanes(self.tls_id)
                self.incoming_lanes = sorted(list(set(lanes)))
            except Exception:
                pass

    def measure_queue(self) -> int:
        """Đo tổng số lượng xe dừng chờ (vận tốc < 0.1 m/s) trên các làn đường đi vào.

        Trả về:
            Tổng số lượng xe trong hàng đợi trên tất cả các làn vào được theo dõi.
        """
        if not self.incoming_lanes:
            return 0
        total_halted = 0
        for lane_id in self.incoming_lanes:
            try:
                total_halted += self.conn.lane.getLastStepHaltingNumber(lane_id)
            except Exception:
                pass
        return total_halted

    def step(self) -> Tuple[str, int]:
        """Thực thi một bước điều phối mô phỏng đơn lẻ:

        1. Cập nhật pha tín hiệu đèn thông qua bộ điều khiển
        2. Đo độ dài hàng đợi trên các làn đi vào
        3. Tiến một bước mô phỏng (conn.simulationStep())

        Trả về:
            Tuple[str, int]: (trạng_thái_đèn, độ_dài_hàng_đợi)
        """
        # 1. Cập nhật trạng thái tín hiệu đèn
        state, _ = self.controller.step(self.step_duration)
        if self.tls_id:
            self.conn.trafficlight.setRedYellowGreenState(self.tls_id, state)

        # 2. Đo hàng đợi
        queue = self.measure_queue()

        # 3. Bước mô phỏng tiếp theo
        self.conn.simulationStep()

        return state, queue

    def close(self) -> None:
        """Đóng an toàn kết nối TraCI hoặc libsumo nếu đang tải."""
        targets = [self._conn, libsumo if HAS_LIBSUMO else None, traci]
        for conn in targets:
            if conn is not None:
                try:
                    if getattr(conn, "isLoaded", lambda: False)():
                        conn.close()
                except Exception:
                    pass
        self._conn = None

    def __enter__(self) -> "SimulationRunner":
        """Hỗ trợ context manager khi bắt đầu khối with."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Hỗ trợ context manager tự động đóng TraCI hoặc libsumo khi thoát."""
        self.close()

    def run(
        self,
        gui: bool = False,
        max_steps: int = 3600,
        step_callback: Optional[Callable[[int, str, int], None]] = None,
    ) -> SimulationResult:
        """Chạy mô phỏng tối đa max_steps bước với cơ chế xử lý ngoại lệ an toàn.

        Tham số:
            gui: True để bật giao diện GUI, False để chạy ngầm (headless).
            max_steps: Số giây/bước mô phỏng tối đa cần chạy.
            step_callback: Callback tùy chọn được gọi tại mỗi bước với dạng (step_idx, state, queue).

        Trả về:
            SimulationResult chứa số bước thực hiện và các chỉ số hàng đợi.
        """
        if max_steps <= 0:
            raise ValueError(f"max_steps must be positive, got {max_steps}.")

        queue_history: List[int] = []
        sim_time_start: float = 0.0
        sim_time_end: float = 0.0
        completed = False

        self.controller.reset()

        try:
            self.start(gui=gui)
            sim_time_start = float(self.conn.simulation.getTime())

            for step_idx in range(max_steps):
                state, current_queue = self.step()
                queue_history.append(current_queue)

                if step_callback is not None:
                    step_callback(step_idx, state, current_queue)

                # Nếu không còn phương tiện nào trên mạng lưới, kết thúc sớm
                if self.conn.simulation.getMinExpectedNumber() <= 0 and step_idx > 10:
                    break

            sim_time_end = float(self.conn.simulation.getTime())
            completed = True
        finally:
            # Luôn đảm bảo đóng TraCI / libsumo an toàn
            self.close()

        mean_queue = float(sum(queue_history) / len(queue_history)) if queue_history else 0.0
        max_queue = max(queue_history) if queue_history else 0

        return SimulationResult(
            total_steps=len(queue_history),
            sim_time_start=sim_time_start,
            sim_time_end=sim_time_end,
            queue_history=queue_history,
            controller_step_count=self.controller.step_count,
            mean_queue_length=mean_queue,
            max_queue_length=max_queue,
            tripinfo_path=self.tripinfo_output,
            completed=completed,
            tls_id=self.tls_id,
            controller_type=self.controller_type,
            backend=self.backend,
        )
