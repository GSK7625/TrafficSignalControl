"""Vòng lặp mô phỏng SUMO và thu thập chỉ số giao thông dùng chung (Simulation Runner).

Triển khai quy trình mô phỏng tuần tự, rõ ràng:
  1. Đọc cấu hình mạng lưới
  2. Mở SUMO thông qua TraCI
  3. Tiến hành chạy từng bước mô phỏng với bộ điều khiển đèn
  4. Thu thập hàng đợi và sự kiện theo thời gian thực
  5. Đóng TraCI đúng mốc thời gian quy định
  6. Phân tích tệp TripInfo và tổng hợp 5 chỉ số đánh giá
"""

import json
import os
import sys
from typing import Any, Dict, List, Optional

from src.controllers.fixed_time import (
    FixedTimeController,
    build_all_red_state,
    build_yellow_state,
)
from src.data_inspector import read_scenario
from src.metrics import MetricsCollector


def get_simulation_backend(gui: bool = False):
    """Lấy backend điều khiển mô phỏng SUMO (libsumo chạy nhúng nhanh gấp 5-10 lần khi headless, traci cho GUI)."""
    if gui:
        import traci
        return traci
    try:
        import libsumo
        return libsumo
    except ImportError:
        import traci
        return traci


def run_simulation(
    scenario_name: str,
    controller: Optional[Any] = None,
    data_dir: str = "data",
    green_time: float = 30.0,
    yellow_time: float = 3.0,
    all_red_time: float = 1.0,
    seed: int = 42,
    max_steps: Optional[int] = None,
    gui: bool = False,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Khởi chạy mô phỏng điều khiển đèn và thu thập metrics đầy đủ.

    Tham số:
        scenario_name: Tên kịch bản ('cologne1' hoặc 'ingolstadt1').
        controller: Đối tượng controller (nếu None, mặc định là FixedTimeController).
        data_dir: Thư mục chứa dữ liệu kịch bản.
        green_time: Thời gian giữ xanh cố định (giây).
        yellow_time: Thời gian đèn vàng giải phóng nút (giây).
        all_red_time: Thời gian đỏ toàn phần giải phóng nút (giây).
        seed: Hạt giống ngẫu nhiên cho SUMO.
        max_steps: Số bước mô phỏng tối đa (nếu None, chạy hết khung giờ gốc 3600s).
        gui: Bật giao diện đồ họa SUMO-GUI nếu True.
        output_dir: Thư mục lưu kết quả thô và tổng hợp.

    Trả về:
        Dictionary chứa toàn bộ 5 metrics và thông tin phân tích mô phỏng.
    """
    # 1. Đọc thông tin kịch bản
    scenario = read_scenario(scenario_name, data_dir=data_dir)
    tls_id = scenario["tls_id"]
    green_phases = scenario["green_phases"]
    incoming_lanes = scenario["incoming_lanes"]
    begin_time = scenario["begin_time"]
    nominal_end_time = scenario["end_time"]
    total_demand = scenario["total_demand"]

    # Xác định thời gian kết thúc: nếu max_steps được truyền vào, giới hạn thời gian chạy
    if max_steps is not None and max_steps > 0:
        end_time = begin_time + float(max_steps)
    else:
        end_time = nominal_end_time

    # Chuẩn bị thư mục xuất kết quả
    if output_dir is None:
        controller_name = getattr(controller, "__class__", type(controller)).__name__ if controller else "FixedTime"
        output_dir = os.path.join("results", "runs", f"{scenario_name}_{controller_name}_seed{seed}")
    os.makedirs(output_dir, exist_ok=True)
    tripinfo_path = os.path.join(output_dir, "tripinfo.xml")

    # 2. Khởi tạo controller nếu chưa truyền vào
    if controller is None:
        controller = FixedTimeController(
            green_phases=green_phases,
            green_time=green_time,
            yellow_time=yellow_time,
            all_red_time=all_red_time,
            tls_id=tls_id,
        )

    # 3. Khởi tạo bộ thu thập metrics và nhật ký chuyển pha
    collector = MetricsCollector(incoming_lanes=incoming_lanes, total_demand=total_demand)
    phase_log: List[Dict[str, Any]] = []

    # 4. Chuẩn bị câu lệnh khởi chạy SUMO
    sumo_binary = "sumo-gui" if gui else "sumo"
    sumo_cmd = [
        sumo_binary,
        "-c", scenario["cfg_path"],
        "--seed", str(seed),
        "--tripinfo-output", tripinfo_path,
        "--no-step-log", "true",
        "--time-to-teleport", "-1",  # Tắt teleport để giữ nguyên bản chất tắc nghẽn
        "--collision.action", "warn",
        "--duration-log.disable", "false",
    ]

    # 5. Mở kết nối backend mô phỏng (libsumo hoặc traci)
    traci = get_simulation_backend(gui=gui)
    traci.start(sumo_cmd)

    sumo_backend_name = "libsumo" if getattr(traci, "__name__", "") == "libsumo" else "traci"
    sumo_version = "unknown"
    try:
        ver = traci.getVersion()
        if isinstance(ver, (tuple, list)) and len(ver) >= 2:
            sumo_version = f"{ver[1]} (API v{ver[0]})"
        else:
            sumo_version = str(ver)
    except Exception:
        pass

    try:
        current_sim_time = traci.simulation.getTime()

        # Ghi nhận trạng thái kích hoạt pha ban đầu
        initial_state = controller.get_current_state()
        traci.trafficlight.setRedYellowGreenState(tls_id, initial_state)
        phase_log.append({
            "step_time": current_sim_time,
            "event": "start_simulation",
            "phase_index": controller.current_phase_idx,
            "state": initial_state,
        })

        # 6. Vòng lặp từng bước mô phỏng
        while current_sim_time < end_time:
            # A. Bước tiến của bộ điều khiển tín hiệu đèn
            step_res = controller.step(step_duration=1.0)
            state = step_res[0]
            changed = step_res[1]
            event_name = step_res[2]
            decision_info = step_res[3] if len(step_res) > 3 else None

            traci.trafficlight.setRedYellowGreenState(tls_id, state)

            # Ghi nhận toàn bộ sự kiện chuyển pha VÀ mọi quyết định RL (bao gồm cả KEEP)
            if (changed and event_name) or decision_info:
                log_entry: Dict[str, Any] = {
                    "step_time": current_sim_time,
                    "event": event_name or ("rl_decision_keep" if decision_info else "phase_step"),
                    "phase_index": getattr(controller, "current_phase_idx", -1),
                    "state": state,
                }
                if decision_info:
                    log_entry["decision"] = decision_info
                phase_log.append(log_entry)

            # B. Tiến mô phỏng 1 bước thời gian
            traci.simulationStep()
            current_sim_time = traci.simulation.getTime()

            # C. Thu thập dữ liệu giao thông ngay sau khi bước mô phỏng vừa hoàn thành
            step_queue = sum(
                traci.lane.getLastStepHaltingNumber(lane_id)
                for lane_id in incoming_lanes
            )
            step_waiting_time = sum(
                traci.lane.getWaitingTime(lane_id)
                for lane_id in incoming_lanes
            )
            running_count = traci.vehicle.getIDCount()
            inserted_step = traci.simulation.getDepartedNumber()
            arrived_step = traci.simulation.getArrivedNumber()
            teleport_step = traci.simulation.getStartingTeleportNumber()
            collision_step = traci.simulation.getCollidingVehiclesNumber()
            emergency_step = traci.simulation.getEmergencyStoppingVehiclesNumber()

            collector.record_step(
                sim_time=current_sim_time,
                step_queue=step_queue,
                running_count=running_count,
                inserted_step=inserted_step,
                arrived_step=arrived_step,
                teleported_step=teleport_step,
                collisions_step=collision_step,
                emergency_step=emergency_step,
            )

            # D. Chuyển dữ liệu quan sát về controller để tích lũy reward SMDP sau bước thực thi
            if hasattr(controller, "observe_step"):
                try:
                    controller.observe_step(step_queue=step_queue, step_waiting_time=step_waiting_time)
                except TypeError:
                    controller.observe_step(step_queue=step_queue)

        # E. Xử lý chuyển tiếp cuối kỳ cho mô hình học tăng cường (Time-limit Truncation)
        if hasattr(controller, "finalize_episode"):
            controller.finalize_episode()

    finally:
        # 7. Đóng TraCI an toàn
        traci.close()

    # 8. Tổng hợp metrics và xuất dữ liệu thô
    summary = collector.get_summary(
        tripinfo_path=tripinfo_path,
        begin_time=begin_time,
        end_time=end_time,
    )

    # Phân loại và trích xuất cấu hình điều khiển chính xác, tránh gán green_time cố định cho các bộ điều khiển thích ứng
    is_fixed = hasattr(controller, "green_time") and getattr(controller, "green_time") is not None
    if is_fixed:
        controller_params = {
            "control_type": "fixed_time",
            "green_time": float(getattr(controller, "green_time")),
            "cycle_time": getattr(controller, "cycle_time", None),
        }
    else:
        controller_params = {
            "control_type": "adaptive",
            "decision_interval": getattr(controller, "decision_interval", None),
            "min_green": getattr(controller, "min_green", None),
            "max_green": getattr(controller, "max_green", None),
            "max_red": getattr(controller, "max_red", None),
        }

    # Trích xuất định danh model checkpoint (đường dẫn và checksum)
    model_path = getattr(controller, "model_path", None)
    model_hash = getattr(controller, "model_hash", None)
    if model_path and not model_hash and os.path.isfile(model_path):
        try:
            import hashlib
            with open(model_path, "rb") as f_bin:
                model_hash = hashlib.sha256(f_bin.read()).hexdigest()[:16]
        except Exception:
            model_hash = None

    # Thêm cấu hình chạy vào bản tổng kết phục vụ đối chuẩn tái lập 100%
    run_config = {
        "scenario": scenario_name,
        "controller": controller.__class__.__name__,
        "seed": seed,
        **controller_params,
        "yellow_time": getattr(controller, "yellow_time", yellow_time),
        "all_red_time": getattr(controller, "all_red_time", all_red_time),
        "begin_time": begin_time,
        "end_time": end_time,
        "total_demand": total_demand,
        "incoming_lanes": incoming_lanes,
        "sumo_backend": sumo_backend_name,
        "sumo_version": sumo_version,
        "checkpoint_path": model_path,
        "checkpoint_hash": model_hash,
        "checkpoint_metadata": getattr(controller, "model_metadata", None),
    }
    summary["config"] = run_config

    # Lưu tệp summary.json, phase_log.json và run_config.json
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "phase_log.json"), "w", encoding="utf-8") as f:
        json.dump(phase_log, f, indent=2, ensure_ascii=False)

    with open(os.path.join(output_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    return summary


if __name__ == "__main__":
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Chạy mô phỏng SUMO và thu thập 5 chỉ số giao thông.")
    parser.add_argument("--scenario", type=str, default="cologne1", choices=["cologne1", "ingolstadt1"], help="Tên kịch bản")
    parser.add_argument("--green-time", type=float, default=30.0, help="Thời gian xanh cố định (s)")
    parser.add_argument("--yellow-time", type=float, default=3.0, help="Thời gian vàng (s)")
    parser.add_argument("--all-red-time", type=float, default=1.0, help="Thời gian đỏ toàn phần (s)")
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống ngẫu nhiên")
    parser.add_argument("--max-steps", type=int, default=None, help="Số bước tối đa (mặc định: hết khung giờ 1h)")
    parser.add_argument("--gui", action="store_true", help="Bật giao diện SUMO-GUI")
    parser.add_argument("--output-dir", type=str, default=None, help="Thư mục xuất kết quả")

    args = parser.parse_args()

    print(f"=== BẮT ĐẦU MÔ PHỎNG: {args.scenario} (Seed {args.seed}) ===")
    result = run_simulation(
        scenario_name=args.scenario,
        green_time=args.green_time,
        yellow_time=args.yellow_time,
        all_red_time=args.all_red_time,
        seed=args.seed,
        max_steps=args.max_steps,
        gui=args.gui,
        output_dir=args.output_dir,
    )

    print("\n=== KẾT QUẢ TỔNG HỢP (5 CORE METRICS) ===")
    print(f"1. Average Travel Time: {result['average_travel_time']} s")
    print(f"2. Average Waiting Time: {result['average_waiting_time']} s")
    print(f"3. Average Time Loss: {result['average_time_loss']} s")
    print(f"4. Mean Queue Length: {result['mean_queue_length']} xe (Max: {result['max_queue_length']} xe)")
    print(f"5. Throughput: {result['throughput']} xe hoàn thành")
    print(f"   - Tổng nhu cầu: {result['total_demand']} xe | Đã vào mạng: {result['inserted_vehicles']} xe")
    print(f"   - Xe còn trên mạng: {result['running_vehicles_at_end']} xe | Xe chờ vào: {result['waiting_to_enter']} xe")
    print(f"   - Tỷ lệ hoàn thành: {result['completion_rate_pct']} %")
