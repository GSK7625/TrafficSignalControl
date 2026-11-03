"""Kịch bản đánh giá Max-Pressure trên cùng bộ 5 hạt giống (Test Seeds) như Fixed-Time.

Chạy đầy đủ 1 giờ (3600 bước mô phỏng) trên cả cologne1 và ingolstadt1,
tính toán giá trị trung bình và độ lệch chuẩn (mean +- stdev) cho 5 metrics cốt lõi.
"""

import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

from src.controllers.max_pressure import MaxPressureController
from src.data_inspector import read_scenario
from src.simulation_runner import run_simulation

TEST_SEEDS = [42, 101, 2024, 3407, 9999]


def evaluate_max_pressure_scenario(scenario_name: str, seeds: List[int] = TEST_SEEDS) -> Dict[str, Any]:
    """Chạy đánh giá Max-Pressure trên một kịch bản với danh sách hạt giống đã chọn."""
    print(f"\n=======================================================")
    print(f"ĐÁNH GIÁ MAX-PRESSURE: {scenario_name.upper()} (Seeds: {seeds})")
    print(f"=======================================================")

    scenario = read_scenario(scenario_name)
    runs_data: List[Dict[str, Any]] = []
    base_out_dir = os.path.join("results", "max_pressure", scenario_name)
    os.makedirs(base_out_dir, exist_ok=True)

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"[{seed_idx}/{len(seeds)}] Đang chạy {scenario_name} với Seed {seed}...", end="", flush=True)
        t_start = time.time()
        seed_out_dir = os.path.join(base_out_dir, f"seed_{seed}")

        # Khởi tạo controller Max-Pressure riêng cho mỗi seed
        controller = MaxPressureController(
            green_phases=scenario["green_phases"],
            movements=scenario["movements"],
            min_green=20.0,
            max_green=45.0,
            yellow_time=3.0,
            all_red_time=1.0,
            max_red=90.0,
            tls_id=scenario["tls_id"],
        )

        summary = run_simulation(
            scenario_name=scenario_name,
            controller=controller,
            seed=seed,
            max_steps=None,  # Chạy đủ 3600 bước (1 giờ gốc)
            gui=False,
            output_dir=seed_out_dir,
        )
        t_elapsed = time.time() - t_start
        print(f" Xong ({t_elapsed:.1f}s) | Throughput: {summary['throughput']} | Waiting: {summary['average_waiting_time']}s")
        runs_data.append(summary)

    def calc_stats(values: List[Any]) -> Dict[str, Optional[float]]:
        valid = [float(v) for v in values if v is not None]
        if not valid:
            return {"mean": None, "stdev": None}
        if len(valid) == 1:
            return {"mean": round(valid[0], 2), "stdev": 0.0}
        return {
            "mean": round(statistics.mean(valid), 2),
            "stdev": round(statistics.stdev(valid), 2),
        }

    travel_times = [r["average_travel_time"] for r in runs_data]
    waiting_times = [r["average_waiting_time"] for r in runs_data]
    time_losses = [r["average_time_loss"] for r in runs_data]
    mean_queues = [r["mean_queue_length"] for r in runs_data]
    max_queues = [r["max_queue_length"] for r in runs_data]
    throughputs = [r["throughput"] for r in runs_data]
    inserted_vehs = [r["inserted_vehicles"] for r in runs_data]
    running_vehs = [r["running_vehicles_at_end"] for r in runs_data]
    completion_rates = [r["completion_rate_pct"] for r in runs_data]

    total_demand = runs_data[0]["total_demand"]

    aggregated: Dict[str, Any] = {
        "scenario": scenario_name,
        "controller": "MaxPressure",
        "seeds": seeds,
        "total_demand": total_demand,
        "metrics": {
            "average_travel_time": calc_stats(travel_times),
            "average_waiting_time": calc_stats(waiting_times),
            "average_time_loss": calc_stats(time_losses),
            "mean_queue_length": calc_stats(mean_queues),
            "max_queue_length": calc_stats(max_queues),
            "throughput": calc_stats(throughputs),
        },
        "vehicle_stats": {
            "total_demand": total_demand,
            "inserted_vehicles": calc_stats(inserted_vehs),
            "completed_vehicles": calc_stats(throughputs),
            "running_at_end": calc_stats(running_vehs),
            "running_vehicles_at_end": calc_stats(running_vehs),
            "completion_rate_pct": calc_stats(completion_rates),
        },
        "anomalies": {
            "teleport_count": sum(r["teleport_count"] for r in runs_data),
            "collision_count": sum(r["collision_count"] for r in runs_data),
            "emergency_stops": sum(r["emergency_stops"] for r in runs_data),
        },
        "runs": runs_data,
    }

    # Lưu file tổng kết
    summary_path = os.path.join("results", "max_pressure", f"{scenario_name}_mp_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    print(f"\n=> Kết quả tổng hợp MP lưu tại: {summary_path}")
    print(f"   - Average Travel Time: {aggregated['metrics']['average_travel_time']['mean']} +- {aggregated['metrics']['average_travel_time']['stdev']} s")
    print(f"   - Average Waiting Time: {aggregated['metrics']['average_waiting_time']['mean']} +- {aggregated['metrics']['average_waiting_time']['stdev']} s")
    print(f"   - Average Time Loss: {aggregated['metrics']['average_time_loss']['mean']} +- {aggregated['metrics']['average_time_loss']['stdev']} s")
    print(f"   - Mean Queue Length: {aggregated['metrics']['mean_queue_length']['mean']} +- {aggregated['metrics']['mean_queue_length']['stdev']} xe")
    print(f"   - Throughput: {aggregated['metrics']['throughput']['mean']} +- {aggregated['metrics']['throughput']['stdev']} xe")
    print(f"   - Tỷ lệ hoàn thành: {aggregated['vehicle_stats']['completion_rate_pct']['mean']}%")

    return aggregated


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    os.makedirs(os.path.join("results", "max_pressure"), exist_ok=True)
    cologne1_mp = evaluate_max_pressure_scenario("cologne1")
    ingolstadt1_mp = evaluate_max_pressure_scenario("ingolstadt1")


if __name__ == "__main__":
    main()
