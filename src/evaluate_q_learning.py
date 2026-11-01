"""Kịch bản đánh giá Tabular Q-learning trên bộ 5 Test Seeds [42, 101, 2024, 3407, 9999].

Chạy đầy đủ 1 giờ (3600 bước mô phỏng) trên cả cologne1 và ingolstadt1,
tính toán giá trị trung bình và độ lệch chuẩn (mean +- stdev) cho 5 metrics cốt lõi.
Chế độ đánh giá độc lập: epsilon = 0.0, learning_enabled = False, model checkpoint cố định.
"""

import json
import os
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

from src.controllers.q_learning import QLearningController
from src.data_inspector import read_scenario
from src.simulation_runner import run_simulation

TEST_SEEDS = [42, 101, 2024, 3407, 9999]


def evaluate_q_learning_scenario(
    scenario_name: str,
    model_path: Optional[str] = None,
    seeds: List[int] = TEST_SEEDS,
) -> Dict[str, Any]:
    """Chạy đánh giá Q-learning trên một kịch bản với danh sách hạt giống test đã chọn."""
    print(f"\n=======================================================")
    print(f"ĐÁNH GIÁ TABULAR Q-LEARNING: {scenario_name.upper()} (Seeds: {seeds})")
    print(f"=======================================================")

    scenario = read_scenario(scenario_name)
    runs_data: List[Dict[str, Any]] = []
    base_out_dir = os.path.join("results", "q_learning", scenario_name)
    os.makedirs(base_out_dir, exist_ok=True)

    if model_path is None:
        model_path = os.path.join("models", "q_learning", f"{scenario_name}_best.json")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Không tìm thấy model checkpoint: {model_path}")

    print(f"--> Nạp model từ: {model_path}")

    for seed_idx, seed in enumerate(seeds, 1):
        print(f"[{seed_idx}/{len(seeds)}] Đang chạy {scenario_name} với Test Seed {seed}...", end="", flush=True)
        t_start = time.time()
        seed_out_dir = os.path.join(base_out_dir, f"seed_{seed}")

        # Khởi tạo controller Q-learning ở chế độ Evaluation
        controller = QLearningController(
            green_phases=scenario["green_phases"],
            movements=scenario["movements"],
            decision_interval=5.0,
            min_green=20.0,
            max_green=45.0,
            yellow_time=3.0,
            all_red_time=1.0,
            max_red=90.0,
            epsilon=0.0,
            learning_enabled=False,
        )
        controller.load(model_path)
        controller.epsilon = 0.0
        controller.learning_enabled = False
        controller.reset()

        summary = run_simulation(
            scenario_name=scenario_name,
            controller=controller,
            seed=seed,
            max_steps=None,
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
    aggregated = {
        "scenario": scenario_name,
        "controller": "Tabular_Q_Learning",
        "model_path": model_path,
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
            "inserted_vehicles": calc_stats(inserted_vehs),
            "running_vehicles_at_end": calc_stats(running_vehs),
            "completion_rate_pct": calc_stats(completion_rates),
        },
        "anomalies": {
            "total_teleports": sum(r.get("teleport_count", 0) for r in runs_data),
            "total_collisions": sum(r.get("collision_count", 0) for r in runs_data),
            "total_emergency_stops": sum(r.get("emergency_stops", 0) for r in runs_data),
        },
        "runs": runs_data,
    }

    summary_file = os.path.join("results", "q_learning", f"{scenario_name}_ql_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, ensure_ascii=False)

    print(f"\n--> BẢNG TỔNG HỢP {scenario_name.upper()} (Tabular Q-Learning):")
    m = aggregated["metrics"]
    print(f"   - Average Travel Time: {m['average_travel_time']['mean']} +- {m['average_travel_time']['stdev']} s")
    print(f"   - Average Waiting Time: {m['average_waiting_time']['mean']} +- {m['average_waiting_time']['stdev']} s")
    print(f"   - Average Time Loss: {m['average_time_loss']['mean']} +- {m['average_time_loss']['stdev']} s")
    print(f"   - Mean Queue Length: {m['mean_queue_length']['mean']} +- {m['mean_queue_length']['stdev']} xe")
    print(f"   - Max Queue Length: {m['max_queue_length']['mean']} +- {m['max_queue_length']['stdev']} xe")
    print(f"   - Throughput: {m['throughput']['mean']} +- {m['throughput']['stdev']} xe")
    print(f"--> Đã lưu tệp kết quả: {summary_file}")

    return aggregated


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    evaluate_q_learning_scenario("cologne1")
    evaluate_q_learning_scenario("ingolstadt1")
