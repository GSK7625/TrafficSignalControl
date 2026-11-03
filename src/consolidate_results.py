"""Script tổng hợp và so sánh toàn diện 4 thuật toán: FT, MP, Q-learning, DQN
trên cả hai mạng giao thông cologne1 và ingolstadt1 qua 5 test seeds.
"""

import json
import os
import sys
from typing import Any, Dict


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    scenarios = ["cologne1", "ingolstadt1"]
    algos = [
        ("Fixed-Time", "results/fixed_time/{scen}_ft_summary.json"),
        ("Max-Pressure", "results/max_pressure/{scen}_mp_summary.json"),
        ("Tabular Q-Learning", "results/q_learning/{scen}_ql_summary.json"),
        ("Deep Q-Network (DQN)", "results/dqn/{scen}_dqn_summary.json"),
    ]

    metrics_keys = [
        ("average_waiting_time", "Waiting Time (s)"),
        ("average_travel_time", "Travel Time (s)"),
        ("average_time_loss", "Time Loss (s)"),
        ("mean_queue_length", "Mean Queue (xe)"),
        ("max_queue_length", "Max Queue (xe)"),
        ("throughput", "Throughput (xe)"),
    ]

    for scen in scenarios:
        print(f"\n=======================================================")
        print(f"BẢNG SO SÁNH TỔNG HỢP: {scen.upper()} (5 Test Seeds: [42, 101, 2024, 3407, 9999])")
        print(f"=======================================================")

        data = {}
        for algo_name, path_pattern in algos:
            p = path_pattern.format(scen=scen)
            if os.path.exists(p):
                data[algo_name] = load_summary(p)
            else:
                print(f"Cảnh báo: không tìm thấy {p}")

        ft_data = data.get("Fixed-Time", {})

        # In bảng Markdown
        header = "| Thuật toán | Waiting Time (s) | Travel Time (s) | Time Loss (s) | Mean Queue (xe) | Max Queue (xe) | Throughput (xe) | Xe chưa xong |"
        sep = "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |"
        print(header)
        print(sep)

        for algo_name, s in data.items():
            m = s["metrics"]
            v = s.get("vehicle_stats", {})
            uncompleted_stats = v.get("running_vehicles_at_end") or v.get("running_at_end") or {}
            uncompleted = uncompleted_stats.get("mean", 0)

            # Delta so với FT cho waiting time
            wt_mean = m["average_waiting_time"]["mean"]
            wt_std = m["average_waiting_time"]["stdev"]
            tt_mean = m["average_travel_time"]["mean"]
            tt_std = m["average_travel_time"]["stdev"]
            tl_mean = m["average_time_loss"]["mean"]
            tl_std = m["average_time_loss"]["stdev"]
            mq_mean = m["mean_queue_length"]["mean"]
            mq_std = m["mean_queue_length"]["stdev"]
            maxq_mean = m["max_queue_length"]["mean"]
            maxq_std = m["max_queue_length"]["stdev"]
            tp_mean = m["throughput"]["mean"]
            tp_std = m["throughput"]["stdev"]

            row = (
                f"| **{algo_name}** "
                f"| {wt_mean:.2f} ± {wt_std:.2f} "
                f"| {tt_mean:.2f} ± {tt_std:.2f} "
                f"| {tl_mean:.2f} ± {tl_std:.2f} "
                f"| {mq_mean:.2f} ± {mq_std:.2f} "
                f"| {maxq_mean:.1f} ± {maxq_std:.1f} "
                f"| {tp_mean:.1f} ± {tp_std:.1f} "
                f"| {uncompleted} |"
            )
            print(row)

        print("\n--- PHẦN TRĂM THAY ĐỔI SO VỚI FIXED-TIME (Δ% = (Algo - FT) / FT * 100%) ---")
        if ft_data:
            ft_m = ft_data["metrics"]
            for algo_name, s in data.items():
                if algo_name == "Fixed-Time":
                    continue
                m = s["metrics"]
                d_wt = ((m["average_waiting_time"]["mean"] - ft_m["average_waiting_time"]["mean"]) / ft_m["average_waiting_time"]["mean"]) * 100
                d_tt = ((m["average_travel_time"]["mean"] - ft_m["average_travel_time"]["mean"]) / ft_m["average_travel_time"]["mean"]) * 100
                d_tl = ((m["average_time_loss"]["mean"] - ft_m["average_time_loss"]["mean"]) / ft_m["average_time_loss"]["mean"]) * 100
                d_mq = ((m["mean_queue_length"]["mean"] - ft_m["mean_queue_length"]["mean"]) / ft_m["mean_queue_length"]["mean"]) * 100
                d_tp = ((m["throughput"]["mean"] - ft_m["throughput"]["mean"]) / ft_m["throughput"]["mean"]) * 100
                print(f"[{algo_name}]")
                print(f"  * Waiting Time : {d_wt:+.2f}%")
                print(f"  * Travel Time  : {d_tt:+.2f}%")
                print(f"  * Time Loss    : {d_tl:+.2f}%")
                print(f"  * Mean Queue   : {d_mq:+.2f}%")
                print(f"  * Throughput   : {d_tp:+.2f}%")


if __name__ == "__main__":
    main()
