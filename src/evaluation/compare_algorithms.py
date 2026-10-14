"""Aggregates and formats comparison benchmarks across all four traffic signal control algorithms."""

import json
import os
from typing import Any, Dict, List, Optional


def load_json(filepath: str) -> Dict[str, Any]:
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def generate_comparison_report(
    eval_dir: str = "results/evaluation",
    experiments_summary: str = "results/experiments/benchmark_summary.json",
    phase6_summary: str = "results/evaluation/benchmark_summary.json",
    output_report: str = "results/evaluation/benchmark_comparison.md",
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Combines benchmark metrics into a unified comparison report."""
    os.makedirs(eval_dir, exist_ok=True)

    combined_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    # Prefer Phase 6 comprehensive benchmark_summary.json if available
    p6_data = load_json(phase6_summary)
    if p6_data:
        for scen, algos in p6_data.items():
            combined_results[scen] = {}
            for algo, metrics in algos.items():
                if algo not in ("Fixed-Time", "Max-Pressure", "Q-Learning"):
                    continue
                # Flatten stats for reporting
                flat_entry: Dict[str, Any] = {
                    "mean_waiting_time": metrics.get("tripinfo_mean_waiting_time", {}).get("mean", metrics.get("mean_waiting_time", {}).get("mean", 0.0)),
                    "waiting_time_std": metrics.get("tripinfo_mean_waiting_time", {}).get("std", 0.0),
                    "mean_queue_length": metrics.get("mean_queue_length", {}).get("mean", 0.0),
                    "queue_length_std": metrics.get("mean_queue_length", {}).get("std", 0.0),
                    "total_completed_vehicles": int(metrics.get("total_completed_vehicles", {}).get("mean", 0)),
                    "completion_rate_pct": metrics.get("completion_rate_pct", {}).get("mean", 0.0),
                    "mean_travel_time": metrics.get("tripinfo_mean_travel_time", {}).get("mean", 0.0),
                    "total_time_loss": metrics.get("tripinfo_total_time_loss", {}).get("mean", 0.0),
                    "teleport_count": int(metrics.get("teleport_count", {}).get("mean", 0)),
                    "episode_reward": metrics.get("episode_reward", {}).get("mean", "-"),
                    "improvement_waiting_pct": metrics.get("improvements_vs_baseline", {}).get("tripinfo_mean_waiting_time_improvement_pct", 0.0),
                }
                combined_results[scen][algo] = flat_entry
    else:
        # Fallback to Phase 4 / Phase 5 separate files
        baseline_data = load_json(experiments_summary)
        scenarios = ["cologne1", "ingolstadt1"]

        for scen in scenarios:
            combined_results[scen] = {}
            scen_baselines = baseline_data.get(scen, {})
            for b_name in ["Fixed-Time", "Max-Pressure"]:
                if b_name in scen_baselines:
                    combined_results[scen][b_name] = scen_baselines[b_name]

            ql_file = os.path.join(eval_dir, f"{scen}_q_learning_eval.json")
            ql_data = load_json(ql_file)
            if ql_data:
                combined_results[scen]["Q-Learning"] = ql_data

    # Generate Markdown Report
    lines = [
        "# BÁO CÁO TIẾN ĐỘ & SO SÁNH CÁC PHƯƠNG PHÁP ĐIỀU KHIỂN TÍN HIỆU GIAO THÔNG",
        "",
        "**Dự án:** TrafficSignalControl — SUMO Reinforcement Learning & Baseline Benchmark  ",
        "**Mục tiêu:** Đánh giá và đối chuẩn hiệu năng các thuật toán điều khiển đèn giao thông.  ",
        "**Điều kiện mô phỏng:** Chu kỳ 3600s, pha vàng chuyển tiếp = 3.0s, đèn xanh tối thiểu (min_green) = 5.0s, dữ liệu hành trình TripInfo chuẩn.",
        "",
        "---",
        "",
    ]

    for scen, scen_res in combined_results.items():
        if not scen_res:
            continue
        lines.append(f"## Kịch bản mạng lưới: `{scen.upper()}`")
        lines.append("")
        lines.append("| Thuật toán | Waiting Time (s) | Queue Length (xe) | Throughput (xe) | Completion Rate (%) | Time Loss (s) |")
        lines.append("|---|---|---|---|---|---|")

        algos_display = ["Fixed-Time", "Max-Pressure", "Q-Learning"]
        for algo in algos_display:
            if algo in scen_res:
                r = scen_res[algo]
                wt = r.get("mean_waiting_time", 0.0)
                ql = r.get("mean_queue_length", 0.0)
                tp = r.get("total_completed_vehicles", 0)
                cr = r.get("completion_rate_pct", 0.0)
                cr_str = f"{cr:.1f}%" if cr > 0 else "-"
                tl = r.get("total_time_loss", 0.0)
                lines.append(
                    f"| **{algo}** | {wt:.2f} | {ql:.2f} | {tp} | {cr_str} | {tl:.1f} |"
                )
        lines.append("")

    with open(output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also save as JSON
    out_json = os.path.join(eval_dir, "benchmark_comparison.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    print(f"\n[INFO] Report generated: '{output_report}'")
    print(f"[INFO] Comparison data saved: '{out_json}'")
    return combined_results


if __name__ == "__main__":
    generate_comparison_report()

