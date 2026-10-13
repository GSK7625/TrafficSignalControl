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
    output_report: str = "results/evaluation/benchmark_comparison.md",
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Combines Fixed-Time, Max-Pressure, Q-Learning, and DQN benchmark metrics into a unified report."""
    os.makedirs(eval_dir, exist_ok=True)
    baseline_data = load_json(experiments_summary)

    scenarios = ["cologne1", "ingolstadt1"]
    algos = ["Fixed-Time", "Max-Pressure", "Q-Learning", "DQN"]

    combined_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for scen in scenarios:
        combined_results[scen] = {}

        # 1. Load Fixed-Time and Max-Pressure from Phase 4
        scen_baselines = baseline_data.get(scen, {})
        for b_name in ["Fixed-Time", "Max-Pressure"]:
            if b_name in scen_baselines:
                combined_results[scen][b_name] = scen_baselines[b_name]

        # 2. Load Q-Learning evaluation
        ql_file = os.path.join(eval_dir, f"{scen}_q_learning_eval.json")
        ql_data = load_json(ql_file)
        if ql_data:
            combined_results[scen]["Q-Learning"] = ql_data

        # 3. Load DQN evaluation
        dqn_file = os.path.join(eval_dir, f"{scen}_dqn_eval.json")
        dqn_data = load_json(dqn_file)
        if dqn_data:
            combined_results[scen]["DQN"] = dqn_data

    # Generate Markdown Report
    lines = [
        "# BÁO CÁO SO SÁNH 4 THUẬT TOÁN ĐIỀU KHIỂN TÍN HIỆU GIAO THÔNG",
        "",
        "**Dự án:** TrafficSignalControl  ",
        "**Giai đoạn 5:** Q-Learning và Deep Q-Network (DQN) so sánh với Fixed-Time và Max-Pressure.  ",
        "**Cấu hình đánh giá:** 3600 giây mô phỏng SUMO, Seed 42, Yellow clearance = 3.0s, Min green = 5.0s, Tripinfo ground-truth.",
        "",
        "---",
        "",
    ]

    for scen in scenarios:
        lines.append(f"## Kịch bản: `{scen.upper()}`")
        lines.append("")
        lines.append("| Thuật toán | Waiting Time (s) | Queue Length (xe) | Throughput (xe) | Travel Time (s) | Time Loss (s) | Teleports | Reward |")
        lines.append("|---|---|---|---|---|---|---|---|")

        scen_res = combined_results.get(scen, {})
        for algo in ["Fixed-Time", "Max-Pressure", "Q-Learning", "DQN"]:
            if algo in scen_res:
                r = scen_res[algo]
                wt = r.get("mean_waiting_time", 0.0)
                ql = r.get("mean_queue_length", 0.0)
                tp = r.get("total_completed_vehicles", 0)
                tt = r.get("mean_travel_time", 0.0)
                tl = r.get("total_time_loss", 0.0)
                tel = r.get("teleport_count", 0)
                rew = r.get("episode_reward", "-")
                lines.append(
                    f"| **{algo}** | {wt:.2f} | {ql:.2f} | {tp} | {tt:.2f} | {tl:.1f} | {tel} | {rew} |"
                )
        lines.append("")

    with open(output_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Also save as JSON
    out_json = os.path.join(eval_dir, "benchmark_comparison.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2)

    print(f"\nReport generated: {output_report}")
    print(f"Comparison data saved: {out_json}")
    return combined_results


if __name__ == "__main__":
    generate_comparison_report()
