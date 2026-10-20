"""Aggregates and formats comparison benchmarks across all three traffic signal control algorithms."""

import math
import json
import os
from typing import Any, Dict, Optional

from src.evaluation.experiment_runner import normalize_algo_name
from src.metrics import REPORT_COMPLETION_THRESHOLD, service_status


def load_json(filepath: str) -> Dict[str, Any]:
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _stat(metrics, key, statistic="mean", default=None):
    entry = metrics.get(key, {})
    if entry.get("available") is False:
        return default
    value = entry.get(statistic, default)
    return value if isinstance(value, (int, float)) and math.isfinite(value) else default


def _format_mean_std(result, mean_key, std_key):
    mean, std = result.get(mean_key), result.get(std_key)
    return f"{mean:.2f} ± {std:.2f}" if mean is not None and std is not None else "N/A"


def generate_comparison_report(
    eval_dir: str = "results/evaluation",
    summary_file: Optional[str] = None,
    output_report: str = "results/evaluation/benchmark_comparison.md",
    *,
    phase6_summary: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Combines benchmark metrics into a unified comparison report."""
    os.makedirs(eval_dir, exist_ok=True)

    combined_results: Dict[str, Dict[str, Dict[str, Any]]] = {}

    summary_path = summary_file or phase6_summary or "results/evaluation/benchmark_summary.json"
    benchmark_data = load_json(summary_path)
    algos_display = ["Fixed-Time", "Max-Pressure", "Q-Learning"]

    if benchmark_data:
        for scen, raw_algos in benchmark_data.items():
            combined_results[scen] = {}
            algos = {normalize_algo_name(k): v for k, v in raw_algos.items()}
            for algo in algos_display:
                metrics = algos.get(algo)
                if not metrics:
                    combined_results[scen][algo] = {
                        "status": "NOT_RUN",
                        "error_messages": ["Chưa có dữ liệu thực nghiệm."],
                    }
                    continue
                if metrics.get("status") == "FAILED" or not metrics.get("is_valid", True):
                    err_msgs = metrics.get("error_messages", ["Thực nghiệm thất bại"])
                    combined_results[scen][algo] = {
                        "status": "FAILED",
                        "error_messages": err_msgs,
                    }
                    continue

                col_sum = _stat(metrics, "collision_count", "sum", _stat(metrics, "collision_count", "max"))
                col_max = _stat(metrics, "collision_count", "max", col_sum)
                tel_sum = _stat(metrics, "teleport_count", "sum", _stat(metrics, "teleport_count", "max"))
                tel_max = _stat(metrics, "teleport_count", "max", tel_sum)
                cr = _stat(metrics, "completion_rate_pct")
                safety_stat = metrics.get("safety_status", "UNKNOWN")
                service_stat = "NO_DEMAND" if metrics.get("service_status") == "NO_DEMAND" else service_status(cr / 100.0 if cr is not None else None, REPORT_COMPLETION_THRESHOLD)

                flat_entry: Dict[str, Any] = {
                    "mean_waiting_time": _stat(metrics, "tripinfo_mean_waiting_time", default=_stat(metrics, "mean_waiting_time")),
                    "waiting_time_std": metrics.get("tripinfo_mean_waiting_time", {}).get("std", 0.0),
                    "system_mean_delay": metrics.get("system_mean_delay", {}).get("mean", 0.0),
                    "system_delay_std": metrics.get("system_mean_delay", {}).get("std", 0.0),
                    "mean_queue_length": metrics.get("mean_queue_length", {}).get("mean", 0.0),
                    "queue_length_std": metrics.get("mean_queue_length", {}).get("std", 0.0),
                    "total_completed_vehicles": _stat(metrics, "total_completed_vehicles"),
                    "completion_rate_pct": cr,
                    "mean_travel_time": metrics.get("tripinfo_mean_travel_time", {}).get("mean", 0.0),
                    "total_time_loss": metrics.get("tripinfo_total_time_loss", {}).get("mean", 0.0),
                    "collision_sum": col_sum,
                    "collision_max": col_max,
                    "teleport_sum": tel_sum,
                    "teleport_max": tel_max,
                    "safety_status": safety_stat,
                    "service_status": service_stat,
                    "status": metrics.get("status", "SUCCESS"),
                    "effective_config": metrics.get("effective_config"),
                    "failed_count": metrics.get("failed_count", 0),
                    "episode_reward": metrics.get("episode_reward", {}).get("mean", "-"),
                    "improvement_waiting_pct": metrics.get("improvements_vs_baseline", {}).get("tripinfo_mean_waiting_time_improvement_pct"),
                    "comparison_unavailable_reason": metrics.get("comparison_unavailable_reason"),
                }
                combined_results[scen][algo] = flat_entry

    # Generate Markdown Report
    lines = [
        "# BÁO CÁO TIẾN ĐỘ & SO SÁNH CÁC PHƯƠNG PHÁP ĐIỀU KHIỂN TÍN HIỆU GIAO THÔNG",
        "",
        "**Dự án:** TrafficSignalControl — SUMO Reinforcement Learning & Baseline Benchmark  ",
        "**Mục tiêu:** Đánh giá và đối chuẩn hiệu năng các thuật toán điều khiển đèn giao thông.  ",
        "",
        "---",
        "",
    ]

    for scen, scen_res in combined_results.items():
        if not scen_res:
            continue
        lines.append(f"## Kịch bản mạng lưới: `{scen.upper()}`")
        lines.append("")

        for algo, result in scen_res.items():
            if result.get("status") == "PARTIAL":
                lines.append(f"- {algo}: PARTIAL; {result.get('failed_count', 0)} failed run(s). Statistics use successful runs only.")
            if result.get("comparison_unavailable_reason"):
                lines.append(f"- {algo}: {result['comparison_unavailable_reason']}")
        lines.append("")
        lines.append("| Thuật toán | Completed Wait Time (s) | System Delay (s) | Queue (xe) | Throughput (xe) | Completion (%) | Va chạm / Teleport |")
        lines.append("|---|---|---|---|---|---|---|")

        for algo in algos_display:
            r = scen_res.get(algo, {})
            algo_status = r.get("status", "NOT_RUN")
            if algo_status == "FAILED":
                errs = r.get("error_messages", ["Lỗi không xác định"])
                err_text = "; ".join(errs)
                lines.append(
                    f"| **{algo}** | FAILED: {err_text} | - | - | - | - | - |"
                )
                continue
            elif algo_status == "NOT_RUN":
                lines.append(
                    f"| **{algo}** | - | - | - | - | - | - |"
                )
                continue

            wt_str = _format_mean_std(r, "mean_waiting_time", "waiting_time_std")
            sys_d = r.get("system_mean_delay")
            sys_d_str = _format_mean_std(r, "system_mean_delay", "system_delay_std")
            ql_str = _format_mean_std(r, "mean_queue_length", "queue_length_std")

            tp = r.get("total_completed_vehicles")
            tp_str = f"{tp:.0f}" if isinstance(tp, (int, float)) else ("N/A" if tp is None else str(tp))
            cr = r.get("completion_rate_pct", 0.0)
            cr_str = f"{cr:.1f}%" if cr is not None else "N/A"
            col_sum = r.get("collision_sum", 0)
            col_max = r.get("collision_max", 0)
            tel_sum = r.get("teleport_sum", 0)

            if (col_sum or 0) > 0 or (tel_sum or 0) > 0:
                col_str = f"⚠️ {col_sum} / {tel_sum} (max {col_max})"
            else:
                col_str = "0 / 0" if col_sum is not None and tel_sum is not None else "N/A"

            lines.append(
                f"| **{algo}** | {wt_str} | {sys_d_str} | {ql_str} | {tp_str} | {cr_str} | {col_str} |"
            )

        lines.append("")

    os.makedirs(os.path.dirname(os.path.abspath(output_report)), exist_ok=True)
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

