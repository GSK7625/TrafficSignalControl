"""Aggregates and formats comparison benchmarks across all three traffic signal control algorithms."""

import math
import json
import os
from typing import Any, Dict

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
    algos_display = ["Fixed-Time", "Max-Pressure", "Q-Learning"]

    if p6_data:
        for scen, algos in p6_data.items():
            combined_results[scen] = {}
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
                    "improvement_waiting_pct": metrics.get("improvements_vs_baseline", {}).get("tripinfo_mean_waiting_time_improvement_pct", 0.0),
                }
                combined_results[scen][algo] = flat_entry

    # Generate Markdown Report
    lines = [
        "# BÁO CÁO TIẾN ĐỘ & SO SÁNH CÁC PHƯƠNG PHÁP ĐIỀU KHIỂN TÍN HIỆU GIAO THÔNG",
        "",
        "**Dự án:** TrafficSignalControl — SUMO Reinforcement Learning & Baseline Benchmark  ",
        "**Mục tiêu:** Đánh giá và đối chuẩn hiệu năng các thuật toán điều khiển đèn giao thông.  ",
        "**Điều kiện mô phỏng:** Xem cấu hình hiệu lực của từng thuật toán bên dưới; dữ liệu cũ không có metadata được ghi N/A.",
        "",
        "---",
        "",
    ]

    for scen, scen_res in combined_results.items():
        if not scen_res:
            continue
        lines.append(f"## Kịch bản mạng lưới: `{scen.upper()}`")
        lines.append("")
        lines.append("| Thuật toán | Completed Wait Time (s)* | System Delay (s) | Queue (xe) | Throughput (xe) | Completion (%) | Va chạm / Teleport | Trạng thái |")
        lines.append("|---|---|---|---|---|---|---|---|")

        scenario_warnings = []
        for algo in algos_display:
            r = scen_res.get(algo, {})
            algo_status = r.get("status", "NOT_RUN")
            if algo_status == "FAILED":
                errs = r.get("error_messages", ["Lỗi không xác định"])
                err_text = "; ".join(errs)
                status_badge = "💥 FAILED"
                lines.append(
                    f"| **{algo}** | - | - | - | - | - | - | {status_badge} |"
                )
                scenario_warnings.append(f"- **{algo}**: Đánh giá thất bại: {err_text}")
                continue
            elif algo_status == "NOT_RUN":
                status_badge = "❓ CHƯA CHẠY"
                lines.append(
                    f"| **{algo}** | - | - | - | - | - | - | {status_badge} |"
                )
                scenario_warnings.append(f"- **{algo}**: Không có dữ liệu thực nghiệm (chưa hoàn thành hoặc bị lỗi).")
                continue

            wt = r.get("mean_waiting_time")
            wt_std = r.get("waiting_time_std")
            wt_str = f"{wt:.2f} ± {wt_std:.2f}" if (wt is not None and wt_std is not None) else "N/A"

            sys_d = r.get("system_mean_delay")
            sys_d_std = r.get("system_delay_std")
            sys_d_str = f"{sys_d:.2f} ± {sys_d_std:.2f}" if (sys_d is not None and sys_d_std is not None) else "N/A"

            ql = r.get("mean_queue_length")
            ql_std = r.get("queue_length_std")
            ql_str = f"{ql:.2f} ± {ql_std:.2f}" if (ql is not None and ql_std is not None) else "N/A"

            tp = r.get("total_completed_vehicles", 0)
            cr = r.get("completion_rate_pct", 0.0)
            cr_str = f"{cr:.1f}%" if cr is not None else "N/A"
            col_sum = r.get("collision_sum", 0)
            col_max = r.get("collision_max", 0)
            tel_sum = r.get("teleport_sum", 0)
            tel_max = r.get("teleport_max", 0)

            if (col_sum or 0) > 0 or (tel_sum or 0) > 0:
                col_str = f"⚠️ {col_sum} / {tel_sum} (max {col_max})"
                status_badge = "🚨 INCIDENTS"
                scenario_warnings.append(
                    f"- **{algo}**: Ghi nhận tổng cộng {col_sum} va chạm và {tel_sum} teleport qua các seed "
                    f"(lớn nhất {col_max} va chạm/lượt). Cần kiểm tra dọn nút giao và xung đột luồng."
                )
            else:
                col_str = "0 / 0" if col_sum is not None and tel_sum is not None else "N/A"
                if r.get("service_status") == "NO_DEMAND":
                    status_badge = "NO_DEMAND"
                elif cr is None:
                    status_badge = "NO_DATA"
                elif cr < REPORT_COMPLETION_THRESHOLD * 100.0:
                    status_badge = "⚠️ LOW_COMPLETION"
                    sys_d_display = f"{sys_d:.2f}s" if sys_d is not None else "N/A"
                    scenario_warnings.append(
                        f"- **{algo}**: Tỷ lệ hoàn thành thấp ({cr:.1f}%). "
                        f"Có hiện tượng nghẽn mạng ngoài cổng (System Delay {sys_d_display})."
                    )
                else:
                    status_badge = "✅ SAFE" if col_sum is not None and tel_sum is not None else "UNKNOWN"

            if r.get("failed_count"):
                scenario_warnings.append(f"- **{algo}**: PARTIAL, {r['failed_count']} seed thất bại; thống kê chỉ dùng lượt hợp lệ.")
            config = r.get("effective_config")
            if config:
                scenario_warnings.append(f"- **{algo}** — cấu hình hiệu lực: " + ", ".join(f"{key}={value}" for key, value in config.items()))
            lines.append(
                f"| **{algo}** | {wt_str} | {sys_d_str} | {ql_str} | {tp} | {cr_str} | {col_str} | {status_badge} |"
            )

        lines.append("")
        lines.append("> [!NOTE]")
        lines.append("> [*] **Lưu ý về Survivorship Bias:** Completed Wait Time chỉ tính trên các xe đã về đích. Khi Completion (%) < 98%, chỉ số này có thể thấp phiến diện do bỏ qua các xe đang kẹt trong mạng; System Delay và Throughput phản ánh khách quan toàn bộ nhu cầu giao thông.")

        if scenario_warnings:
            lines.append("")
            lines.append("> [!WARNING]")
            lines.append("> **Cảnh báo quan sát:**")
            for w in scenario_warnings:
                lines.append(f"> {w}")
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

