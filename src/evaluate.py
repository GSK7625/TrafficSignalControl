"""Multi-scenario Evaluation and Comparison module for Traffic Signal Control.

Coordinates simulation runs across multiple SUMO scenarios, collects the 5 core
traffic performance metrics, and exports reports:
  - Formatted Markdown comparison tables
  - JSON result summary (results/fixed_time_evaluation.json)
  - Visual comparison bar charts (results/fixed_time_metrics.png)
"""

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")  # Ensure headless backend
import matplotlib.pyplot as plt

from src.metrics import MetricsCollector
from src.simulation_runner import SimulationRunner


# Default scenario pair for dual-map evaluation
DEFAULT_SCENARIOS = ["cologne1", "ingolstadt1"]

# 5 Core Traffic Metrics metadata
METRIC_DEFINITIONS = [
    ("average_waiting_time", "Average Waiting Time", "s", "Thời gian dừng chờ trung bình (v < 0.1 m/s)"),
    ("average_travel_time", "Average Travel Time", "s", "Thời gian hành trình trung bình từ xuất phát đến đích"),
    ("mean_queue_length", "Mean Queue Length", "veh", "Độ dài hàng đợi trung bình trên các nhánh vào nút"),
    ("average_time_loss", "Average Time Loss", "s", "Thời gian tổn thất trung bình so với điều kiện lý tưởng"),
    ("throughput", "Throughput", "veh", "Số lượng phương tiện hoàn thành hành trình"),
]


def export_evaluation_json(
    results: Dict[str, Dict[str, Any]],
    output_path: str = "results/fixed_time_evaluation.json",
) -> str:
    """Exports evaluation results dictionary to a JSON file.

    Args:
        results: Dictionary mapping scenario names to metrics dictionary.
        output_path: Destination path for JSON file.

    Returns:
        Absolute path to the created JSON file.
    """
    abs_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return abs_path


def generate_comparison_markdown(results: Dict[str, Dict[str, Any]]) -> str:
    """Generates a formatted Markdown table comparing 5 metrics across scenarios.

    Args:
        results: Dictionary mapping scenario names to their metric dictionaries.

    Returns:
        Markdown string containing the formatted comparison table.
    """
    scenarios = list(results.keys())

    # Build header row
    header = ["| Metric | Unit |"] + [f" {scen} |" for scen in scenarios] + [" Mô tả |"]
    separator = ["| :--- | :--- |"] + [":---: |" for _ in scenarios] + [":--- |"]

    lines = [
        "## Bảng So sánh Hiệu năng Điều khiển Đèn Giao thông (Fixed Time - 30s Green)",
        "",
        "".join(header),
        "".join(separator),
    ]

    for key, display_name, unit, desc in METRIC_DEFINITIONS:
        row = [f"| **{display_name}** | {unit} |"]
        for scen in scenarios:
            val = results[scen].get(key, 0.0)
            if isinstance(val, (int,)):
                formatted_val = f"`{val}`"
            elif isinstance(val, (float,)):
                formatted_val = f"`{val:.2f}`"
            else:
                formatted_val = f"`{val}`"
            row.append(f" {formatted_val} |")
        row.append(f" {desc} |")
        lines.append("".join(row))

    # Add extra metadata row (Total Steps)
    row_steps = ["| **Simulation Steps** | step |"]
    for scen in scenarios:
        steps = results[scen].get("total_steps", 0)
        row_steps.append(f" `{steps}` |")
    row_steps.append(" Tổng số bước mô phỏng thực tế |")
    lines.append("".join(row_steps))

    return "\n".join(lines)


def plot_comparison_chart(
    results: Dict[str, Dict[str, Any]],
    output_path: str = "results/fixed_time_metrics.png",
) -> str:
    """Generates a multi-panel bar chart comparing 5 metrics across scenarios.

    Each metric is plotted in its own dedicated subplot to accommodate differing
    scales and units (seconds, vehicles).

    Args:
        results: Dictionary mapping scenario names to metrics dictionary.
        output_path: Target path for the output PNG image.

    Returns:
        Absolute path to the generated image file.
    """
    abs_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    scenarios = list(results.keys())
    metric_keys = [
        ("average_waiting_time", "Average Waiting Time (s)", "#2b5c8f"),
        ("average_travel_time", "Average Travel Time (s)", "#4575b4"),
        ("mean_queue_length", "Mean Queue Length (veh)", "#d73027"),
        ("average_time_loss", "Average Time Loss (s)", "#fc8d59"),
        ("throughput", "Throughput (completed vehicles)", "#1a9850"),
    ]

    # Create 1 row x 5 columns or 2 rows x 3 columns grid
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))

    scenario_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, (m_key, m_title, default_color) in enumerate(metric_keys):
        ax = axes[idx]
        values = [results[scen].get(m_key, 0.0) for scen in scenarios]

        bar_colors = [scenario_colors[i % len(scenario_colors)] for i in range(len(scenarios))]
        bars = ax.bar(scenarios, values, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.8)

        ax.set_title(m_title, fontsize=10, fontweight="bold", pad=10)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)

        # Bar value annotations
        for bar, val in zip(bars, values):
            y_val = bar.get_height()
            label_text = f"{int(val)}" if isinstance(val, int) or m_key == "throughput" else f"{val:.2f}"
            ax.annotate(
                label_text,
                xy=(bar.get_x() + bar.get_width() / 2, y_val),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

        # Slight headroom for annotations
        max_y = max(values) if values and max(values) > 0 else 1.0
        ax.set_ylim(0, max_y * 1.18)

    fig.suptitle(
        "So sánh 5 Chỉ số Giao thông Thuật toán Fixed Time (30s) trên Đa Kịch bản SUMO",
        fontsize=13,
        fontweight="bold",
        y=1.03,
    )
    plt.tight_layout()
    plt.savefig(abs_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return abs_path


def evaluate_scenario(
    scenario_name: str,
    max_steps: int = 3600,
    gui: bool = False,
    output_dir: str = "results",
    data_dir: str = "data",
) -> Dict[str, Any]:
    """Runs simulation and extracts 5 traffic metrics for a single scenario.

    Args:
        scenario_name: Name of scenario subdirectory under data_dir (e.g. 'cologne1').
        max_steps: Maximum simulation steps (default: 3600).
        gui: Whether to launch sumo-gui graphical interface.
        output_dir: Directory where tripinfo and results will be stored.
        data_dir: Root directory containing scenario folders.

    Returns:
        Dictionary containing the 5 traffic metrics and run details.
    """
    cfg_path = os.path.join(data_dir, scenario_name, f"{scenario_name}.sumocfg")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"SUMO config file not found for scenario '{scenario_name}': {cfg_path}")

    os.makedirs(output_dir, exist_ok=True)
    tripinfo_path = os.path.abspath(os.path.join(output_dir, f"{scenario_name}_tripinfo.xml"))

    collector = MetricsCollector()
    runner = SimulationRunner(cfg_path=cfg_path, tripinfo_output=tripinfo_path)

    # Set monitored lanes in collector
    collector.monitored_lanes = runner.incoming_lanes

    def step_callback(step_idx: int, state: str, current_queue: int) -> None:
        collector.record_step(step_time=float(step_idx), queue_length=float(current_queue))

    sim_result = runner.run(gui=gui, max_steps=max_steps, step_callback=step_callback)
    summary = collector.get_summary(tripinfo_path=tripinfo_path if os.path.isfile(tripinfo_path) else None)

    return {
        "scenario": scenario_name,
        "average_waiting_time": summary["average_waiting_time"],
        "average_travel_time": summary["average_travel_time"],
        "mean_queue_length": summary["mean_queue_length"],
        "max_queue_length": summary["max_queue_length"],
        "average_time_loss": summary["average_time_loss"],
        "throughput": summary["throughput"],
        "total_steps": sim_result.total_steps,
        "tripinfo_path": tripinfo_path,
    }


def evaluate_multiple(
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    max_steps: int = 3600,
    gui: bool = False,
    output_dir: str = "results",
    data_dir: str = "data",
) -> Dict[str, Dict[str, Any]]:
    """Evaluates multiple scenarios, generates comparison chart and JSON results.

    Args:
        scenarios: List of scenario names (defaults to ['cologne1', 'ingolstadt1']).
        max_steps: Maximum simulation steps per scenario.
        gui: Whether to launch graphical interface.
        output_dir: Output directory for reports and charts.
        data_dir: Root data directory.

    Returns:
        Dictionary mapping each scenario name to its metrics dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: Dict[str, Dict[str, Any]] = {}

    for scen in scenarios:
        res = evaluate_scenario(
            scenario_name=scen,
            max_steps=max_steps,
            gui=gui,
            output_dir=output_dir,
            data_dir=data_dir,
        )
        results[scen] = res

    # 1. Export JSON results
    json_path = os.path.join(output_dir, "fixed_time_evaluation.json")
    export_evaluation_json(results, output_path=json_path)

    # 2. Export comparison chart
    chart_path = os.path.join(output_dir, "fixed_time_metrics.png")
    plot_comparison_chart(results, output_path=chart_path)

    # 3. Export Markdown Report
    report_path = os.path.join(output_dir, "REPORT.md")
    markdown_content = generate_comparison_markdown(results)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
        f.write("\n\n")
        f.write("![Biểu đồ So sánh](fixed_time_metrics.png)\n")

    return results
