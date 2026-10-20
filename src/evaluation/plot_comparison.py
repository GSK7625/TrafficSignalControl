"""Comprehensive visualization and publication-quality chart generator for benchmark evaluations.

Generates benchmark charts across Groups A, B, and C:
- Group A: Algorithm Comparisons (Waiting Time, Queue Length, Throughput, Time Loss, Completion Rate)
- Group B: Traffic Dynamics Over Time (Queue over time, Running vehicles, Cumulative completed, Green time allocation)
- Group C: Reinforcement Learning Dynamics (Q-Learning reward, Epsilon decay, RL Convergence)
"""

import csv
import os
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np

from src.evaluation.compare_algorithms import load_json
from src.evaluation.experiment_runner import normalize_algo_name
from src.metrics import COMPARISON_METRICS


def set_plot_style():
    """Sets a clean, professional academic aesthetic style for generated charts."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#bcbcbc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#e5e5e5"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.75


# Standard algorithm color palette
ALGO_COLORS = {
    "Fixed-Time": "#1f77b4",     # Steel Blue
    "Max-Pressure": "#2ca02c",   # Forest Green
    "Q-Learning": "#ff7f0e",     # Amber Orange
}


def extract_metric_series(
    scen_data: Dict[str, Any],
    active_algos: List[str],
    metric_key: str,
    scale: float = 1.0,
    fallback_key: Optional[str] = None,
) -> Tuple[List[str], List[float], List[float], List[str]]:
    """Extracts (categories, means, stds, colors) for a metric across active algorithms."""
    categories, means, stds, colors = [], [], [], []
    for algo in active_algos:
        stats = scen_data.get(algo, {}).get(metric_key, {})
        if stats.get("mean") is None and fallback_key:
            stats = scen_data.get(algo, {}).get(fallback_key, {})
        mean = stats.get("mean")
        if mean is not None:
            categories.append(algo)
            means.append(float(mean) / scale)
            std = stats.get("std") or 0.0
            stds.append(float(std) / scale)
            colors.append(ALGO_COLORS.get(algo, "#333333"))
    return categories, means, stds, colors


def _draw_metric_bar(
    ax: plt.Axes,
    categories: List[str],
    means: List[float],
    stds: Optional[List[float]] = None,
    ylabel: str = "",
    title: str = "",
    fmt: str = "%.2f",
    colors: Optional[List[str]] = None,
    fontsize: int = 9,
    padding: int = 4,
    yerr_capsize: int = 5,
    title_fontsize: int = 12,
    ylabel_fontsize: int = 11,
) -> Any:
    """Draws a standardized bar chart with error bars and value labels."""
    bars = ax.bar(
        categories,
        means,
        yerr=stds,
        capsize=yerr_capsize,
        color=colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.6,
    )
    ax.bar_label(bars, padding=padding, fmt=fmt, fontsize=fontsize, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=ylabel_fontsize, fontweight="bold")
    pad = 12 if title_fontsize >= 12 else None
    ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=pad)
    ax.grid(True, axis="y")
    return bars


def _save_plot(fig: plt.Figure, filepath: str) -> str:
    """Standardized figure save and resource cleanup."""
    fig.tight_layout()
    fig.savefig(filepath)
    plt.close(fig)
    return filepath


def _load_step_metrics_csv(csv_path: str) -> List[Dict[str, float]]:
    """Loads step metrics from a CSV file into a list of row dictionaries."""
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                parsed = {
                    "step": int(row["step"]),
                    "sim_time": float(row["sim_time"]),
                    "queue_length": float(row.get("queue_length", 0.0)),
                    "running_vehicles": float(row.get("running_vehicles", 0.0)),
                    "cumulative_completed_vehicles": float(row.get("cumulative_completed_vehicles", 0.0)),
                    "pending_vehicles": float(row.get("pending_vehicles", 0.0)),
                }
                rows.append(parsed)
            except (ValueError, KeyError):
                continue
    return rows


def plot_group_a_comparisons(
    summary_file: str = "results/evaluation/benchmark_summary.json",
    output_dir: str = "results/figures",
) -> List[str]:
    """Generates Group A: Comparative bar charts with standard deviation error bars."""
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()
    data = load_json(summary_file)
    if not data:
        print(f"[WARN] Summary file '{summary_file}' not found or empty.")
        return []

    generated = []
    algos_order = ["Fixed-Time", "Max-Pressure", "Q-Learning"]

    for scen, scen_data in data.items():
        active_algos = [a for a in algos_order if a in scen_data and scen_data[a].get("is_valid", True) and scen_data[a].get("status") != "FAILED"]
        if not active_algos:
            continue

        for key, suffix, ylabel, title, fmt, scale in COMPARISON_METRICS:
            available_algos, means, stds, colors = extract_metric_series(scen_data, active_algos, key, scale=scale)
            if not available_algos:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
            _draw_metric_bar(
                ax, available_algos, means, stds,
                ylabel=ylabel, title=f"{title} — {scen.upper()}", fmt=fmt,
                colors=colors,
            )
            if suffix == "completion_rate":
                ax.set_ylim(0, 110)
            generated.append(_save_plot(fig, os.path.join(output_dir, f"{scen}_{suffix}_comparison.png")))

    return generated


def plot_group_b_dynamics(
    runs_file: str = "results/evaluation/benchmark_runs.json",
    output_dir: str = "results/figures",
    selected_records: Optional[List[dict]] = None,
) -> List[str]:
    """Generates Group B: Time-series dynamic trajectories (Queue, Running, Cumulative completed)."""
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()
    raw_runs = selected_records if selected_records is not None else load_json(runs_file)
    if not isinstance(raw_runs, list) or not raw_runs:
        return []

    allowed_algos = ["Fixed-Time", "Max-Pressure", "Q-Learning"]
    valid_runs = []
    for r in raw_runs:
        if not r.get("is_valid", False) or r.get("status") == "FAILED":
            continue
        norm_algo = normalize_algo_name(r.get("algorithm", ""))
        if norm_algo in allowed_algos:
            valid_runs.append({**r, "algorithm": norm_algo})

    if selected_records is not None:
        runs = valid_runs
    else:
        latest_cohorts = {}
        for r in valid_runs:
            key = (r.get("scenario_name"), r.get("algorithm"))
            cohort = r.get("cohort_fingerprint")
            if cohort:
                latest_cohorts[key] = cohort

        runs_by_key = {}
        for idx, r in enumerate(valid_runs):
            key = (r.get("scenario_name"), r.get("algorithm"))
            cohort = r.get("cohort_fingerprint")
            target_cohort = latest_cohorts.get(key)
            if target_cohort is not None and cohort != target_cohort:
                continue
            seed_key = r.get("seed", idx)
            runs_by_key[(key[0], key[1], seed_key)] = r

        runs = list(runs_by_key.values())

    if not runs:
        return []

    generated = []
    scenarios = sorted({run["scenario_name"] for run in runs})

    for scen in scenarios:
        scen_runs = [
            r for r in runs
            if r.get("scenario_name") == scen
            and r.get("algorithm") in allowed_algos
        ]
        if not scen_runs:
            continue

        dynamics = (
            ("queue_length", "Queue Length Trajectory Over Time", "Total Queue Length (vehicles)", "queue_length_over_time"),
            ("running_vehicles", "Active Running Vehicles in Network", "Running Vehicles", "running_vehicles_over_time"),
            ("cumulative_completed_vehicles", "Cumulative Completed Vehicles", "Completed Vehicles", "cumulative_completed_over_time"),
            ("pending_vehicles", "Pending Uninserted Vehicles", "Pending Vehicles", "pending_vehicles_over_time"),
        )
        figures = {}

        for algo in allowed_algos:
            algo_runs = [r for r in scen_runs if r.get("algorithm") == algo]
            if not algo_runs:
                continue

            color = ALGO_COLORS.get(algo, "#333333")
            by_step: Dict[int, Dict[str, List[float]]] = {}
            for r in algo_runs:
                csv_path = r.get("output_csv")
                if not csv_path or not os.path.isfile(csv_path):
                    r_dir = r.get("output_dir")
                    if r_dir:
                        candidate = os.path.join(r_dir, "step_metrics.csv")
                        if os.path.isfile(candidate):
                            csv_path = candidate

                if csv_path and os.path.isfile(csv_path):
                    for row in _load_step_metrics_csv(csv_path):
                        s = row["step"]
                        if s not in by_step:
                            by_step[s] = {
                                "sim_time": [], "queue_length": [], "running_vehicles": [],
                                "cumulative_completed_vehicles": [], "pending_vehicles": [],
                            }
                        for k in by_step[s]:
                            by_step[s][k].append(row[k])

            if not by_step:
                continue

            steps_unique = sorted(by_step.keys())
            sampled_steps = steps_unique[::15] if len(steps_unique) > 100 else steps_unique

            times = [float(np.mean(by_step[s]["sim_time"])) for s in sampled_steps]

            for key, _, _, _ in dynamics:
                if key not in figures:
                    figures[key] = plt.subplots(figsize=(10, 5), dpi=300)
                _, ax = figures[key]
                means = np.array([float(np.mean(by_step[s][key])) for s in sampled_steps])
                stds = np.array([
                    float(np.std(by_step[s][key], ddof=1)) if len(by_step[s][key]) > 1 else 0.0
                    for s in sampled_steps
                ])
                ax.plot(times, means, label=algo, color=color, linewidth=1.8, alpha=0.9)
                ax.fill_between(times, np.clip(means - stds, a_min=0, a_max=None), means + stds, color=color, alpha=0.15)

        for key, title, ylabel, suffix in dynamics:
            if key not in figures:
                continue
            fig, ax = figures[key]
            ax.set_title(f"{title} — {scen.upper()} (All Seeds Mean ± Std)", fontsize=12, fontweight="bold", pad=12)
            ax.set_xlabel("Simulation Time (seconds)", fontsize=11)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True)
            ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
            generated.append(_save_plot(fig, os.path.join(output_dir, f"{scen}_{suffix}.png")))

        # Combine seeds per TLS before drawing; no last-seed overwrite.
        allocations = {}
        for run in scen_runs:
            for tls_id, phases in run.get("phase_durations_per_tls", {}).items():
                values = allocations.setdefault((run["algorithm"], tls_id), {})
                for phase, duration in phases.items():
                    values[str(phase)] = values.get(str(phase), 0.0) + duration
        for (algorithm, tls_id), durations in allocations.items():
            total = sum(durations.values())
            if total <= 0:
                continue
            fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
            phases = sorted(durations, key=int)
            _draw_metric_bar(ax, [f"Phase {phase}" for phase in phases],
                             [durations[phase] / total * 100 for phase in phases],
                             ylabel="Green Time Allocation (%)",
                             title=f"Green Time Allocation — {algorithm} on {scen.upper()} ({tls_id})", fmt="%.1f%%")
            suffix = algorithm.lower().replace("-", "_").replace(" ", "_")
            tls_suffix = f"_{tls_id}" if sum(key[0] == algorithm for key in allocations) > 1 else ""
            generated.append(_save_plot(fig, os.path.join(output_dir, f"{scen}_{suffix}{tls_suffix}_phase_allocation.png")))

    return generated


def plot_group_c_rl(
    log_dir: str = "results/training",
    output_dir: str = "results/figures",
    scenarios: Optional[List[str]] = None,
) -> List[str]:
    """Generates Group C: Reinforcement Learning training dynamics."""
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()
    generated = []

    target_scenarios = scenarios or ["cologne1", "ingolstadt1", "cologne3"]
    for scen in target_scenarios:
        ql_file = os.path.join(log_dir, f"q_learning_{scen}_log.json")
        ql_data = load_json(ql_file)
        if not ql_data:
            continue
        ql_hist = ql_data.get("history", [])

        # 11. RL Training Reward Progression
        if ql_hist:
            fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
            ax.plot(
                [h["episode"] for h in ql_hist],
                [h["reward"] for h in ql_hist],
                marker="o",
                color=ALGO_COLORS["Q-Learning"],
                label="Q-Learning (Tabular)",
                linewidth=2.0,
            )
            ax.set_title(f"RL Training Reward Progression — {scen.upper()}", fontsize=12, fontweight="bold", pad=12)
            ax.set_xlabel("Training Episode", fontsize=11)
            ax.set_ylabel("Episode Cumulative Reward (Negative Pressure + Pending Penalty)", fontsize=11)
            ax.grid(True)
            ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
            generated.append(_save_plot(fig, os.path.join(output_dir, f"{scen}_rl_reward_progression.png")))

        # 14. Epsilon Decay Schedule
        if ql_hist:
            fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
            epsilons = [h.get("epsilon", 0.0) for h in ql_hist]
            eps = [h["episode"] for h in ql_hist]
            ax.plot(eps, epsilons, marker=".", color="#9467bd", linewidth=2.0, label="Exploration Epsilon (ε)")
            ax.set_title(f"Epsilon Decay Schedule — {scen.upper()}", fontsize=12, fontweight="bold", pad=12)
            ax.set_xlabel("Episode", fontsize=10)
            ax.set_ylabel("Epsilon (ε)", fontsize=10)
            ax.set_ylim(0.0, 1.05)
            ax.grid(True)
            ax.legend()
            generated.append(_save_plot(fig, os.path.join(output_dir, f"{scen}_epsilon_schedule.png")))

    return generated


def plot_benchmark_dashboard(
    summary_file: str = "results/evaluation/benchmark_summary.json",
    output_dir: str = "results/figures",
) -> List[str]:
    """Generates a comprehensive 1x3 Dashboard per scenario combining the 3 core metrics: Waiting Time, Queue Length, and Throughput."""
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()
    data = load_json(summary_file)
    if not data:
        return []

    generated = []
    algos_order = ["Fixed-Time", "Max-Pressure", "Q-Learning"]

    for scen, scen_data in data.items():
        active_algos = [a for a in algos_order if a in scen_data and scen_data[a].get("is_valid", True) and scen_data[a].get("status") != "FAILED"]
        if not active_algos:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)

        for ax, (key, _, ylabel, title, fmt, _) in zip(axes.flat, COMPARISON_METRICS[:3]):
            fallback = "mean_waiting_time" if key == "tripinfo_mean_waiting_time" else None
            algos, means, stds, colors = extract_metric_series(
                scen_data, active_algos, key, fallback_key=fallback
            )
            _draw_metric_bar(
                ax, algos, means, stds,
                ylabel=ylabel, title=title, fmt=fmt,
                colors=colors,
                fontsize=8, padding=3, yerr_capsize=4, title_fontsize=11, ylabel_fontsize=10,
            )

        fig.suptitle(f"Benchmark Performance Dashboard — {scen.upper()}", fontsize=13, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out_p = os.path.join(output_dir, f"{scen}_benchmark_dashboard.png")
        plt.savefig(out_p)
        plt.close(fig)
        generated.append(out_p)

    return generated


def generate_all_benchmark_charts(
    output_dir: str = "results/figures", dashboard_only: bool = False, *,
    summary_file: str = "results/evaluation/benchmark_summary.json",
    runs_file: str = "results/evaluation/benchmark_runs.json", log_dir: str = "results/training",
    selected_records: Optional[List[dict]] = None,
) -> List[str]:
    """Generates benchmark charts. If dashboard_only is True, creates consolidated dashboards."""
    if dashboard_only:
        dashboards = plot_benchmark_dashboard(summary_file=summary_file, output_dir=output_dir)
        print(f"\n[INFO] Generated {len(dashboards)} consolidated dashboards in '{output_dir}'.")
        return dashboards

    all_charts = []
    all_charts.extend(plot_group_a_comparisons(summary_file=summary_file, output_dir=output_dir))
    all_charts.extend(plot_group_b_dynamics(runs_file=runs_file, output_dir=output_dir, selected_records=selected_records))
    all_charts.extend(plot_group_c_rl(log_dir=log_dir, output_dir=output_dir))
    all_charts.extend(plot_benchmark_dashboard(summary_file=summary_file, output_dir=output_dir))
    print(f"\n[INFO] Generated {len(all_charts)} publication-quality charts in '{output_dir}'.")
    return all_charts


# Backwards compatibility alias
generate_all_phase6_charts = generate_all_benchmark_charts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Benchmark Comparison Charts")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    parser.add_argument("--dashboard-only", action="store_true", help="Only generate consolidated dashboard figures")
    parser.add_argument("--summary-file", default="results/evaluation/benchmark_summary.json")
    parser.add_argument("--runs-file", default="results/evaluation/benchmark_runs.json")
    parser.add_argument("--log-dir", default="results/training")
    args = parser.parse_args()
    generate_all_benchmark_charts(**vars(args))
