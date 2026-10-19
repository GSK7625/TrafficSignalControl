"""Comprehensive visualization and publication-quality chart generator for Phase 6.

Generates benchmark charts across Groups A, B, and C:
- Group A: Algorithm Comparisons (Waiting Time, Queue Length, Throughput, Time Loss, Completion Rate, Improvement %)
- Group B: Traffic Dynamics Over Time (Queue over time, Running vehicles, Cumulative completed, Green time allocation)
- Group C: Reinforcement Learning Dynamics (Q-Learning reward, Epsilon decay, RL Convergence)
"""

import os
from typing import Any, List, Optional
import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.compare_algorithms import load_json


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
    scenarios = list(data)
    algos_order = ["Fixed-Time", "Max-Pressure", "Q-Learning"]

    for scen in scenarios:
        scen_data = data[scen]
        active_algos = [a for a in algos_order if a in scen_data and scen_data[a].get("is_valid", True) and scen_data[a].get("status") != "FAILED"]
        if not active_algos:
            continue

        # Metric definitions keep filenames, units and scaling together.
        metric_charts = (
            ("tripinfo_mean_waiting_time", "waiting_time", "Average Waiting Time (seconds)",
             f"Average Waiting Time — {scen.upper()} (TripInfo Ground Truth)", "%.2fs", 1.0),
            ("mean_queue_length", "queue_length", "Average Queue Length (vehicles)",
             f"Average Queue Length — {scen.upper()}", "%.2f", 1.0),
            ("total_completed_vehicles", "throughput", "Completed Vehicles (veh)",
             f"Throughput (Completed Vehicles) — {scen.upper()}", "%.0f", 1.0),
            ("tripinfo_total_time_loss", "time_loss", "Total Time Loss (×1,000 seconds)",
             f"Completed Vehicles Total Time Loss — {scen.upper()} (TripInfo)", "%.1fk s", 1000.0),
            ("completion_rate_pct", "completion_rate", "Completion Rate (%)",
             f"Traffic Demand Completion Rate — {scen.upper()}", "%.1f%%", 1.0),
        )
        for key, suffix, ylabel, title, fmt, scale in metric_charts:
            available_algos = [a for a in active_algos if scen_data[a].get(key, {}).get("mean") is not None]
            if not available_algos:
                continue
            fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
            _draw_metric_bar(
                ax, available_algos,
                [scen_data[a][key]["mean"] / scale for a in available_algos],
                [(scen_data[a][key].get("std") or 0.0) / scale for a in available_algos],
                ylabel=ylabel, title=title, fmt=fmt,
                colors=[ALGO_COLORS.get(a, "#333333") for a in available_algos],
            )
            if suffix == "completion_rate":
                ax.set_ylim(0, 110)
            generated.append(_save_plot(fig, os.path.join(output_dir, f"{scen}_{suffix}_comparison.png")))

        # 6. Percentage Improvement vs Fixed-Time
        compared_algos = [a for a in active_algos if a != "Fixed-Time" and scen_data[a].get("improvements_vs_baseline", {}).get("tripinfo_mean_waiting_time_improvement_pct") is not None]
        if compared_algos:
            fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
            c_colors = [ALGO_COLORS.get(a, "#333333") for a in compared_algos]
            imps = [
                scen_data[a].get("improvements_vs_baseline", {}).get("tripinfo_mean_waiting_time_improvement_pct", 0.0)
                for a in compared_algos
            ]
            _draw_metric_bar(
                ax, compared_algos, imps,
                ylabel="Waiting Time Improvement (%)",
                title=f"Relative Improvement vs Fixed-Time — {scen.upper()}",
                fmt="%+.1f%%", colors=c_colors,
            )
            ax.axhline(0, color="gray", linewidth=1.0, linestyle="--")
            generated.append(_save_plot(fig, os.path.join(output_dir, f"{scen}_improvement_vs_fixed_time.png")))

    return generated


def plot_group_b_dynamics(
    runs_file: str = "results/evaluation/benchmark_runs.json",
    output_dir: str = "results/figures",
) -> List[str]:
    """Generates Group B: Time-series dynamic trajectories (Queue, Running, Cumulative completed)."""
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()
    runs = load_json(runs_file)
    if not isinstance(runs, list) or not runs:
        return []

    generated = []
    scenarios = sorted({run["scenario_name"] for run in runs})

    for scen in scenarios:
        allowed_algos = ["Fixed-Time", "Max-Pressure", "Q-Learning"]
        scen_runs = [
            r for r in runs
            if r.get("scenario_name") == scen
            and r.get("is_valid", False)
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
            dfs = []
            for r in algo_runs:
                r_dir = r.get("output_dir")
                if not r_dir:
                    continue
                csv_path = os.path.join(r_dir, "step_metrics.csv")
                if not os.path.isfile(csv_path):
                    for root, _, files in os.walk(r_dir):
                        if "step_metrics.csv" in files:
                            csv_path = os.path.join(root, "step_metrics.csv")
                            break

                if os.path.isfile(csv_path):
                    dfs.append(pd.read_csv(csv_path))

            if not dfs:
                continue

            combined = pd.concat(dfs, ignore_index=True)
            steps_unique = sorted(combined["step"].unique())
            sampled_steps = steps_unique[::15] if len(steps_unique) > 100 else steps_unique
            combined_sampled = combined[combined["step"].isin(sampled_steps)]

            grouped = combined_sampled.groupby("step").agg({
                "sim_time": "mean",
                "queue_length": ["mean", "std"],
                "running_vehicles": ["mean", "std"],
                "cumulative_completed_vehicles": ["mean", "std"],
                "pending_vehicles": ["mean", "std"],
            })

            times = grouped["sim_time"]["mean"]

            for key, _, _, _ in dynamics:
                if key not in figures:
                    figures[key] = plt.subplots(figsize=(10, 5), dpi=300)
                _, ax = figures[key]
                means = grouped[key]["mean"]
                stds = grouped[key]["std"].fillna(0)
                ax.plot(times, means, label=algo, color=color, linewidth=1.8, alpha=0.9)
                ax.fill_between(times, (means - stds).clip(lower=0), means + stds, color=color, alpha=0.15)

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
) -> List[str]:
    """Generates Group C: Reinforcement Learning training dynamics."""
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()
    generated = []

    for scen in ["cologne1", "ingolstadt1"]:
        ql_file = os.path.join(log_dir, f"q_learning_{scen}_log.json")
        ql_data = load_json(ql_file)

        # 11. RL Training Reward Progression
        if ql_data:
            ql_hist = ql_data.get("history", [])
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
                plt.tight_layout()
                p11 = os.path.join(output_dir, f"{scen}_rl_reward_progression.png")
                plt.savefig(p11)
                plt.close(fig)
                generated.append(p11)

        # 14. Epsilon Decay Schedule
        if ql_data and "history" in ql_data:
            ql_hist = ql_data["history"]
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
            plt.tight_layout()
            p14 = os.path.join(output_dir, f"{scen}_epsilon_schedule.png")
            plt.savefig(p14)
            plt.close(fig)
            generated.append(p14)

    return generated


def plot_benchmark_dashboard(
    summary_file: str = "results/evaluation/benchmark_summary.json",
    output_dir: str = "results/figures",
) -> List[str]:
    """Generates a single comprehensive 2x2 Dashboard per scenario combining key comparison metrics."""
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()
    data = load_json(summary_file)
    if not data:
        return []

    generated = []
    scenarios = list(data)
    algos_order = ["Fixed-Time", "Max-Pressure", "Q-Learning"]

    for scen in scenarios:
        scen_data = data[scen]
        active_algos = [a for a in algos_order if a in scen_data and scen_data[a].get("is_valid", True) and scen_data[a].get("status") != "FAILED"]
        if not active_algos:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300)

        dashboard_metrics = (
            ("tripinfo_mean_waiting_time", "mean_waiting_time", "Waiting Time (s)", "Average Waiting Time", "%.1fs"),
            ("mean_queue_length", None, "Queue Length (veh)", "Average Queue Length", "%.1f"),
            ("total_completed_vehicles", None, "Completed Vehicles", "Total Completed Vehicles", "%.0f"),
        )
        for ax, (key, fallback, ylabel, title, fmt) in zip(axes.flat, dashboard_metrics):
            values = []
            for algo in active_algos:
                stats = scen_data[algo].get(key, {})
                if stats.get("mean") is None and fallback:
                    stats = scen_data[algo].get(fallback, {})
                if stats.get("mean") is not None:
                    values.append((algo, stats["mean"], stats.get("std") or 0.0))
            _draw_metric_bar(
                ax, [v[0] for v in values], [v[1] for v in values], [v[2] for v in values],
                ylabel=ylabel, title=title, fmt=fmt,
                colors=[ALGO_COLORS.get(v[0], "#333333") for v in values],
                fontsize=8, padding=3, yerr_capsize=4, title_fontsize=11, ylabel_fontsize=10,
            )

        # 4. Relative Improvement vs Fixed-Time
        compared_algos = [a for a in active_algos if a != "Fixed-Time" and scen_data[a].get("improvements_vs_baseline", {}).get("tripinfo_mean_waiting_time_improvement_pct") is not None]
        if compared_algos:
            c_colors = [ALGO_COLORS.get(a, "#333333") for a in compared_algos]
            imps = [
                scen_data[a].get("improvements_vs_baseline", {}).get("tripinfo_mean_waiting_time_improvement_pct", 0.0)
                for a in compared_algos
            ]
            _draw_metric_bar(axes[1, 1], compared_algos, imps, ylabel="Improvement (%)", title="Waiting Time Improvement vs Fixed-Time", fmt="%+.1f%%", colors=c_colors, fontsize=8, padding=3, yerr_capsize=4, title_fontsize=11, ylabel_fontsize=10)
            axes[1, 1].axhline(0, color="gray", linewidth=0.8, linestyle="--")

        fig.suptitle(f"Benchmark Performance Dashboard — {scen.upper()}", fontsize=14, fontweight="bold", y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out_p = os.path.join(output_dir, f"{scen}_benchmark_dashboard.png")
        plt.savefig(out_p)
        plt.close(fig)
        generated.append(out_p)

    return generated


def generate_all_phase6_charts(output_dir: str = "results/figures", dashboard_only: bool = False) -> List[str]:
    """Generates benchmark charts. If dashboard_only is True, creates consolidated dashboards."""
    if dashboard_only:
        dashboards = plot_benchmark_dashboard(output_dir=output_dir)
        print(f"\n[INFO] Generated {len(dashboards)} consolidated dashboards in '{output_dir}'.")
        return dashboards

    all_charts = []
    all_charts.extend(plot_group_a_comparisons(output_dir=output_dir))
    all_charts.extend(plot_group_b_dynamics(output_dir=output_dir))
    all_charts.extend(plot_group_c_rl(output_dir=output_dir))
    all_charts.extend(plot_benchmark_dashboard(output_dir=output_dir))
    print(f"\n[INFO] Generated {len(all_charts)} publication-quality charts in '{output_dir}'.")
    return all_charts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Benchmark Comparison Charts")
    parser.add_argument("--output-dir", type=str, default="results/figures")
    parser.add_argument("--dashboard-only", action="store_true", help="Only generate consolidated dashboard figures")
    args = parser.parse_args()
    generate_all_phase6_charts(output_dir=args.output_dir, dashboard_only=args.dashboard_only)
