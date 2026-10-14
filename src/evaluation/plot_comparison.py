"""Comprehensive visualization and publication-quality chart generator for Phase 6.

Generates benchmark charts across Groups A, B, and C:
- Group A: Algorithm Comparisons (Waiting Time, Queue Length, Throughput, Time Loss, Completion Rate, Improvement %)
- Group B: Traffic Dynamics Over Time (Queue over time, Running vehicles, Cumulative completed, Green time allocation)
- Group C: Reinforcement Learning Dynamics (Q-Learning reward, Epsilon decay, RL Convergence)
"""

import json
import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def load_json(filepath: str) -> Dict[str, Any]:
    if os.path.isfile(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


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
    scenarios = [s for s in ["cologne1", "ingolstadt1"] if s in data]
    algos_order = ["Fixed-Time", "Max-Pressure", "Q-Learning"]

    for scen in scenarios:
        scen_data = data[scen]
        active_algos = [a for a in algos_order if a in scen_data]
        if not active_algos:
            continue

        colors = [ALGO_COLORS.get(a, "#333333") for a in active_algos]

        # 1. Average Waiting Time
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
        means = [scen_data[a]["tripinfo_mean_waiting_time"]["mean"] for a in active_algos]
        stds = [scen_data[a]["tripinfo_mean_waiting_time"]["std"] for a in active_algos]
        bars = ax.bar(active_algos, means, yerr=stds, capsize=5, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
        ax.bar_label(bars, padding=4, fmt="%.2fs", fontsize=9, fontweight="bold")
        ax.set_ylabel("Average Waiting Time (seconds)", fontsize=11, fontweight="bold")
        ax.set_title(f"Average Waiting Time — {scen.upper()} (TripInfo Ground Truth)", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, axis="y")
        plt.tight_layout()
        p1 = os.path.join(output_dir, f"{scen}_waiting_time_comparison.png")
        plt.savefig(p1)
        plt.close(fig)
        generated.append(p1)

        # 2. Average Queue Length
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
        means = [scen_data[a]["mean_queue_length"]["mean"] for a in active_algos]
        stds = [scen_data[a]["mean_queue_length"]["std"] for a in active_algos]
        bars = ax.bar(active_algos, means, yerr=stds, capsize=5, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
        ax.bar_label(bars, padding=4, fmt="%.2f", fontsize=9, fontweight="bold")
        ax.set_ylabel("Average Queue Length (vehicles)", fontsize=11, fontweight="bold")
        ax.set_title(f"Average Queue Length — {scen.upper()}", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, axis="y")
        plt.tight_layout()
        p2 = os.path.join(output_dir, f"{scen}_queue_length_comparison.png")
        plt.savefig(p2)
        plt.close(fig)
        generated.append(p2)

        # 3. Throughput
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
        means = [scen_data[a]["total_completed_vehicles"]["mean"] for a in active_algos]
        stds = [scen_data[a]["total_completed_vehicles"]["std"] for a in active_algos]
        bars = ax.bar(active_algos, means, yerr=stds, capsize=5, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
        ax.bar_label(bars, padding=4, fmt="%.0f", fontsize=9, fontweight="bold")
        ax.set_ylabel("Completed Vehicles (veh / hour)", fontsize=11, fontweight="bold")
        ax.set_title(f"Throughput (Completed Vehicles) — {scen.upper()}", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, axis="y")
        plt.tight_layout()
        p3 = os.path.join(output_dir, f"{scen}_throughput_comparison.png")
        plt.savefig(p3)
        plt.close(fig)
        generated.append(p3)

        # 4. Average Time Loss
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
        means = [scen_data[a]["tripinfo_total_time_loss"]["mean"] / 1000.0 for a in active_algos]
        stds = [scen_data[a]["tripinfo_total_time_loss"]["std"] / 1000.0 for a in active_algos]
        bars = ax.bar(active_algos, means, yerr=stds, capsize=5, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
        ax.bar_label(bars, padding=4, fmt="%.1fk s", fontsize=9, fontweight="bold")
        ax.set_ylabel("Total Time Loss (×1,000 seconds)", fontsize=11, fontweight="bold")
        ax.set_title(f"Total Network Time Loss — {scen.upper()} (TripInfo)", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, axis="y")
        plt.tight_layout()
        p4 = os.path.join(output_dir, f"{scen}_time_loss_comparison.png")
        plt.savefig(p4)
        plt.close(fig)
        generated.append(p4)

        # 5. Completion Rate
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
        means = [scen_data[a]["completion_rate_pct"]["mean"] for a in active_algos]
        stds = [scen_data[a]["completion_rate_pct"]["std"] for a in active_algos]
        bars = ax.bar(active_algos, means, yerr=stds, capsize=5, color=colors, alpha=0.9, edgecolor="black", linewidth=0.6)
        ax.bar_label(bars, padding=4, fmt="%.1f%%", fontsize=9, fontweight="bold")
        ax.set_ylabel("Completion Rate (%)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 110)
        ax.set_title(f"Traffic Demand Completion Rate — {scen.upper()}", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, axis="y")
        plt.tight_layout()
        p5 = os.path.join(output_dir, f"{scen}_completion_rate_comparison.png")
        plt.savefig(p5)
        plt.close(fig)
        generated.append(p5)

        # 6. Percentage Improvement vs Fixed-Time
        fig, ax = plt.subplots(figsize=(8, 4.8), dpi=300)
        compared_algos = [a for a in active_algos if a != "Fixed-Time"]
        c_colors = [ALGO_COLORS.get(a, "#333333") for a in compared_algos]
        imps = [
            scen_data[a].get("improvements_vs_baseline", {}).get("tripinfo_mean_waiting_time_improvement_pct", 0.0)
            for a in compared_algos
        ]
        bars = ax.bar(compared_algos, imps, color=c_colors, alpha=0.9, edgecolor="black", linewidth=0.6)
        ax.bar_label(bars, padding=4, fmt="%+.1f%%", fontsize=9, fontweight="bold")
        ax.axhline(0, color="gray", linewidth=1.0, linestyle="--")
        ax.set_ylabel("Waiting Time Improvement (%)", fontsize=11, fontweight="bold")
        ax.set_title(f"Relative Improvement vs Fixed-Time — {scen.upper()}", fontsize=12, fontweight="bold", pad=12)
        ax.grid(True, axis="y")
        plt.tight_layout()
        p6 = os.path.join(output_dir, f"{scen}_improvement_vs_fixed_time.png")
        plt.savefig(p6)
        plt.close(fig)
        generated.append(p6)

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
    scenarios = ["cologne1", "ingolstadt1"]

    for scen in scenarios:
        allowed_algos = {"Fixed-Time", "Max-Pressure", "Q-Learning"}
        scen_runs = [
            r for r in runs
            if r.get("scenario_name") == scen
            and r.get("seed") == 42
            and r.get("is_valid", False)
            and r.get("algorithm") in allowed_algos
        ]
        if not scen_runs:
            continue

        fig_q, ax_q = plt.subplots(figsize=(10, 5), dpi=300)
        fig_r, ax_r = plt.subplots(figsize=(10, 5), dpi=300)
        fig_c, ax_c = plt.subplots(figsize=(10, 5), dpi=300)

        for r in scen_runs:
            algo = r["algorithm"]
            r_dir = r.get("output_dir", "")
            csv_path = os.path.join(r_dir, "step_metrics.csv")
            if not os.path.isfile(csv_path):
                for root, _, files in os.walk(r_dir):
                    if "step_metrics.csv" in files:
                        csv_path = os.path.join(root, "step_metrics.csv")
                        break

            if not os.path.isfile(csv_path):
                continue

            df = pd.read_csv(csv_path)
            color = ALGO_COLORS.get(algo, "#333333")

            # Downsample step curve for smooth rendering (step every 30s)
            df_sampled = df.iloc[::15, :] if len(df) > 100 else df
            times = df_sampled["sim_time"]

            # Queue length over time
            ax_q.plot(times, df_sampled["queue_length"], label=algo, color=color, linewidth=1.5, alpha=0.85)

            # Running vehicles over time
            ax_r.plot(times, df_sampled["running_vehicles"], label=algo, color=color, linewidth=1.5, alpha=0.85)

            # Cumulative completed vehicles over time
            ax_c.plot(times, df_sampled["cumulative_completed_vehicles"], label=algo, color=color, linewidth=1.8, alpha=0.85)

        # 7. Queue Length over Time
        ax_q.set_title(f"Queue Length Trajectory Over Time — {scen.upper()} (Seed 42)", fontsize=12, fontweight="bold", pad=12)
        ax_q.set_xlabel("Simulation Time (seconds)", fontsize=11)
        ax_q.set_ylabel("Total Queue Length (vehicles)", fontsize=11)
        ax_q.grid(True)
        ax_q.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
        fig_q.tight_layout()
        p7 = os.path.join(output_dir, f"{scen}_queue_length_over_time.png")
        fig_q.savefig(p7)
        plt.close(fig_q)
        generated.append(p7)

        # 8. Running Vehicles over Time
        ax_r.set_title(f"Active Running Vehicles in Network — {scen.upper()} (Seed 42)", fontsize=12, fontweight="bold", pad=12)
        ax_r.set_xlabel("Simulation Time (seconds)", fontsize=11)
        ax_r.set_ylabel("Running Vehicles", fontsize=11)
        ax_r.grid(True)
        ax_r.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
        fig_r.tight_layout()
        p8 = os.path.join(output_dir, f"{scen}_running_vehicles_over_time.png")
        fig_r.savefig(p8)
        plt.close(fig_r)
        generated.append(p8)

        # 9. Cumulative Completed Vehicles over Time
        ax_c.set_title(f"Cumulative Completed Vehicles — {scen.upper()} (Seed 42)", fontsize=12, fontweight="bold", pad=12)
        ax_c.set_xlabel("Simulation Time (seconds)", fontsize=11)
        ax_c.set_ylabel("Completed Vehicles", fontsize=11)
        ax_c.grid(True)
        ax_c.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
        fig_c.tight_layout()
        p9 = os.path.join(output_dir, f"{scen}_cumulative_completed_over_time.png")
        fig_c.savefig(p9)
        plt.close(fig_c)
        generated.append(p9)

        # 10. Green Time Allocation per Phase
        for r in scen_runs:
            durations = r.get("phase_durations", {})
            if not durations:
                continue
            first_tls = list(durations.keys())[0]
            phase_times = durations[first_tls]
            if not phase_times:
                continue

            fig_p, ax_p = plt.subplots(figsize=(7, 4), dpi=300)
            p_indices = sorted([int(k) for k in phase_times.keys()])
            p_vals = [phase_times.get(str(k), phase_times.get(k, 0.0)) for k in p_indices]
            total_sec = sum(p_vals) if sum(p_vals) > 0 else 1.0
            p_pcts = [v / total_sec * 100.0 for v in p_vals]

            bars = ax_p.bar([f"Phase {k}" for k in p_indices], p_pcts, color="#1f77b4", alpha=0.85, edgecolor="black")
            ax_p.bar_label(bars, padding=3, fmt="%.1f%%", fontsize=9)
            ax_p.set_ylabel("Green Time Allocation (%)", fontsize=10, fontweight="bold")
            ax_p.set_title(f"Green Time Allocation — {r['algorithm']} on {scen.upper()}", fontsize=11, fontweight="bold")
            ax_p.grid(True, axis="y")
            fig_p.tight_layout()
            p10 = os.path.join(output_dir, f"{scen}_{r['algorithm'].lower().replace('-', '_').replace(' ', '_')}_phase_allocation.png")
            fig_p.savefig(p10)
            plt.close(fig_p)
            generated.append(p10)

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
                ax.set_ylabel("Episode Cumulative Reward (- Queue)", fontsize=11)
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


def generate_all_phase6_charts(output_dir: str = "results/figures") -> List[str]:
    """Generates all 15 required charts for Phase 6."""
    all_charts = []
    all_charts.extend(plot_group_a_comparisons(output_dir=output_dir))
    all_charts.extend(plot_group_b_dynamics(output_dir=output_dir))
    all_charts.extend(plot_group_c_rl(output_dir=output_dir))
    print(f"\n[INFO] Generated {len(all_charts)} publication-quality charts in '{output_dir}'.")
    return all_charts


if __name__ == "__main__":
    generate_all_phase6_charts()
