"""Generates publication-quality charts for RL training progress and 3-algorithm benchmark comparisons."""

import json
import os
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


def set_plot_style():
    """Sets consistent styling for all benchmark figures."""
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#eeeeee"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


def plot_training_curves(
    log_dir: str = "results/training",
    figures_dir: str = "results/figures",
):
    """Plots training reward and epsilon curves for Q-Learning."""
    os.makedirs(figures_dir, exist_ok=True)
    set_plot_style()

    scenarios = ["ingolstadt1", "cologne1"]

    for scen in scenarios:
        ql_file = os.path.join(log_dir, f"q_learning_{scen}_log.json")
        ql_data = {}

        if os.path.isfile(ql_file):
            with open(ql_file, "r", encoding="utf-8") as f:
                ql_data = json.load(f)

        if not ql_data:
            continue

        # 1. Training Reward
        fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
        if "history" in ql_data:
            eps = [h["episode"] for h in ql_data["history"]]
            rews = [h["reward"] for h in ql_data["history"]]
            ax.plot(eps, rews, marker="o", color="#2ca02c", linewidth=2.0, label="Q-Learning")

        ax.set_title(f"Episode Reward Progression — {scen.upper()}", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Training Episode", fontsize=11)
        ax.set_ylabel("Episode Reward (- Average Queue)", fontsize=11)
        ax.grid(True)
        ax.legend(frameon=True, facecolor="white", edgecolor="#cccccc")
        plt.tight_layout()
        plt.savefig(os.path.join(figures_dir, f"reward_progression_{scen}.png"))
        plt.close(fig)

        # 2. Epsilon Exploration Schedule
        if "history" in ql_data:
            eps = [h["episode"] for h in ql_data["history"]]
            epsilons = [h.get("epsilon", 0.0) for h in ql_data["history"]]
            fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
            ax.plot(eps, epsilons, marker=".", color="#9467bd", linewidth=2.0, label="Epsilon Decay")
            ax.set_title(f"Epsilon Exploration Schedule — {scen.upper()}", fontsize=12, fontweight="bold", pad=10)
            ax.set_xlabel("Training Episode", fontsize=10)
            ax.set_ylabel("Exploration Probability (epsilon)", fontsize=10)
            ax.grid(True)
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f"epsilon_schedule_{scen}.png"))
            plt.close(fig)


def plot_comparison_metrics(
    eval_json: str = "results/evaluation/benchmark_comparison.json",
    figures_dir: str = "results/figures",
):
    """Plots comparative bar charts across 4 algorithms for each scenario."""
    if not os.path.isfile(eval_json):
        print(f"Comparison data not found at: {eval_json}")
        return

    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(figures_dir, exist_ok=True)
    set_plot_style()

    algo_order = ["Fixed-Time", "Max-Pressure", "Q-Learning"]
    algo_colors = ["#7f7f7f", "#ff7f0e", "#2ca02c"]

    metrics_config = [
        ("mean_waiting_time", "Average Waiting Time (seconds)", "waiting_time_comparison.png"),
        ("mean_queue_length", "Average Queue Length (vehicles)", "queue_length_comparison.png"),
        ("total_completed_vehicles", "Throughput (completed vehicles)", "throughput_comparison.png"),
        ("total_time_loss", "Total Time Loss (seconds)", "time_loss_comparison.png"),
    ]

    for scen, algos_dict in data.items():
        present_algos = [a for a in algo_order if a in algos_dict]
        colors = [algo_colors[algo_order.index(a)] for a in present_algos]

        for metric_key, ylabel, fname in metrics_config:
            vals = [algos_dict[a].get(metric_key, 0.0) for a in present_algos]

            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
            bars = ax.bar(present_algos, vals, color=colors, width=0.55, edgecolor="#333333", linewidth=0.8)

            # Add value labels on top of bars
            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:.1f}" if isinstance(h, float) else f"{h}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.set_title(f"{ylabel} — {scen.upper()}", fontsize=13, fontweight="bold", pad=12)
            ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(axis="y")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f"{scen}_{fname}"))
            plt.close(fig)

        # Plot Improvement % over Fixed-Time
        if "Fixed-Time" in algos_dict:
            ft_wait = algos_dict["Fixed-Time"].get("mean_waiting_time", 1.0)
            comp_algos = [a for a in ["Max-Pressure", "Q-Learning"] if a in algos_dict]
            impr_vals = [
                ((ft_wait - algos_dict[a].get("mean_waiting_time", ft_wait)) / max(1e-6, ft_wait)) * 100.0
                for a in comp_algos
            ]

            fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
            impr_colors = ["#ff7f0e", "#2ca02c"][:len(comp_algos)]
            bars = ax.bar(comp_algos, impr_vals, color=impr_colors, width=0.5, edgecolor="#333333")

            for bar in bars:
                h = bar.get_height()
                ax.annotate(
                    f"{h:+.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, max(0, h)),
                    xytext=(0, 4),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

            ax.set_title(f"Waiting Time Improvement vs Fixed-Time — {scen.upper()}", fontsize=12, fontweight="bold", pad=12)
            ax.set_ylabel("Improvement (%)", fontsize=11)
            ax.axhline(0, color="#666666", linewidth=0.8)
            ax.grid(axis="y")
            plt.tight_layout()
            plt.savefig(os.path.join(figures_dir, f"{scen}_improvement_vs_fixed_time.png"))
            plt.close(fig)

    print(f"All figures generated successfully in: {figures_dir}")


def generate_all_rl_figures():
    plot_training_curves()
    plot_comparison_metrics()


if __name__ == "__main__":
    generate_all_rl_figures()
