"""Plotting and visualization module for traffic signal control experiments."""

import os
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def set_plot_style():
    """Sets a clean, professional aesthetic style for generated plots."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
    plt.rcParams["axes.edgecolor"] = "#cccccc"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["grid.color"] = "#e0e0e0"
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["grid.alpha"] = 0.7


def generate_benchmark_plots(
    results: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: str = "results/experiments/charts",
) -> List[str]:
    """Generates comparative visualization charts for benchmark results.

    Args:
        results: Nested dict: {scenario: {algo_name: summary_dict}}
        output_dir: Directory where charts will be saved.

    Returns:
        List of generated image file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    set_plot_style()

    scenarios = list(results.keys())
    if not scenarios:
        return []

    # Identify algorithms from the first scenario
    first_scen = scenarios[0]
    algos = list(results[first_scen].keys())

    colors = {
        "Original SUMO": "#7f7f7f",      # Neutral gray for baseline
        "Fixed-Time": "#1f77b4",          # Classic blue
        "Max-Pressure": "#2ca02c",        # Energetic green
    }
    palette = [colors.get(a, "#ff7f0e") for a in algos]

    generated_files = []

    # 1. Bar Chart: Mean Waiting Time
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    x = np.arange(len(scenarios))
    width = 0.25

    for i, algo in enumerate(algos):
        vals = [results[s].get(algo, {}).get("mean_waiting_time", 0.0) for s in scenarios]
        rects = ax.bar(x + i * width, vals, width, label=algo, color=palette[i], alpha=0.9, edgecolor="black", linewidth=0.5)
        ax.bar_label(rects, padding=3, fmt="%.1fs", fontsize=9)

    ax.set_ylabel("Average Waiting Time (s)", fontsize=11, fontweight="bold")
    ax.set_title("Average Waiting Time by Traffic Control Algorithm", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x + width * (len(algos) - 1) / 2)
    ax.set_xticklabels([s.upper() for s in scenarios], fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#ddd")
    fig.tight_layout()
    p1 = os.path.join(output_dir, "waiting_time_comparison.png")
    fig.savefig(p1)
    plt.close(fig)
    generated_files.append(p1)

    # 2. Bar Chart: Mean Queue Length
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for i, algo in enumerate(algos):
        vals = [results[s].get(algo, {}).get("mean_queue_length", 0.0) for s in scenarios]
        rects = ax.bar(x + i * width, vals, width, label=algo, color=palette[i], alpha=0.9, edgecolor="black", linewidth=0.5)
        ax.bar_label(rects, padding=3, fmt="%.2f", fontsize=9)

    ax.set_ylabel("Average Queue Length (vehicles)", fontsize=11, fontweight="bold")
    ax.set_title("Average Queue Length Across Intersections", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x + width * (len(algos) - 1) / 2)
    ax.set_xticklabels([s.upper() for s in scenarios], fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#ddd")
    fig.tight_layout()
    p2 = os.path.join(output_dir, "queue_length_comparison.png")
    fig.savefig(p2)
    plt.close(fig)
    generated_files.append(p2)

    # 3. Bar Chart: Throughput (Completed Vehicles)
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for i, algo in enumerate(algos):
        vals = [results[s].get(algo, {}).get("total_completed_vehicles", 0) for s in scenarios]
        rects = ax.bar(x + i * width, vals, width, label=algo, color=palette[i], alpha=0.9, edgecolor="black", linewidth=0.5)
        ax.bar_label(rects, padding=3, fmt="%d", fontsize=9)

    ax.set_ylabel("Completed Vehicles (Throughput)", fontsize=11, fontweight="bold")
    ax.set_title("Vehicle Throughput Completed During Simulation", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x + width * (len(algos) - 1) / 2)
    ax.set_xticklabels([s.upper() for s in scenarios], fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#ddd")
    fig.tight_layout()
    p3 = os.path.join(output_dir, "throughput_comparison.png")
    fig.savefig(p3)
    plt.close(fig)
    generated_files.append(p3)

    # 4. Bar Chart: Total Time Loss
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    for i, algo in enumerate(algos):
        vals = [results[s].get(algo, {}).get("total_time_loss", 0.0) for s in scenarios]
        rects = ax.bar(x + i * width, vals, width, label=algo, color=palette[i], alpha=0.9, edgecolor="black", linewidth=0.5)
        ax.bar_label(rects, padding=3, fmt="%.0fs", fontsize=9)

    ax.set_ylabel("Total Network Time Loss (s)", fontsize=11, fontweight="bold")
    ax.set_title("Total Delay & Time Loss Induced by Traffic Signals", fontsize=13, fontweight="bold", pad=12)
    ax.set_xticks(x + width * (len(algos) - 1) / 2)
    ax.set_xticklabels([s.upper() for s in scenarios], fontsize=11, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", edgecolor="#ddd")
    fig.tight_layout()
    p4 = os.path.join(output_dir, "time_loss_comparison.png")
    fig.savefig(p4)
    plt.close(fig)
    generated_files.append(p4)

    # 5. Combined 2x2 Dashboard
    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # Top-Left: Waiting Time
    for i, algo in enumerate(algos):
        vals = [results[s].get(algo, {}).get("mean_waiting_time", 0.0) for s in scenarios]
        r = axs[0, 0].bar(x + i * width, vals, width, label=algo, color=palette[i], alpha=0.9)
        axs[0, 0].bar_label(r, padding=2, fmt="%.1fs", fontsize=8)
    axs[0, 0].set_ylabel("Mean Waiting Time (s)", fontweight="bold")
    axs[0, 0].set_title("(A) Average Vehicle Waiting Time", fontweight="bold")
    axs[0, 0].set_xticks(x + width * (len(algos) - 1) / 2)
    axs[0, 0].set_xticklabels([s.upper() for s in scenarios])
    axs[0, 0].legend(fontsize=9)

    # Top-Right: Queue Length
    for i, algo in enumerate(algos):
        vals = [results[s].get(algo, {}).get("mean_queue_length", 0.0) for s in scenarios]
        r = axs[0, 1].bar(x + i * width, vals, width, label=algo, color=palette[i], alpha=0.9)
        axs[0, 1].bar_label(r, padding=2, fmt="%.2f", fontsize=8)
    axs[0, 1].set_ylabel("Mean Queue Length (vehs)", fontweight="bold")
    axs[0, 1].set_title("(B) Average Intersection Queue Length", fontweight="bold")
    axs[0, 1].set_xticks(x + width * (len(algos) - 1) / 2)
    axs[0, 1].set_xticklabels([s.upper() for s in scenarios])

    # Bottom-Left: Throughput
    for i, algo in enumerate(algos):
        vals = [results[s].get(algo, {}).get("total_completed_vehicles", 0) for s in scenarios]
        r = axs[1, 0].bar(x + i * width, vals, width, label=algo, color=palette[i], alpha=0.9)
        axs[1, 0].bar_label(r, padding=2, fmt="%d", fontsize=8)
    axs[1, 0].set_ylabel("Completed Vehicles", fontweight="bold")
    axs[1, 0].set_title("(C) Vehicle Throughput (Higher is better)", fontweight="bold")
    axs[1, 0].set_xticks(x + width * (len(algos) - 1) / 2)
    axs[1, 0].set_xticklabels([s.upper() for s in scenarios])

    # Bottom-Right: Time Loss
    for i, algo in enumerate(algos):
        vals = [results[s].get(algo, {}).get("total_time_loss", 0.0) for s in scenarios]
        r = axs[1, 1].bar(x + i * width, vals, width, label=algo, color=palette[i], alpha=0.9)
        axs[1, 1].bar_label(r, padding=2, fmt="%.0fs", fontsize=8)
    axs[1, 1].set_ylabel("Total Time Loss (s)", fontweight="bold")
    axs[1, 1].set_title("(D) Cumulative Delay / Time Loss", fontweight="bold")
    axs[1, 1].set_xticks(x + width * (len(algos) - 1) / 2)
    axs[1, 1].set_xticklabels([s.upper() for s in scenarios])

    fig.suptitle("Benchmark Comparison: Fixed-Time vs Max-Pressure vs Original SUMO", fontsize=15, fontweight="bold", y=0.99)
    fig.tight_layout()
    p5 = os.path.join(output_dir, "combined_metrics_dashboard.png")
    fig.savefig(p5)
    plt.close(fig)
    generated_files.append(p5)

    return generated_files
