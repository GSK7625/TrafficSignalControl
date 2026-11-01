"""Script vẽ biểu đồ trực quan hóa kết quả Benchmark 4 thuật toán
(Fixed-Time, Max-Pressure, Tabular Q-Learning, DQN) và đường cong học tập.
Xuất ảnh vào thư mục results/plots/.
"""

import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    out_dir = os.path.join("results", "plots")
    os.makedirs(out_dir, exist_ok=True)

    scenarios = ["cologne1", "ingolstadt1"]
    algos = [
        ("Fixed-Time", "results/fixed_time/{scen}_ft_summary.json", "#4A90E2"),
        ("Max-Pressure", "results/max_pressure/{scen}_mp_summary.json", "#50E3C2"),
        ("Tabular Q-Learning", "results/q_learning/{scen}_ql_summary.json", "#F5A623"),
        ("Deep Q-Network (DQN)", "results/dqn/{scen}_dqn_summary.json", "#9013FE"),
    ]

    # --- 1. BIỂU ĐỒ SO SÁNH 4 METRICS TRÊN 2 BẢN ĐỒ ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Benchmark So Sánh 4 Thuật Toán Điều Khiển Đèn Tín Hiệu (5 Test Seeds)", fontsize=16, fontweight="bold")

    metrics_to_plot = [
        ("average_waiting_time", "Average Waiting Time (s)", axes[0, 0]),
        ("average_travel_time", "Average Travel Time (s)", axes[0, 1]),
        ("mean_queue_length", "Mean Queue Length (xe)", axes[1, 0]),
        ("throughput", "Throughput (xe hoàn thành)", axes[1, 1]),
    ]

    bar_width = 0.18
    x_indices = np.arange(len(scenarios))

    for m_idx, (m_key, m_label, ax) in enumerate(metrics_to_plot):
        for a_idx, (algo_name, path_fmt, color) in enumerate(algos):
            means = []
            stds = []
            for scen in scenarios:
                p = path_fmt.format(scen=scen)
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                means.append(data["metrics"][m_key]["mean"])
                stds.append(data["metrics"][m_key]["stdev"] or 0.0)

            x_pos = x_indices + (a_idx - 1.5) * bar_width
            bars = ax.bar(x_pos, means, yerr=stds, width=bar_width, label=algo_name, color=color, capsize=4, alpha=0.9, edgecolor="black", linewidth=0.8)
            
            # Thêm nhãn số trên đầu cột
            for bar, mean_val in zip(bars, means):
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2.0, h + (max(means) * 0.02), f"{mean_val:.1f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_title(m_label, fontsize=12, fontweight="bold")
        ax.set_xticks(x_indices)
        ax.set_xticklabels([s.upper() for s in scenarios], fontsize=11, fontweight="bold")
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        if m_idx == 0:
            ax.legend(loc="upper right", framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    comp_path = os.path.join(out_dir, "benchmark_comparison.png")
    plt.savefig(comp_path, dpi=200)
    plt.close()
    print(f"--> Đã lưu biểu đồ so sánh: {comp_path}")

    # --- 2. BIỂU ĐỒ ĐƯỜNG CONG HỌC TẬP (TRAINING CURVES) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Đường Cong Học Tập: Tabular Q-Learning vs Deep Q-Network (30 Episodes)", fontsize=16, fontweight="bold")

    for s_idx, scen in enumerate(scenarios):
        # Đọc log huấn luyện
        ql_log_path = f"results/q_learning/{scen}_training_log.json"
        dqn_log_path = f"results/dqn/{scen}_training_log.json"

        with open(ql_log_path, "r", encoding="utf-8") as f:
            ql_raw = json.load(f)
        with open(dqn_log_path, "r", encoding="utf-8") as f:
            dqn_raw = json.load(f)

        ql_items = ql_raw["logs"]
        dqn_items = dqn_raw["logs"]

        episodes = [item["episode"] for item in ql_items]
        ql_train_wait = [item["train_waiting_time"] for item in ql_items]
        dqn_train_wait = [item["train_waiting_time"] for item in dqn_items]

        ql_val_episodes = [item["episode"] for item in ql_items if item.get("validation")]
        ql_val_wait = [item["validation"]["val_waiting_time"] for item in ql_items if item.get("validation")]

        dqn_val_episodes = [item["episode"] for item in dqn_items if item.get("validation")]
        dqn_val_wait = [item["validation"]["val_waiting_time"] for item in dqn_items if item.get("validation")]

        # Cột 1: Train Waiting Time
        ax_rew = axes[s_idx, 0]
        ax_rew.plot(episodes, ql_train_wait, label="Tabular Q-Learning", color="#F5A623", marker="o", markersize=3, linewidth=1.2)
        ax_rew.plot(episodes, dqn_train_wait, label="Deep Q-Network", color="#9013FE", marker="s", markersize=3, linewidth=1.2)
        ax_rew.set_title(f"{scen.upper()}: Training Waiting Time (s)", fontsize=11, fontweight="bold")
        ax_rew.set_xlabel("Episode")
        ax_rew.set_ylabel("Waiting Time (s)")
        ax_rew.grid(True, linestyle="--", alpha=0.5)
        ax_rew.legend(loc="upper right")

        # Cột 2: Validation Waiting Time (hạt giống cố định val_seed=123)
        ax_val = axes[s_idx, 1]
        ax_val.plot(ql_val_episodes, ql_val_wait, label="Tabular Q-Learning", color="#F5A623", marker="o", markersize=5, linewidth=1.5)
        ax_val.plot(dqn_val_episodes, dqn_val_wait, label="Deep Q-Network", color="#9013FE", marker="s", markersize=5, linewidth=1.5)
        ax_val.set_title(f"{scen.upper()}: Validation Waiting Time (s)", fontsize=11, fontweight="bold")
        ax_val.set_xlabel("Episode")
        ax_val.set_ylabel("Validation Waiting Time (s)")
        ax_val.grid(True, linestyle="--", alpha=0.5)
        ax_val.legend(loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    train_path = os.path.join(out_dir, "training_curves.png")
    plt.savefig(train_path, dpi=200)
    plt.close()
    print(f"--> Đã lưu biểu đồ học tập: {train_path}")


if __name__ == "__main__":
    main()
