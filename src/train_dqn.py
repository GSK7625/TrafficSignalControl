"""Kịch bản huấn luyện Deep Q-Network (DQN) với Validation Checkpoint.

Tuân thủ nghiêm ngặt Giai đoạn 9 & 10 và quy tắc AGENT.md:
  1. Đặt seed đầy đủ cho Python, NumPy, PyTorch và SUMO để tối đa hóa tính tái lập.
  2. Huấn luyện trong ngân sách đã chốt: đúng 30 episodes (1 giờ mô phỏng SUMO headless/ep).
  3. Đánh giá Validation định kỳ (mỗi 5 episodes) trên seed cố định (Val Seed: 99).
  4. Lựa chọn Checkpoint tốt nhất hoàn toàn dựa trên Validation Average Waiting Time thấp nhất.
  5. Tuyệt đối KHÔNG đụng đến tập Test seeds [42, 101, 2024, 3407, 9999].
"""

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.controllers.dqn import DQNController
from src.data_inspector import read_scenario
from src.simulation_runner import run_simulation

# Tập seed huấn luyện luân phiên (1 -> 20)
TRAIN_SEEDS: List[int] = list(range(1, 21))

# Seed validation cố định (độc lập với train và test)
VAL_SEEDS: List[int] = [99]


def set_random_seed(seed: int) -> None:
    """Đặt hạt giống ngẫu nhiên cho toàn bộ các thư viện để tối đa hóa tính tái lập."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_on_validation(
    scenario_name: str,
    controller: DQNController,
    val_seeds: List[int] = VAL_SEEDS,
) -> Dict[str, float]:
    """Đánh giá hiệu năng controller trên tập Validation seeds ở chế độ Evaluation."""
    original_learning = controller.learning_enabled
    original_eps = controller.epsilon

    controller.learning_enabled = False
    controller.epsilon = 0.0

    val_waitings: List[float] = []
    val_travels: List[float] = []
    val_queues: List[float] = []
    val_throughputs: List[int] = []

    try:
        for seed in val_seeds:
            set_random_seed(seed)
            controller.reset()
            val_out_dir = os.path.join("results", "temp_val", f"{scenario_name}_seed{seed}")
            summary = run_simulation(
                scenario_name=scenario_name,
                controller=controller,
                seed=seed,
                max_steps=None,
                gui=False,
                output_dir=val_out_dir,
            )
            if summary["average_waiting_time"] is not None:
                val_waitings.append(summary["average_waiting_time"])
            if summary["average_travel_time"] is not None:
                val_travels.append(summary["average_travel_time"])
            if summary["mean_queue_length"] is not None:
                val_queues.append(summary["mean_queue_length"])
            val_throughputs.append(summary["throughput"])
    finally:
        controller.learning_enabled = original_learning
        controller.epsilon = original_eps

    avg_wait = sum(val_waitings) / len(val_waitings) if val_waitings else 999.0
    avg_travel = sum(val_travels) / len(val_travels) if val_travels else 999.0
    avg_queue = sum(val_queues) / len(val_queues) if val_queues else 999.0
    avg_tp = sum(val_throughputs) / len(val_throughputs) if val_throughputs else 0.0

    return {
        "val_waiting_time": round(avg_wait, 2),
        "val_travel_time": round(avg_travel, 2),
        "val_queue_length": round(avg_queue, 2),
        "val_throughput": round(avg_tp, 1),
    }


def train_dqn(
    scenario_name: str,
    total_episodes: int = 30,
    eval_interval: int = 5,
    lr: float = 0.0005,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.90,
    buffer_capacity: int = 10000,
    batch_size: int = 32,
    min_replay_size: int = 200,
    target_update_freq: int = 200,
    train_seeds: List[int] = TRAIN_SEEDS,
    val_seeds: List[int] = VAL_SEEDS,
    model_dir: str = "models/dqn",
) -> Dict[str, Any]:
    """Huấn luyện Deep Q-Network với đầy đủ cơ chế logging và checkpointing."""
    print(f"\n=======================================================")
    print(f"BẮT ĐẦU HUẤN LUYỆN DQN: {scenario_name.upper()}")
    print(f"Tổng số Episodes: {total_episodes} | Eval Interval: {eval_interval}")
    print(f"LR: {lr} | Gamma: {gamma} | Batch Size: {batch_size} | Buffer: {buffer_capacity}")
    print(f"Epsilon: {epsilon_start} -> {epsilon_min} (decay={epsilon_decay}) | Target Sync: {target_update_freq} steps")
    print(f"=======================================================")

    scenario = read_scenario(scenario_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join("results", "dqn"), exist_ok=True)

    # Cấu hình kích thước vector state và ngưỡng hàng đợi theo kịch bản
    if scenario_name == "ingolstadt1":
        state_dim = 6  # 3 one-hot + 1 time + 2 queues
        q_max = 30.0
    elif scenario_name == "cologne1":
        state_dim = 11  # 8 one-hot + 1 time + 2 queues
        q_max = 50.0
    else:
        raise ValueError(f"Kịch bản không được hỗ trợ: {scenario_name}")

    controller = DQNController(
        green_phases=scenario["green_phases"],
        movements=scenario["movements"],
        state_dim=state_dim,
        q_max=q_max,
        decision_interval=5.0,
        min_green=20.0,
        max_green=45.0,
        yellow_time=3.0,
        all_red_time=1.0,
        max_red=90.0,
        gamma=gamma,
        lr=lr,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        buffer_capacity=buffer_capacity,
        batch_size=batch_size,
        min_replay_size=min_replay_size,
        target_update_freq=target_update_freq,
        learning_enabled=True,
    )

    training_logs: List[Dict[str, Any]] = []
    best_val_waiting = float("inf")
    best_episode = 1
    best_model_path = os.path.join(model_dir, f"{scenario_name}_best.pt")
    final_model_path = os.path.join(model_dir, f"{scenario_name}_final.pt")

    for ep in range(1, total_episodes + 1):
        t_ep_start = time.time()
        # Tính epsilon theo lịch suy giảm hình học
        current_eps = max(epsilon_min, round(epsilon_start * (epsilon_decay ** (ep - 1)), 4))
        controller.epsilon = current_eps
        controller.learning_enabled = True

        # Chọn seed luân phiên từ danh sách train seeds
        train_seed = train_seeds[(ep - 1) % len(train_seeds)]
        set_random_seed(train_seed)
        controller.reset()

        ep_out_dir = os.path.join("results", "temp_train_dqn", f"{scenario_name}_ep{ep}")
        summary = run_simulation(
            scenario_name=scenario_name,
            controller=controller,
            seed=train_seed,
            max_steps=None,
            gui=False,
            output_dir=ep_out_dir,
        )

        ep_time = time.time() - t_ep_start
        buffer_size = len(controller.replay_buffer)
        updates_count = controller.update_count
        avg_loss = (
            round(sum(controller.recent_losses) / len(controller.recent_losses), 4)
            if controller.recent_losses
            else 0.0
        )

        log_entry: Dict[str, Any] = {
            "episode": ep,
            "train_seed": train_seed,
            "epsilon": current_eps,
            "training_time_s": round(ep_time, 2),
            "total_updates": updates_count,
            "buffer_size": buffer_size,
            "avg_loss": avg_loss,
            "train_waiting_time": summary["average_waiting_time"],
            "train_travel_time": summary["average_travel_time"],
            "train_queue_length": summary["mean_queue_length"],
            "train_throughput": summary["throughput"],
            "validation": None,
        }

        print(
            f"Ep {ep:02d}/{total_episodes:02d} | Seed: {train_seed:02d} | eps: {current_eps:.3f} | "
            f"Buf: {buffer_size:04d} | Updates: {updates_count:04d} | Loss: {avg_loss:.4f} | "
            f"Wait: {summary['average_waiting_time']}s | Q: {summary['mean_queue_length']} | Time: {ep_time:.1f}s",
            end="",
        )

        # Chạy Validation định kỳ (và ở episode đầu tiên làm mốc baseline)
        if ep == 1 or ep % eval_interval == 0 or ep == total_episodes:
            val_metrics = evaluate_on_validation(scenario_name, controller, val_seeds)
            log_entry["validation"] = val_metrics
            val_wait = val_metrics["val_waiting_time"]
            print(f" --> [VAL Wait: {val_wait:.2f}s | Q: {val_metrics['val_queue_length']:.2f}]", end="")

            # Cập nhật checkpoint tốt nhất theo tiêu chí Validation Waiting Time
            if val_wait < best_val_waiting:
                best_val_waiting = val_wait
                best_episode = ep
                controller.save(
                    best_model_path,
                    metadata={
                        "scenario": scenario_name,
                        "best_episode": best_episode,
                        "val_waiting_time": best_val_waiting,
                        "val_metrics": val_metrics,
                    },
                )
                print(f" [★ MỚI BEST!]", end="")

        print()
        training_logs.append(log_entry)

    # Lưu checkpoint cuối cùng
    controller.save(
        final_model_path,
        metadata={
            "scenario": scenario_name,
            "final_episode": total_episodes,
            "best_episode": best_episode,
            "best_val_waiting": best_val_waiting,
        },
    )

    # Lưu toàn bộ training log
    log_path = os.path.join("results", "dqn", f"{scenario_name}_training_log.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scenario": scenario_name,
                "total_episodes": total_episodes,
                "best_episode": best_episode,
                "best_val_waiting": best_val_waiting,
                "best_model_path": best_model_path,
                "final_model_path": final_model_path,
                "logs": training_logs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n=> HUẤN LUYỆN DQN HOÀN TẤT!")
    print(f"   - Checkpoint tốt nhất tại Episode {best_episode} (Validation Wait = {best_val_waiting}s)")
    print(f"   - Tệp model lưu tại: {best_model_path}")
    print(f"   - Tệp log lưu tại: {log_path}\n")

    return {
        "best_episode": best_episode,
        "best_val_waiting": best_val_waiting,
        "best_model_path": best_model_path,
        "final_model_path": final_model_path,
        "log_path": log_path,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Huấn luyện DQN cho điều khiển đèn giao thông.")
    parser.add_argument("--scenario", type=str, required=True, choices=["cologne1", "ingolstadt1"])
    parser.add_argument("--episodes", type=int, default=30, help="Tổng số episode huấn luyện")
    parser.add_argument("--eval-interval", type=int, default=5, help="Chu kỳ chạy validation (episodes)")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate Adam")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor")
    parser.add_argument("--batch-size", type=int, default=32, help="Kích thước mini-batch")
    parser.add_argument("--buffer-size", type=int, default=10000, help="Kích thước Replay Buffer")
    parser.add_argument("--target-update-freq", type=int, default=200, help="Chu kỳ cập nhật Target Network")
    parser.add_argument("--min-replay-size", type=int, default=200, help="Số transition tối thiểu trước khi học")

    args = parser.parse_args()

    train_dqn(
        scenario_name=args.scenario,
        total_episodes=args.episodes,
        eval_interval=args.eval_interval,
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_size,
        target_update_freq=args.target_update_freq,
        min_replay_size=args.min_replay_size,
    )
