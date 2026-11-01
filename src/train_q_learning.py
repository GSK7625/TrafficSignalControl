"""Quy trình huấn luyện và chọn lọc mô hình Tabular Q-Learning.

Huấn luyện model riêng cho từng bản đồ theo ngân sách 30 episodes,
luân phiên tập Train Seeds, đánh giá Validation định kỳ và lưu checkpoint tốt nhất.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

from src.controllers.q_learning import QLearningController
from src.data_inspector import read_scenario
from src.simulation_runner import run_simulation

TRAIN_SEEDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
VAL_SEEDS = [11, 22, 33]


def evaluate_on_validation(
    scenario_name: str,
    controller: QLearningController,
    val_seeds: List[int] = VAL_SEEDS,
) -> Dict[str, float]:
    """Chạy đánh giá mô hình trên tập hạt giống Validation (tắt khám phá, không cập nhật Q)."""
    # Tạm thời tắt chế độ học và đặt epsilon = 0.0
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
            controller.reset()
            # Chạy đánh giá không ghi đè thư mục chính
            val_out_dir = os.path.join("results", "temp_val", f"{scenario_name}_seed{seed}")
            summary = run_simulation(
                scenario_name=scenario_name,
                controller=controller,
                seed=seed,
                max_steps=None,  # Đầy đủ 3600s
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
        # Khôi phục trạng thái ban đầu cho controller
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


def train_q_learning(
    scenario_name: str,
    total_episodes: int = 30,
    eval_interval: int = 5,
    alpha: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.90,
    train_seeds: List[int] = TRAIN_SEEDS,
    val_seeds: List[int] = VAL_SEEDS,
    model_dir: str = "models/q_learning",
) -> Dict[str, Any]:
    """Huấn luyện Tabular Q-Learning với đầy đủ cơ chế logging và checkpointing."""
    print(f"\n=======================================================")
    print(f"BẮT ĐẦU HUẤN LUYỆN Q-LEARNING: {scenario_name.upper()}")
    print(f"Tổng số Episodes: {total_episodes} | Eval Interval: {eval_interval}")
    print(f"Alpha: {alpha} | Gamma: {gamma} | Epsilon: {epsilon_start} -> {epsilon_min} (decay={epsilon_decay})")
    print(f"=======================================================")

    scenario = read_scenario(scenario_name)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(os.path.join("results", "q_learning"), exist_ok=True)

    controller = QLearningController(
        green_phases=scenario["green_phases"],
        movements=scenario["movements"],
        decision_interval=5.0,
        min_green=20.0,
        max_green=45.0,
        yellow_time=3.0,
        all_red_time=1.0,
        max_red=90.0,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon_start,
        learning_enabled=True,
        tls_id=scenario["tls_id"],
    )

    training_logs: List[Dict[str, Any]] = []
    best_val_waiting = float("inf")
    best_episode = 0

    best_model_path = os.path.join(model_dir, f"{scenario_name}_best.json")
    final_model_path = os.path.join(model_dir, f"{scenario_name}_final.json")

    for ep in range(1, total_episodes + 1):
        t_ep_start = time.time()
        # Tính epsilon theo lịch suy giảm
        current_eps = max(epsilon_min, round(epsilon_start * (epsilon_decay ** (ep - 1)), 4))
        controller.epsilon = current_eps
        controller.learning_enabled = True

        # Chọn seed luân phiên từ danh sách train seeds
        train_seed = train_seeds[(ep - 1) % len(train_seeds)]
        controller.reset()

        ep_out_dir = os.path.join("results", "temp_train", f"{scenario_name}_ep{ep}")
        summary = run_simulation(
            scenario_name=scenario_name,
            controller=controller,
            seed=train_seed,
            max_steps=None,
            gui=False,
            output_dir=ep_out_dir,
        )

        ep_time = time.time() - t_ep_start
        states_count = len(controller.q_table)
        updates_count = controller.update_count

        log_entry: Dict[str, Any] = {
            "episode": ep,
            "train_seed": train_seed,
            "epsilon": current_eps,
            "training_time_s": round(ep_time, 2),
            "total_updates": updates_count,
            "states_visited": states_count,
            "train_waiting_time": summary["average_waiting_time"],
            "train_travel_time": summary["average_travel_time"],
            "train_queue_length": summary["mean_queue_length"],
            "train_throughput": summary["throughput"],
            "validation": None,
        }

        print(
            f"Ep {ep:02d}/{total_episodes:02d} | Seed: {train_seed:02d} | eps: {current_eps:.3f} | "
            f"States: {states_count:02d} | Updates: {updates_count:04d} | "
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
    log_path = os.path.join("results", "q_learning", f"{scenario_name}_training_log.json")
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

    print(f"\n=> HUẤN LUYỆN HOÀN TẤT!")
    print(f"   - Checkpoint tốt nhất tại Episode {best_episode} (Validation Wait = {best_val_waiting:.2f}s)")
    print(f"   - Tệp model lưu tại: {best_model_path}")
    print(f"   - Tệp log lưu tại: {log_path}")

    return {
        "scenario": scenario_name,
        "best_episode": best_episode,
        "best_val_waiting": best_val_waiting,
        "best_model_path": best_model_path,
        "training_logs": training_logs,
    }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="ingolstadt1", choices=["cologne1", "ingolstadt1"])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--eval-interval", type=int, default=5)
    args = parser.parse_args()

    train_q_learning(scenario_name=args.scenario, total_episodes=args.episodes, eval_interval=args.eval_interval)
