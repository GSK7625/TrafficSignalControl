"""Script tự động hóa huấn luyện 100 episodes cho toàn bộ mô hình (Q-Learning và DQN)
trên cả hai bản đồ cologne1 và ingolstadt1, sau đó chạy đánh giá 5 test seeds,
tổng hợp bảng số liệu Mean +- Stdev và vẽ biểu đồ kết quả.
"""

import sys
import time

from src.train_q_learning import train_q_learning
from src.train_dqn import train_dqn
from src.evaluate_q_learning import evaluate_q_learning_scenario
from src.evaluate_dqn import evaluate_dqn_scenario
from src.consolidate_results import main as consolidate_main
from src.plot_benchmark import main as plot_main


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    total_episodes = 100
    t_start_all = time.time()

    print("================================================================================")
    print(f"BẮT ĐẦU CHU KỲ HUẤN LUYỆN VÀ ĐÁNH GIÁ 100 EPISODES TOÀN DIỆN")
    print("================================================================================")

    # 1. Huấn luyện Q-Learning (Ingolstadt1 & Cologne1)
    print("\n[BƯỚC 1/6] Huấn luyện Tabular Q-Learning trên ingolstadt1 (100 episodes)...")
    train_q_learning(scenario_name="ingolstadt1", total_episodes=total_episodes, eval_interval=5)

    print("\n[BƯỚC 2/6] Huấn luyện Tabular Q-Learning trên cologne1 (100 episodes)...")
    train_q_learning(scenario_name="cologne1", total_episodes=total_episodes, eval_interval=5)

    # 2. Huấn luyện DQN (Ingolstadt1 & Cologne1)
    print("\n[BƯỚC 3/6] Huấn luyện DQN trên ingolstadt1 (100 episodes)...")
    train_dqn(scenario_name="ingolstadt1", total_episodes=total_episodes, eval_interval=5)

    print("\n[BƯỚC 4/6] Huấn luyện DQN trên cologne1 (100 episodes)...")
    train_dqn(scenario_name="cologne1", total_episodes=total_episodes, eval_interval=5)

    # 3. Đánh giá 5 test seeds độc lập
    print("\n[BƯỚC 5/6] Đánh giá độc lập trên 5 test seeds...")
    print("--> Đánh giá Tabular Q-Learning:")
    evaluate_q_learning_scenario("cologne1")
    evaluate_q_learning_scenario("ingolstadt1")

    print("--> Đánh giá Deep Q-Network (DQN):")
    evaluate_dqn_scenario("cologne1")
    evaluate_dqn_scenario("ingolstadt1")

    # 4. Tổng hợp số liệu và vẽ biểu đồ
    print("\n[BƯỚC 6/6] Tổng hợp kết quả và cập nhật biểu đồ trực quan hóa...")
    consolidate_main()
    plot_main()

    elapsed = time.time() - t_start_all
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print("\n================================================================================")
    print(f"HOÀN TẤT TOÀN BỘ TIẾN TRÌNH 100 EPISODES! (Tổng thời gian: {mins}m {secs}s)")
    print("================================================================================")


if __name__ == "__main__":
    main()
