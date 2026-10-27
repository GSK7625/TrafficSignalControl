"""Quy trình huấn luyện cho bộ điều khiển tín hiệu đèn giao thông Tabular Q-Learning.

Huấn luyện tác tử Q-Learning trên các kịch bản SUMO với phần thưởng đa mục tiêu
căn chỉnh trực tiếp theo 5 chỉ số hiệu năng giao thông cốt lõi:
  1. Thời gian chờ trung bình (Average Waiting Time)
  2. Thời gian di chuyển trung bình (Average Travel Time)
  3. Độ dài hàng đợi trung bình & cực đại (Mean & Max Queue Length)
  4. Tổn thất thời gian trung bình (Average Time Loss)
  5. Thông lượng (Throughput)
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

from src.controllers.q_learning import QLearningController
from src.data_inspector import inspect_scenario
from src.rl.rewards import MetricsAlignedReward
from src.simulation_runner import SimulationRunner


def train_q_learning_agent(
    scenario_name: str,
    data_dir: str = "data",
    episodes: int = 15,
    max_steps_per_episode: int = 3600,
    save_path: Optional[str] = None,
    lr: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: Optional[float] = None,
    decision_interval: float = 5.0,
    min_green: float = 20.0,
    max_green: float = 45.0,
    yellow_time: float = 3.0,
    all_red_time: float = 1.0,
    max_red: float = 90.0,
    reward_config: Optional[Dict[str, float]] = None,
    verbose: bool = True,
    use_libsumo: bool = True,
    init_model: Optional[str] = None,
) -> Tuple[QLearningController, List[Dict[str, Any]]]:
    """Huấn luyện tác tử Q-Learning trên một kịch bản giao thông SUMO.

    Tham số:
        scenario_name: Tên thư mục kịch bản (ví dụ: 'cologne1', 'ingolstadt1').
        data_dir: Thư mục cơ sở chứa các thư mục kịch bản.
        episodes: Số lượng tập (episodes) huấn luyện cần chạy.
        max_steps_per_episode: Thời gian mô phỏng tối đa tính bằng giây cho mỗi episode.
        save_path: Đường dẫn file JSON đầu ra lưu bảng Q-table đã học.
        lr: Tốc độ học (learning rate alpha).
        gamma: Hệ số chiết khấu tương lai (discount factor gamma).
        epsilon_start: Tỷ lệ khám phá (exploration rate) ban đầu.
        epsilon_min: Mức sàn tỷ lệ khám phá tối thiểu.
        epsilon_decay: Tốc độ suy giảm epsilon theo từng episode (suy giảm hình học).
        decision_interval: Khoảng thời gian (giây) giữa các lần ra quyết định của RL.
        min_green: Thời gian xanh tối thiểu tính bằng giây.
        max_green: Thời gian xanh tối đa trước khi ép buộc chuyển pha.
        yellow_time: Thời gian đèn vàng giải phóng nút giao.
        reward_config: Các tham số cấu hình tùy chọn cho MetricsAlignedReward.
        verbose: Nếu True, in tiến trình chi tiết qua từng tập.
        use_libsumo: Sử dụng backend libsumo để tăng tốc thực thi.
        init_model: Đường dẫn file JSON Q-table có sẵn để huấn luyện tiếp (fine-tune).

    Trả về:
        Tuple gồm (QLearningController đã huấn luyện, danh sách thống kê qua các episode).
    """
    info = inspect_scenario(scenario_name, data_dir=data_dir)
    tls_id = info.tls_ids[0]
    green_phases = info.get_green_phases(tls_id)
    movements = info.get_movements(tls_id)

    controller = QLearningController(
        green_phases=green_phases,
        movements=movements,
        decision_interval=decision_interval,
        min_green=min_green,
        max_green=max_green,
        yellow_time=yellow_time,
        all_red_time=all_red_time,
        max_red=max_red,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon_start,
        tls_id=tls_id,
    )

    if init_model:
        if os.path.isfile(init_model):
            controller.load(init_model)
            # Áp dụng lại các tham số ghi đè từ caller
            controller.lr = lr
            controller.gamma = gamma
            controller.min_green = min_green
            controller.max_green = max_green
            if verbose:
                print(
                    f"[{scenario_name}] Pre-loaded model from: {init_model} "
                    f"({len(controller.q_table)} states, {len(controller.action_counts)} action entries)"
                )
        else:
            raise FileNotFoundError(f"init_model file not found: {init_model}")

    if save_path is None:
        save_path = os.path.join("models", "q_learning", f"q_learning_{scenario_name}.json")

    r_cfg = reward_config or {
        "scale": 1.0,
        "w_completed": 2.0,
        "w_queue": 0.05,
        "w_waiting": 0.02,
        "w_time_loss": 0.01,
        "w_switch": 2.0,
    }
    reward_fn = MetricsAlignedReward(**r_cfg)

    if epsilon_decay is None:
        if episodes <= 15:
            eff_decay = 0.85
        else:
            eff_decay = (epsilon_min / max(epsilon_start, 1e-6)) ** (1.0 / max(int(episodes * 0.75), 1))
    else:
        eff_decay = float(epsilon_decay)

    episode_stats: List[Dict[str, Any]] = []
    best_reward = -float("inf")

    for ep in range(episodes):
        epsilon = max(epsilon_min, epsilon_start * (eff_decay ** ep))
        controller.epsilon = epsilon
        controller.explore_mode = True
        controller.reset()
        reward_fn.reset()

        runner = SimulationRunner(
            cfg_path=info.cfg_path,
            controller=controller,
            tls_id=tls_id,
            use_libsumo=use_libsumo,
        )

        ep_reward = 0.0
        total_completed = 0
        queue_samples: List[int] = []

        try:
            runner.start(gui=False)
            conn = runner.conn
            last_decision_state: Optional[Tuple[int, ...]] = None
            last_decision_action: Optional[int] = None

            for step_idx in range(max_steps_per_episode):
                # 1. Truy vấn điều kiện trên các làn đi vào
                in_queues: Dict[str, float] = {}
                step_halted = 0
                for lane_id in runner.incoming_lanes:
                    try:
                        q = float(conn.lane.getLastStepHaltingNumber(lane_id))
                        in_queues[lane_id] = q
                        step_halted += int(q)
                    except Exception:
                        pass
                queue_samples.append(step_halted)

                completed = 0
                try:
                    completed = int(conn.simulation.getArrivedNumber())
                    total_completed += completed
                except Exception:
                    pass

                # 2. Tiến bộ điều khiển một bước
                state_str, changed = controller.step(runner.step_duration)
                conn.trafficlight.setRedYellowGreenState(tls_id, state_str)

                # 3. Ghi nhận bước đo lường cho hàm phần thưởng
                reward_fn.record_step(
                    in_lane_queues=in_queues,
                    waiting_time=float(step_halted),
                    completed_vehicles=completed,
                    time_loss=float(step_halted),
                    is_switch=changed,
                )

                # 4. Tiến một bước mô phỏng
                conn.simulationStep()

                # 5. Cập nhật Q-Learning Bellman tại các điểm quyết định rõ ràng
                if controller.last_decision is not None:
                    dec_state, dec_action = controller.last_decision
                    if last_decision_state is not None and last_decision_action is not None:
                        r_val = reward_fn.calculate_reward()
                        ep_reward += r_val
                        controller.update_q(last_decision_state, last_decision_action, r_val, dec_state)
                        reward_fn.reset_interval()
                    last_decision_state = dec_state
                    last_decision_action = dec_action

                # Dừng sớm nếu không còn phương tiện dự kiến
                if step_idx > 300 and conn.simulation.getMinExpectedNumber() <= 0:
                    break

            # Cập nhật trạng thái kết thúc (Terminal update) tại cuối episode
            if last_decision_state is not None and last_decision_action is not None:
                r_val = reward_fn.calculate_reward()
                ep_reward += r_val
                final_state = controller.get_state()
                controller.update_q(last_decision_state, last_decision_action, r_val, final_state)
                reward_fn.reset_interval()

        finally:
            runner.close()

        mean_q = float(sum(queue_samples) / max(len(queue_samples), 1))
        explored_actions = len(controller.action_counts)
        stat = {
            "episode": ep + 1,
            "epsilon": round(epsilon, 4),
            "reward": round(ep_reward, 2),
            "mean_queue": round(mean_q, 2),
            "throughput": total_completed,
            "q_table_size": len(controller.q_table),
            "action_count": explored_actions,
        }
        episode_stats.append(stat)

        if verbose:
            print(
                f"[{scenario_name}] Ep {ep+1:02d}/{episodes:02d} | "
                f"eps: {epsilon:.3f} | rew: {ep_reward:8.1f} | "
                f"mean_q: {mean_q:5.2f} | tp: {total_completed:3d} | "
                f"states: {len(controller.q_table)} | act: {explored_actions}"
            )

        if ep_reward > best_reward:
            best_reward = ep_reward
            best_path = save_path.replace(".json", "_best.json")
            controller.save(best_path)

    # Lưu mô hình đã huấn luyện hoàn tất
    controller.save(save_path)
    if verbose:
        print(f"[{scenario_name}] Training complete. Model saved to: {save_path}")

    return controller, episode_stats


def main() -> None:
    """Điểm vào CLI để huấn luyện các tác tử Q-Learning."""
    parser = argparse.ArgumentParser(description="Train Q-Learning Traffic Signal Controller")
    parser.add_argument("--scenario", type=str, default="cologne1", help="Scenario name (e.g. cologne1, ingolstadt1)")
    parser.add_argument("--episodes", type=int, default=15, help="Number of training episodes")
    parser.add_argument("--max-steps", type=int, default=3600, help="Max simulation steps per episode")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate alpha")
    parser.add_argument("--gamma", type=float, default=0.95, help="Discount factor gamma")
    parser.add_argument("--epsilon-start", type=float, default=1.0, help="Initial exploration rate epsilon (default: 1.0)")
    parser.add_argument("--epsilon-min", type=float, default=0.05, help="Minimum exploration rate floor (default: 0.05)")
    parser.add_argument("--epsilon-decay", type=float, default=None, help="Epsilon decay rate per episode (geometric)")
    parser.add_argument("--init-model", type=str, default=None, help="Path to existing Q-table JSON model to resume or fine-tune")
    parser.add_argument("--save-dir", type=str, default="models/q_learning", help="Directory to save models")
    parser.add_argument("--min-green", type=float, default=20.0, help="Minimum green phase duration in seconds (default: 20.0)")
    parser.add_argument("--use-libsumo", dest="use_libsumo", action="store_true", default=True, help="Use libsumo instead of traci for faster execution (default: True)")
    parser.add_argument("--no-libsumo", dest="use_libsumo", action="store_false", help="Disable libsumo and force TraCI")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"q_learning_{args.scenario}.json")

    train_q_learning_agent(
        scenario_name=args.scenario,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        save_path=out_path,
        min_green=args.min_green,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        init_model=args.init_model,
        use_libsumo=args.use_libsumo,
    )


if __name__ == "__main__":
    main()
