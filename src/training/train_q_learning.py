"""Training pipeline for Tabular Q-Learning Traffic Signal Controller.

Trains a Q-Learning agent on SUMO scenarios with multi-objective rewards aligned
with the 5 core traffic performance metrics:
  1. Average Waiting Time
  2. Average Travel Time
  3. Mean & Max Queue Length
  4. Average Time Loss
  5. Throughput
"""

import argparse
import os
from typing import Any, Dict, List, Optional, Tuple

import traci
from src.controllers.q_learning import CompactStateDiscretizer, QLearningController
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
    """Trains a Q-Learning agent on a SUMO scenario.

    Args:
        scenario_name: Name of scenario directory (e.g. 'cologne1', 'ingolstadt1').
        data_dir: Base directory containing scenario folders.
        episodes: Number of training episodes to run.
        max_steps_per_episode: Simulation duration in seconds per episode.
        save_path: Output JSON file path for learned Q-table.
        lr: Learning rate alpha.
        gamma: Discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_min: Minimum exploration rate floor.
        epsilon_decay: Geometric decay rate per episode.
        decision_interval: Interval in seconds between RL action evaluations.
        min_green: Minimum green phase duration in seconds.
        max_green: Maximum green phase duration before forced phase switch.
        yellow_time: Clearance yellow duration in seconds.
        reward_config: Optional keyword arguments for MetricsAlignedReward.
        verbose: If True, prints per-episode training statistics.

    Returns:
        Tuple of (trained QLearningController, list of episode statistics).
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
            # Re-apply caller overrides
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
                # 1. Query lane conditions
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

                # 2. Advance controller
                state_str, changed = controller.step(runner.step_duration)
                conn.trafficlight.setRedYellowGreenState(tls_id, state_str)

                # 3. Record step for reward
                reward_fn.record_step(
                    in_lane_queues=in_queues,
                    waiting_time=float(step_halted),
                    completed_vehicles=completed,
                    time_loss=float(step_halted),
                    is_switch=changed,
                )

                # 4. Advance simulation step
                conn.simulationStep()

                # 5. Bellman Q-update on explicit decision points
                if controller.last_decision is not None:
                    dec_state, dec_action = controller.last_decision
                    if last_decision_state is not None and last_decision_action is not None:
                        r_val = reward_fn.calculate_reward()
                        ep_reward += r_val
                        controller.update_q(last_decision_state, last_decision_action, r_val, dec_state)
                        reward_fn.reset_interval()
                    last_decision_state = dec_state
                    last_decision_action = dec_action

                # Early break if no more expected vehicles
                if step_idx > 300 and conn.simulation.getMinExpectedNumber() <= 0:
                    break

            # Terminal update at end of episode
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

    # Save final trained model
    controller.save(save_path)
    if verbose:
        print(f"[{scenario_name}] Training complete. Model saved to: {save_path}")

    return controller, episode_stats


def main() -> None:
    """CLI entrypoint for training Q-Learning agents."""
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
