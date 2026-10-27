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
    epsilon_decay: float = 0.85,
    decision_interval: float = 5.0,
    min_green: float = 10.0,
    max_green: float = 60.0,
    yellow_time: float = 3.0,
    reward_config: Optional[Dict[str, float]] = None,
    verbose: bool = True,
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
        lr=lr,
        gamma=gamma,
        epsilon=epsilon_start,
        tls_id=tls_id,
    )

    if save_path is None:
        save_path = os.path.join("models", "q_learning", f"q_learning_{scenario_name}.json")

    r_cfg = reward_config or {
        "scale": 1.0,
        "w_completed": 2.0,
        "w_queue": 0.05,
        "w_waiting": 0.02,
        "w_time_loss": 0.01,
        "w_switch": 0.5,
    }
    reward_fn = MetricsAlignedReward(**r_cfg)

    episode_stats: List[Dict[str, Any]] = []
    best_reward = -float("inf")

    for ep in range(episodes):
        epsilon = max(epsilon_min, epsilon_start * (epsilon_decay ** ep))
        controller.epsilon = epsilon
        controller.explore_mode = True
        controller.reset()
        reward_fn.reset()

        runner = SimulationRunner(
            cfg_path=info.cfg_path,
            controller=controller,
            tls_id=tls_id,
        )

        ep_reward = 0.0
        total_completed = 0
        queue_samples: List[int] = []

        try:
            runner.start(gui=False)
            curr_state = controller.get_state()
            last_action = controller.current_phase

            for step_idx in range(max_steps_per_episode):
                # 1. Query lane conditions
                in_queues: Dict[str, float] = {}
                step_halted = 0
                for lane_id in runner.incoming_lanes:
                    try:
                        q = float(traci.lane.getLastStepHaltingNumber(lane_id))
                        in_queues[lane_id] = q
                        step_halted += int(q)
                    except Exception:
                        pass
                queue_samples.append(step_halted)

                completed = 0
                try:
                    completed = int(traci.simulation.getArrivedNumber())
                    total_completed += completed
                except Exception:
                    pass

                # 2. Advance controller
                state_str, changed = controller.step(runner.step_duration)
                traci.trafficlight.setRedYellowGreenState(tls_id, state_str)

                # 3. Record step for reward
                reward_fn.record_step(
                    in_lane_queues=in_queues,
                    waiting_time=float(step_halted),
                    completed_vehicles=completed,
                    time_loss=float(step_halted),
                    is_switch=changed,
                )

                # 4. Advance TraCI
                traci.simulationStep()

                # 5. Bellman Q-update on decision intervals or phase switches
                if changed or (step_idx > 0 and step_idx % int(decision_interval) == 0):
                    r_val = reward_fn.calculate_reward()
                    ep_reward += r_val
                    next_state = controller.get_state()
                    controller.update_q(curr_state, last_action, r_val, next_state)
                    reward_fn.reset_interval()
                    curr_state = next_state
                    last_action = controller.current_phase

                # Early break if no more expected vehicles
                if step_idx > 300 and traci.simulation.getMinExpectedNumber() <= 0:
                    break

        finally:
            runner.close()

        mean_q = float(sum(queue_samples) / max(len(queue_samples), 1))
        stat = {
            "episode": ep + 1,
            "epsilon": round(epsilon, 4),
            "reward": round(ep_reward, 2),
            "mean_queue": round(mean_q, 2),
            "throughput": total_completed,
            "q_table_size": len(controller.q_table),
        }
        episode_stats.append(stat)

        if verbose:
            print(
                f"[{scenario_name}] Ep {ep+1:02d}/{episodes:02d} | "
                f"eps: {epsilon:.3f} | rew: {ep_reward:8.1f} | "
                f"mean_q: {mean_q:5.2f} | tp: {total_completed:3d} | "
                f"states: {len(controller.q_table)}"
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
    parser.add_argument("--save-dir", type=str, default="models/q_learning", help="Directory to save models")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    out_path = os.path.join(args.save_dir, f"q_learning_{args.scenario}.json")

    train_q_learning_agent(
        scenario_name=args.scenario,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        save_path=out_path,
        lr=args.lr,
        gamma=args.gamma,
    )


if __name__ == "__main__":
    main()
