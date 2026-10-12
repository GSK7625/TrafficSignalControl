"""Evaluation pipeline for trained Q-Learning and DQN agents."""

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

from src.controllers.dqn import DQNController
from src.controllers.q_learning import QLearningController
from src.rl.environment import TrafficSignalRLEnv


def evaluate_agent(
    scenario_name: str,
    algorithm: str,
    model_path: str,
    seed: int = 42,
    steps: Optional[int] = 3600,
    decision_interval: float = 5.0,
    min_green: float = 5.0,
    output_dir: str = "results/evaluation",
) -> Dict[str, Any]:
    """Runs a rigorous evaluation episode with exploration and learning disabled.

    Args:
        scenario_name: 'cologne1' or 'ingolstadt1'.
        algorithm: 'q_learning' or 'dqn'.
        model_path: Path to model checkpoint file (.json for Q-Learning, .pt for DQN).
        seed: Random seed (matches baseline benchmarks).
        steps: Total steps to evaluate (default: 3600s).
        decision_interval: Decision interval in seconds.
        min_green: Minimum green constraint.
        output_dir: Output directory for evaluation results.

    Returns:
        Evaluation summary metrics dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    algo_key = algorithm.lower().replace("-", "_")

    print(f"\n---> Evaluating [{algo_key.upper()}] on [{scenario_name.upper()}] (Seed: {seed})...")

    env = TrafficSignalRLEnv(
        scenario_name=scenario_name,
        decision_interval=decision_interval,
        min_green=min_green,
        max_steps=steps,
        seed=seed,
        output_dir=os.path.join(output_dir, f"eval_{scenario_name}_{algo_key}"),
        use_tripinfo=True,
    )

    try:
        # Instantiate and load agent in eval_mode
        if algo_key in ("q_learning", "qlearning"):
            agent = QLearningController(
                num_actions=env.num_actions,
                seed=seed,
                eval_mode=True,
            )
            agent.load(model_path)
            agent.eval_mode = True
            agent.epsilon = 0.0  # Zero exploration
        elif algo_key in ("dqn", "deep_q_network"):
            agent = DQNController(
                observation_dim=env.observation_dim,
                action_dim=env.num_actions,
                obs_builder=env.obs_builder,
                seed=seed,
                eval_mode=True,
            )
            agent.load(model_path)
            agent.eval_mode = True
            agent.epsilon = 0.0  # Zero exploration
        else:
            raise ValueError(f"Unsupported RL algorithm: {algorithm}")

        start_time = time.time()
        obs_vec, info = env.reset(seed=seed)
        valid_actions = info["valid_actions"]
        discrete_state = info["discrete_state"]

        total_reward = 0.0
        step_count = 0

        while True:
            if algo_key in ("q_learning", "qlearning"):
                action = agent.select_action_for_state(discrete_state, valid_actions=valid_actions)
            else:
                action = agent.select_action_for_state(obs_vec, valid_actions=valid_actions)

            next_obs, reward, terminated, truncated, next_info = env.step(action)
            total_reward += reward
            step_count += 1

            obs_vec = next_obs
            valid_actions = next_info["valid_actions"]
            discrete_state = next_info["discrete_state"]

            if terminated or truncated:
                break

        wall_clock = time.time() - start_time
        summary = env.get_summary() or {}

        eval_result = {
            "scenario_name": scenario_name,
            "algorithm": "Q-Learning" if algo_key in ("q_learning", "qlearning") else "DQN",
            "model_path": model_path,
            "seed": seed,
            "total_steps": summary.get("total_steps", step_count * int(decision_interval)),
            "total_completed_vehicles": summary.get("total_completed_vehicles", 0),
            "mean_queue_length": summary.get("mean_queue_length", 0.0),
            "max_queue_length": summary.get("max_queue_length", 0),
            "mean_waiting_time": summary.get("mean_waiting_time", 0.0),
            "total_waiting_time": summary.get("total_waiting_time", 0.0),
            "total_queue_delay": summary.get("total_queue_delay", 0.0),
            "mean_travel_time": summary.get("mean_travel_time", 0.0),
            "total_time_loss": summary.get("total_time_loss", 0.0),
            "running_vehicles_at_end": summary.get("running_vehicles_at_end", 0),
            "pending_insertion_vehicles": summary.get("pending_insertion_vehicles", 0),
            "teleport_count": summary.get("teleport_count", 0),
            "collision_count": summary.get("collision_count", 0),
            "episode_reward": round(total_reward, 2),
            "wall_clock_time_seconds": round(wall_clock, 2),
            "is_valid": summary.get("is_valid", True),
            "status": summary.get("status", "OK"),
        }

        # Tripinfo metrics if available
        for k, v in summary.items():
            if k.startswith("tripinfo_"):
                eval_result[k] = v

        out_json = os.path.join(output_dir, f"{scenario_name}_{algo_key}_eval.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2)

        print(
            f"     Eval Completed in {wall_clock:.2f}s | "
            f"Completed: {eval_result['total_completed_vehicles']} | "
            f"Mean Wait: {eval_result['mean_waiting_time']:.2f}s | "
            f"Mean Queue: {eval_result['mean_queue_length']:.2f} | "
            f"Reward: {eval_result['episode_reward']:.1f}"
        )

        return eval_result

    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained RL controller.")
    parser.add_argument("--dataset", type=str, required=True, choices=["cologne1", "ingolstadt1"])
    parser.add_argument("--algo", type=str, required=True, choices=["q_learning", "dqn"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3600)

    args = parser.parse_args()
    evaluate_agent(
        scenario_name=args.dataset,
        algorithm=args.algo,
        model_path=args.model,
        seed=args.seed,
        steps=args.steps,
    )
