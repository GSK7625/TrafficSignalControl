"""Evaluation pipeline for trained Q-Learning and DQN agents."""

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

from src.controllers.dqn import DQNController
from src.controllers.q_learning import QLearningController
from src.rl.environment import TrafficSignalRLEnv


def load_checkpoint_env_params(model_path: Optional[str]) -> Dict[str, Any]:
    """Inspects checkpoint metadata to extract observation builder and discretizer configuration."""
    params: Dict[str, Any] = {}
    if not model_path or not os.path.isfile(model_path):
        return params

    try:
        if model_path.endswith(".json"):
            with open(model_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            cfg = data.get("config", {})
            if "decision_interval" in cfg:
                params["decision_interval"] = float(cfg["decision_interval"])
            if "yellow_duration" in cfg:
                params["yellow_duration"] = float(cfg["yellow_duration"])
            if "all_red_duration" in cfg:
                params["all_red_duration"] = float(cfg["all_red_duration"])
            if "min_green" in cfg:
                params["min_green"] = float(cfg["min_green"])
            d_cfg = cfg.get("discretizer_config", {})
            if d_cfg:
                if "min_green" in d_cfg:
                    params["min_green"] = float(d_cfg["min_green"])
                if "max_effective_green" in d_cfg:
                    params["max_effective_green"] = float(d_cfg["max_effective_green"])
                if "serving_thresholds" in d_cfg:
                    params["serving_queue_thresholds"] = tuple(float(x) for x in d_cfg["serving_thresholds"])
                if "waiting_thresholds" in d_cfg:
                    params["waiting_queue_thresholds"] = tuple(float(x) for x in d_cfg["waiting_thresholds"])
                from src.rl.discretization import StateDiscretizer
                params["state_discretizer"] = StateDiscretizer(
                    min_green=params.get("min_green", 5.0),
                    max_effective_green=params.get("max_effective_green", 25.0),
                    serving_queue_thresholds=params.get("serving_queue_thresholds", (1.0, 5.0)),
                    waiting_queue_thresholds=params.get("waiting_queue_thresholds", (1.0, 6.0, 15.0)),
                )
        elif model_path.endswith(".pt") or model_path.endswith(".pth"):
            import torch
            ckpt = torch.load(model_path, map_location="cpu")
            cfg = ckpt.get("config", {})
            if "decision_interval" in cfg:
                params["decision_interval"] = float(cfg["decision_interval"])
            if "yellow_duration" in cfg:
                params["yellow_duration"] = float(cfg["yellow_duration"])
            if "all_red_duration" in cfg:
                params["all_red_duration"] = float(cfg["all_red_duration"])
            if "min_green" in cfg:
                params["min_green"] = float(cfg["min_green"])
            obs_cfg = cfg.get("obs_builder_config", {})
            if obs_cfg:
                if "max_queue_scale" in obs_cfg:
                    params["max_queue_scale"] = float(obs_cfg["max_queue_scale"])
                if "max_time_scale" in obs_cfg:
                    params["max_time_scale"] = float(obs_cfg["max_time_scale"])
                if "min_green" in obs_cfg:
                    params["min_green"] = float(obs_cfg["min_green"])
    except Exception:
        pass
    return params


def evaluate_agent(
    scenario_name: str,
    algorithm: str,
    model_path: str,
    seed: int = 42,
    steps: Optional[int] = 3600,
    decision_interval: Optional[float] = None,
    min_green: Optional[float] = None,
    yellow_duration: Optional[float] = None,
    all_red_duration: Optional[float] = None,
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
        yellow_duration: Yellow transition duration.
        all_red_duration: All-red clearance duration.
        output_dir: Output directory for evaluation results.

    Returns:
        Evaluation summary metrics dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    algo_key = algorithm.lower().replace("-", "_")

    print(f"\n---> Evaluating [{algo_key.upper()}] on [{scenario_name.upper()}] (Seed: {seed})...")

    ckpt_params = load_checkpoint_env_params(model_path)
    dec_int = decision_interval if decision_interval is not None else ckpt_params.get("decision_interval", 5.0)
    m_green = min_green if min_green is not None else ckpt_params.get("min_green", 5.0)
    y_dur = yellow_duration if yellow_duration is not None else ckpt_params.get("yellow_duration", 3.0)
    ar_dur = all_red_duration if all_red_duration is not None else ckpt_params.get("all_red_duration", 0.0)

    env_kwargs: Dict[str, Any] = {
        "scenario_name": scenario_name,
        "decision_interval": dec_int,
        "min_green": m_green,
        "yellow_duration": y_dur,
        "all_red_duration": ar_dur,
        "max_steps": steps,
        "seed": seed,
        "output_dir": os.path.join(output_dir, f"eval_{scenario_name}_{algo_key}"),
        "use_tripinfo": True,
    }
    if "max_queue_scale" in ckpt_params:
        env_kwargs["max_queue_scale"] = ckpt_params["max_queue_scale"]
    if "max_time_scale" in ckpt_params:
        env_kwargs["max_time_scale"] = ckpt_params["max_time_scale"]
    if "state_discretizer" in ckpt_params:
        env_kwargs["state_discretizer"] = ckpt_params["state_discretizer"]

    env = TrafficSignalRLEnv(**env_kwargs)

    try:
        # Instantiate and load agent in eval_mode
        if algo_key in ("q_learning", "qlearning"):
            agent = QLearningController(
                num_actions=env.num_actions,
                discretizer=env.state_discretizer,
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
            "total_steps": summary.get("total_steps", step_count * int(dec_int)),
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
    parser.add_argument("--interval", type=float, default=None, help="Decision interval (default: from checkpoint or 5.0)")
    parser.add_argument("--min-green", type=float, default=None, help="Min green duration (default: from checkpoint or 5.0)")
    parser.add_argument("--yellow", type=float, default=None, help="Yellow transition duration (default: from checkpoint or 3.0)")
    parser.add_argument("--all-red", type=float, default=None, help="All-red clearance duration (default: from checkpoint or 0.0)")

    args = parser.parse_args()
    evaluate_agent(
        scenario_name=args.dataset,
        algorithm=args.algo,
        model_path=args.model,
        seed=args.seed,
        steps=args.steps,
        decision_interval=args.interval,
        min_green=args.min_green,
        yellow_duration=args.yellow,
        all_red_duration=args.all_red,
    )
