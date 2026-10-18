"""Evaluation pipeline for trained Q-Learning agents (simplified)."""

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

from src.config import DEFAULT_SAFETY_CONFIG
from src.controllers.q_learning import QLearningController
from src.rl.environment import TrafficSignalRLEnv


def load_checkpoint_env_params(model_path: Optional[str]) -> Dict[str, Any]:
    """Inspects checkpoint metadata to extract environment and discretizer configuration."""
    params: Dict[str, Any] = {}
    if not model_path or not os.path.isfile(model_path):
        return params

    try:
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return params

    cfg = data.get("config", {}) if isinstance(data, dict) else {}
    if not isinstance(cfg, dict):
        return params

    if "num_actions" in cfg:
        params["num_actions"] = int(cfg["num_actions"])
    if "target_tls_id" in cfg and cfg["target_tls_id"] is not None:
        params["target_tls_id"] = str(cfg["target_tls_id"])

    params["decision_interval"] = float(cfg.get("decision_interval", 5.0))
    params["yellow_duration"] = float(cfg.get("yellow_duration", DEFAULT_SAFETY_CONFIG["yellow_duration"]))
    params["all_red_duration"] = float(cfg.get("all_red_duration", DEFAULT_SAFETY_CONFIG["all_red_duration"]))
    params["min_green"] = float(cfg.get("min_green", DEFAULT_SAFETY_CONFIG["min_green"]))
    params["max_green"] = float(cfg["max_green"]) if cfg.get("max_green") is not None else DEFAULT_SAFETY_CONFIG["max_green"]

    d_cfg = cfg.get("discretizer_config", {})
    if isinstance(d_cfg, dict) and d_cfg:
        from src.rl.discretization import StateDiscretizer
        params["state_discretizer"] = StateDiscretizer(
            min_green=float(d_cfg.get("min_green", params["min_green"])),
            max_effective_green=float(d_cfg.get("max_effective_green", 25.0)),
            serving_queue_thresholds=tuple(float(x) for x in d_cfg.get("serving_thresholds", [1.0, 5.0])),
            waiting_queue_thresholds=tuple(float(x) for x in d_cfg.get("waiting_thresholds", [1.0, 6.0, 15.0])),
        )
    return params


def evaluate_agent(
    scenario_name: str,
    algorithm: str,
    model_path: str,
    seed: int = 42,
    steps: Optional[int] = 3600,
    decision_interval: Optional[float] = None,
    min_green: Optional[float] = None,
    max_green: Optional[float] = None,
    yellow_duration: Optional[float] = None,
    all_red_duration: Optional[float] = None,
    target_tls_id: Optional[str] = None,
    output_dir: str = "results/evaluation",
    gui: bool = False,
    gui_delay: float = 0.0,
    save_run_details: bool = False,
) -> Dict[str, Any]:
    """Runs an evaluation episode with greedy exploitation (epsilon=0.0)."""
    os.makedirs(output_dir, exist_ok=True)
    algo_key = algorithm.lower().replace("-", "_")
    ckpt_params = load_checkpoint_env_params(model_path)

    # Resolve timing parameters gracefully without throwing constraint exceptions
    dec_int = float(decision_interval) if decision_interval is not None else ckpt_params.get("decision_interval", 5.0)
    m_green = float(min_green) if min_green is not None else ckpt_params.get("min_green", DEFAULT_SAFETY_CONFIG["min_green"])
    mx_green = float(max_green) if max_green is not None else ckpt_params.get("max_green", DEFAULT_SAFETY_CONFIG["max_green"])
    y_dur = float(yellow_duration) if yellow_duration is not None else ckpt_params.get("yellow_duration", DEFAULT_SAFETY_CONFIG["yellow_duration"])
    ar_dur = float(all_red_duration) if all_red_duration is not None else ckpt_params.get("all_red_duration", DEFAULT_SAFETY_CONFIG["all_red_duration"])
    effective_tls = target_tls_id or ckpt_params.get("target_tls_id")

    env_kwargs: Dict[str, Any] = {
        "scenario_name": scenario_name,
        "decision_interval": dec_int,
        "min_green": m_green,
        "max_green": mx_green,
        "yellow_duration": y_dur,
        "all_red_duration": ar_dur,
        "target_tls_id": effective_tls,
        "max_steps": steps,
        "seed": seed,
        "output_dir": os.path.join(output_dir, f"eval_{scenario_name}_{algo_key}") if save_run_details else None,
        "save_run_details": save_run_details,
        "use_tripinfo": True,
        "gui": gui,
        "gui_delay": gui_delay,
    }
    if "state_discretizer" in ckpt_params:
        env_kwargs["state_discretizer"] = ckpt_params["state_discretizer"]

    env = TrafficSignalRLEnv(**env_kwargs)
    try:
        agent = QLearningController(
            num_actions=env.num_actions,
            discretizer=env.state_discretizer,
            min_green=m_green,
            max_green=mx_green,
            yellow_duration=y_dur,
            all_red_duration=ar_dur,
            target_tls_id=env.tls_id,
            seed=seed,
            eval_mode=True,
            tie_break_strategy="first",
        )
        if model_path and os.path.isfile(model_path):
            agent.load(model_path)
        agent.decision_interval = dec_int
        agent.eval_mode = True
        agent.epsilon = 0.0

        start_time = time.time()
        _, info = env.reset(seed=seed)
        valid_actions = info.get("valid_actions")
        discrete_state = info.get("discrete_state")

        total_reward = 0.0
        while True:
            action = agent.select_action_for_state(
                discrete_state, valid_actions=valid_actions, current_phase=discrete_state[0] if discrete_state else 0
            )
            _, reward, terminated, truncated, next_info = env.step(action)
            total_reward += reward
            valid_actions = next_info.get("valid_actions")
            discrete_state = next_info.get("discrete_state")
            if terminated or truncated:
                break

        wall_clock = time.time() - start_time
        summary = env.get_summary() or {}

        eval_result = {
            "scenario_name": scenario_name,
            "algorithm": "Q-Learning",
            "model_path": model_path,
            "seed": seed,
            "status": "SUCCESS",
            "is_valid": True,
            "episode_reward": round(total_reward, 2),
            "wall_clock_time_seconds": round(wall_clock, 2),
            "summary": summary,
        }
        # Copy key metrics into the top-level result
        for k in (
            "total_steps", "total_completed_vehicles", "mean_waiting_time",
            "mean_queue_length", "max_queue_length", "total_waiting_time",
            "mean_travel_time", "total_time_loss", "collision_count",
            "teleport_count", "system_mean_delay", "tripinfo_mean_waiting_time",
            "tripinfo_mean_travel_time", "tripinfo_total_time_loss"
        ):
            if k in summary:
                eval_result[k] = summary[k]

        out_json = os.path.join(output_dir, f"{scenario_name}_{algo_key}_eval.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2)

        mw = eval_result.get("tripinfo_mean_waiting_time", eval_result.get("mean_waiting_time", 0.0))
        print(f"[{scenario_name.upper()}] Q-Learning eval finished in {wall_clock:.1f}s | Completed: {eval_result.get('total_completed_vehicles', 0)} | Mean Wait: {mw:.2f}s")
        return eval_result
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained RL controller.")
    parser.add_argument("--dataset", type=str, required=True, choices=["cologne1", "ingolstadt1"])
    parser.add_argument("--algo", type=str, default="q_learning", choices=["q_learning"])
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--interval", type=float, default=None)
    parser.add_argument("--min-green", type=float, default=None)
    parser.add_argument("--max-green", type=float, default=None)
    parser.add_argument("--yellow", type=float, default=None)
    parser.add_argument("--all-red", type=float, default=None)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--gui-delay", type=float, default=0.0)
    parser.add_argument("--save-details", action="store_true", help="Save detailed simulation files (XML/CSV)")

    args = parser.parse_args()
    evaluate_agent(
        scenario_name=args.dataset,
        algorithm=args.algo,
        model_path=args.model,
        seed=args.seed,
        steps=args.steps,
        decision_interval=args.interval,
        min_green=args.min_green,
        max_green=args.max_green,
        yellow_duration=args.yellow,
        all_red_duration=args.all_red,
        gui=args.gui,
        gui_delay=args.gui_delay,
        save_run_details=args.save_details,
    )
