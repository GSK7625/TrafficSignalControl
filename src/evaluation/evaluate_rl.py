"""Evaluation pipeline for trained Q-Learning agents (simplified)."""

import argparse
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

from src.config import DEFAULT_SAFETY_CONFIG
from src.controllers.q_learning import QLearningController
from src.rl.discretization import StateDiscretizer
from src.rl.environment import TrafficSignalRLEnv
from src.rl.rewards import NegativePressureReward


def load_checkpoint_env_params(model_path: Optional[str]) -> Dict[str, Any]:
    """Inspects checkpoint metadata to extract environment and discretizer configuration."""
    if model_path is None:
        return {}
    return checkpoint_env_params(QLearningController.from_checkpoint(model_path).get_config())


def checkpoint_env_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Maps controller metadata to environment options in one place."""
    params: Dict[str, Any] = {}
    if "num_actions" in cfg:
        params["num_actions"] = int(cfg["num_actions"])
    if "target_tls_id" in cfg and cfg["target_tls_id"] is not None:
        params["target_tls_id"] = str(cfg["target_tls_id"])

    params["decision_interval"] = float(cfg.get("decision_interval", 5.0))
    params["yellow_duration"] = float(cfg.get("yellow_duration", DEFAULT_SAFETY_CONFIG["yellow_duration"]))
    params["all_red_duration"] = float(cfg.get("all_red_duration", DEFAULT_SAFETY_CONFIG["all_red_duration"]))
    params["min_green"] = float(cfg.get("min_green", DEFAULT_SAFETY_CONFIG["min_green"]))
    max_green = cfg.get("max_green", DEFAULT_SAFETY_CONFIG["max_green"])
    params["max_green"] = float(max_green) if max_green is not None else None

    d_cfg = cfg.get("discretizer_config", {})
    if isinstance(d_cfg, dict) and d_cfg:
        params["state_discretizer"] = StateDiscretizer.from_config(d_cfg, params["min_green"])
    return params


def create_evaluation_session(
    scenario_name: str,
    model_path: Optional[str] = None,
    seed: int = 42,
    timing_overrides: Optional[Dict[str, Any]] = None,
    **env_options: Any,
) -> Tuple[QLearningController, TrafficSignalRLEnv]:
    """Creates a compatible agent/environment pair for CLI and benchmark evaluation.

    Explicit timing options override checkpoint metadata. An explicit max_green=None
    disables the limit; an absent key inherits checkpoint or project defaults.
    The caller owns the returned environment and must close it.
    """
    if not model_path:
        raise FileNotFoundError("Evaluation requires a trained Q-Learning checkpoint")
    agent = QLearningController.from_checkpoint(model_path, seed=seed, **(timing_overrides or {}))
    params = checkpoint_env_params(agent.get_config())
    params.pop("num_actions", None)
    params.update(timing_overrides or {})
    params["state_discretizer"] = agent.discretizer
    reward = dict(agent.reward_config or {})
    if reward.pop("name", "NegativePressureReward") != "NegativePressureReward":
        raise ValueError("Unsupported checkpoint reward function")
    env_options.setdefault("reward_fn", NegativePressureReward(**reward))
    env = TrafficSignalRLEnv(
        scenario_name=scenario_name, seed=seed, **{**params, **env_options},
    )
    env.reward_config_source = "checkpoint" if agent.reward_config else "legacy_defaults"
    try:
        if agent.num_actions != env.num_actions:
            raise ValueError(f"Checkpoint actions ({agent.num_actions}) do not match environment actions ({env.num_actions})")
        agent.target_tls_id = env.tls_id
        return agent, env
    except Exception:
        env.close()
        raise


def run_greedy_eval_loop(
    agent: Any,
    env: Any,
    seed: Optional[int] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Runs a single deterministic greedy evaluation episode on the environment.

    Temporarily sets agent.eval_mode=True and agent.epsilon=0.0 during execution.
    Returns:
        (total_cumulative_reward, env_summary_dict)
    """
    prev_mode = agent.eval_mode
    prev_eps = agent.epsilon
    agent.eval_mode = True
    agent.epsilon = 0.0

    total_reward = 0.0
    try:
        _, info = env.reset(seed=seed)
        valid_actions = info["valid_actions"]
        discrete_state = info["discrete_state"]

        while True:
            curr_phase = discrete_state[0]
            action = agent.select_action_for_state(
                discrete_state, valid_actions=valid_actions, current_phase=curr_phase
            )
            _, reward, terminated, truncated, next_info = env.step(action)
            total_reward += reward
            valid_actions = next_info["valid_actions"]
            discrete_state = next_info["discrete_state"]
            if terminated or truncated:
                break
    finally:
        agent.eval_mode = prev_mode
        agent.epsilon = prev_eps

    summary = env.get_summary() or {}
    return float(round(total_reward, 2)), summary


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
    timing_overrides = {
        key: value for key, value in {
            "decision_interval": decision_interval, "min_green": min_green,
            "max_green": max_green, "yellow_duration": yellow_duration,
            "all_red_duration": all_red_duration,
        }.items() if value is not None
    }
    env_options = {"target_tls_id": target_tls_id} if target_tls_id else {}
    agent, env = create_evaluation_session(
        scenario_name, model_path, seed=seed, timing_overrides=timing_overrides,
        max_steps=steps, gui=gui, gui_delay=gui_delay, use_tripinfo=True,
        output_dir=os.path.join(output_dir, f"eval_{scenario_name}_{algo_key}") if save_run_details else None,
        save_run_details=save_run_details, **env_options,
    )
    try:
        start_time = time.time()
        total_reward, summary = run_greedy_eval_loop(agent, env, seed=seed)
        wall_clock = time.time() - start_time

        eval_result = {
            "scenario_name": scenario_name,
            "algorithm": "Q-Learning",
            "model_path": model_path,
            "seed": seed,
            "status": "SUCCESS" if summary.get("is_valid", True) else "FAILED",
            "is_valid": summary.get("is_valid", True),
            "episode_reward": round(total_reward, 2),
            "reward_config": env.reward_fn.get_config(),
            "reward_config_source": env.reward_config_source,
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
        wait_label = f"{mw:.2f}s" if mw is not None else "N/A"
        print(f"[{scenario_name.upper()}] Q-Learning eval finished in {wall_clock:.1f}s | Completed: {eval_result.get('total_completed_vehicles', 0)} | Mean Wait: {wait_label}")
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
