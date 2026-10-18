"""Evaluation pipeline for trained Q-Learning agents."""

import argparse
import json
import os
import time
from typing import Any, Dict, Optional

from src.training.train_q_learning import is_validation_viable, validation_data_errors
from src.evaluation.eval_audit import snapshot_agent, finish_eval_audit
from src.controllers.q_learning import QLearningController
from src.rl.environment import TrafficSignalRLEnv


def load_checkpoint_env_params(model_path: Optional[str]) -> Dict[str, Any]:
    """Inspects checkpoint metadata to extract observation builder and discretizer configuration."""
    params: Dict[str, Any] = {}
    if not model_path:
        return params

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model checkpoint file not found: '{model_path}'")

    try:
        with open(model_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Malformed checkpoint file '{model_path}': invalid JSON ({e})") from e

    if not isinstance(data, dict) or "config" not in data:
        raise ValueError(f"Malformed checkpoint file '{model_path}': missing required 'config' section")

    cfg = data.get("config", {})
    if not isinstance(cfg, dict):
        raise ValueError(f"Malformed checkpoint file '{model_path}': 'config' must be a dictionary")

    from src.config import DEFAULT_SAFETY_CONFIG

    if "num_actions" in cfg:
        params["num_actions"] = int(cfg["num_actions"])
    if "target_tls_id" in cfg and cfg["target_tls_id"] is not None:
        params["target_tls_id"] = str(cfg["target_tls_id"])

    params["decision_interval"] = float(cfg.get("decision_interval", 5.0))
    params["yellow_duration"] = float(cfg.get("yellow_duration", DEFAULT_SAFETY_CONFIG["yellow_duration"]))
    params["all_red_duration"] = float(cfg.get("all_red_duration", DEFAULT_SAFETY_CONFIG["all_red_duration"]))
    params["min_green"] = float(cfg.get("min_green", DEFAULT_SAFETY_CONFIG["min_green"]))

    # max_green restoration (F05)
    if "max_green" in cfg:
        params["max_green"] = float(cfg["max_green"]) if cfg["max_green"] is not None else None
    else:
        params["max_green"] = DEFAULT_SAFETY_CONFIG["max_green"]

    d_cfg = cfg.get("discretizer_config", {})
    if isinstance(d_cfg, dict) and d_cfg:
        min_g = float(d_cfg.get("min_green", params["min_green"]))
        max_eff_g = float(d_cfg.get("max_effective_green", 25.0))
        serving_th = tuple(float(x) for x in d_cfg.get("serving_thresholds", [1.0, 5.0]))
        waiting_th = tuple(float(x) for x in d_cfg.get("waiting_thresholds", [1.0, 6.0, 15.0]))
        from src.rl.discretization import StateDiscretizer
        params["state_discretizer"] = StateDiscretizer(
            min_green=min_g,
            max_effective_green=max_eff_g,
            serving_queue_thresholds=serving_th,
            waiting_queue_thresholds=waiting_th,
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
) -> Dict[str, Any]:
    """Runs a rigorous evaluation episode with exploration and learning disabled.

    Args:
        scenario_name: 'cologne1' or 'ingolstadt1'.
        algorithm: 'q_learning'.
        model_path: Path to model checkpoint file (.json for Q-Learning).
        seed: Random seed (matches baseline benchmarks).
        steps: Total steps to evaluate (default: 3600s).
        decision_interval: Decision interval in seconds.
        min_green: Minimum green constraint.
        max_green: Maximum green constraint.
        yellow_duration: Yellow transition duration.
        all_red_duration: All-red clearance duration.
        target_tls_id: Target traffic light ID.
        output_dir: Output directory for evaluation results.
        gui: Enable SUMO graphical user interface (sumo-gui).
        gui_delay: Step delay in milliseconds for GUI viewing.

    Returns:
        Evaluation summary metrics dictionary.
    """
    os.makedirs(output_dir, exist_ok=True)
    algo_key = algorithm.lower().replace("-", "_")

    print(f"\n---> Evaluating [{algo_key.upper()}] on [{scenario_name.upper()}] (Seed: {seed})...")

    from src.config import DEFAULT_SAFETY_CONFIG
    ckpt_params = load_checkpoint_env_params(model_path)
    overrides: Dict[str, Any] = {}

    if decision_interval is not None:
        dec_int = float(decision_interval)
        if "decision_interval" in ckpt_params and dec_int != ckpt_params["decision_interval"]:
            overrides["decision_interval"] = {"checkpoint": ckpt_params["decision_interval"], "override": dec_int}
    else:
        dec_int = ckpt_params.get("decision_interval", 5.0)

    if min_green is not None:
        m_green = float(min_green)
        if "min_green" in ckpt_params and m_green != ckpt_params["min_green"]:
            overrides["min_green"] = {"checkpoint": ckpt_params["min_green"], "override": m_green}
    else:
        m_green = ckpt_params.get("min_green", DEFAULT_SAFETY_CONFIG["min_green"])

    if max_green is not None:
        mx_green = float(max_green)
        if "max_green" in ckpt_params and mx_green != ckpt_params["max_green"]:
            overrides["max_green"] = {"checkpoint": ckpt_params["max_green"], "override": mx_green}
    else:
        mx_green = ckpt_params.get("max_green", DEFAULT_SAFETY_CONFIG["max_green"])

    if yellow_duration is not None:
        y_dur = float(yellow_duration)
        if "yellow_duration" in ckpt_params and y_dur != ckpt_params["yellow_duration"]:
            overrides["yellow_duration"] = {"checkpoint": ckpt_params["yellow_duration"], "override": y_dur}
    else:
        y_dur = ckpt_params.get("yellow_duration", DEFAULT_SAFETY_CONFIG["yellow_duration"])

    if all_red_duration is not None:
        ar_dur = float(all_red_duration)
        if "all_red_duration" in ckpt_params and ar_dur != ckpt_params["all_red_duration"]:
            overrides["all_red_duration"] = {"checkpoint": ckpt_params["all_red_duration"], "override": ar_dur}
    else:
        ar_dur = ckpt_params.get("all_red_duration", DEFAULT_SAFETY_CONFIG["all_red_duration"])

    effective_target_tls = target_tls_id or ckpt_params.get("target_tls_id")

    env_kwargs: Dict[str, Any] = {
        "scenario_name": scenario_name,
        "decision_interval": dec_int,
        "min_green": m_green,
        "max_green": mx_green,
        "yellow_duration": y_dur,
        "all_red_duration": ar_dur,
        "target_tls_id": effective_target_tls,
        "max_steps": steps,
        "seed": seed,
        "output_dir": os.path.join(output_dir, f"eval_{scenario_name}_{algo_key}"),
        "use_tripinfo": True,
        "gui": gui,
        "gui_delay": gui_delay,
    }
    if "max_queue_scale" in ckpt_params:
        env_kwargs["max_queue_scale"] = ckpt_params["max_queue_scale"]
    if "max_time_scale" in ckpt_params:
        env_kwargs["max_time_scale"] = ckpt_params["max_time_scale"]
    if "state_discretizer" in ckpt_params:
        env_kwargs["state_discretizer"] = ckpt_params["state_discretizer"]

    env = TrafficSignalRLEnv(**env_kwargs)

    # Validate action space and target TLS before running
    if "num_actions" in ckpt_params and ckpt_params["num_actions"] != env.num_actions:
        env.close()
        raise ValueError(
            f"Action space mismatch: checkpoint expects {ckpt_params['num_actions']} actions, "
            f"but scenario '{scenario_name}' TLS '{env.tls_id}' has {env.num_actions} green phases."
        )
    if (
        "target_tls_id" in ckpt_params
        and ckpt_params["target_tls_id"]
        and ckpt_params["target_tls_id"] != env.tls_id
    ):
        env.close()
        raise ValueError(
            f"Target TLS mismatch: checkpoint targets '{ckpt_params['target_tls_id']}', "
            f"but environment initialized with '{env.tls_id}'."
        )

    try:
        # Instantiate and load agent in eval_mode
        if algo_key in ("q_learning", "qlearning"):
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
            agent.load(model_path)
            # Loading restores checkpoint timing; explicit evaluation overrides
            # must remain effective in both the environment and loaded controller.
            agent.decision_interval = dec_int
            agent.min_green = m_green
            agent.max_green = mx_green
            agent.yellow_duration = y_dur
            agent.all_red_duration = ar_dur
            agent.eval_mode = True
            agent.epsilon = 0.0  # Zero exploration
        else:
            raise ValueError(f"Unsupported RL algorithm: {algorithm}")

        audit_before = snapshot_agent(agent)
        start_time = time.time()
        _, info = env.reset(seed=seed)
        valid_actions = info["valid_actions"]
        discrete_state = info["discrete_state"]

        total_reward = 0.0
        step_count = 0

        while True:
            # F02: pass current_phase=discrete_state[0] for consistent evaluation tie-breaking
            action = agent.select_action_for_state(
                discrete_state, valid_actions=valid_actions, current_phase=discrete_state[0]
            )

            _, reward, terminated, truncated, next_info = env.step(action)
            total_reward += reward
            step_count += 1

            valid_actions = next_info["valid_actions"]
            discrete_state = next_info["discrete_state"]

            if terminated or truncated:
                break

        wall_clock = time.time() - start_time
        summary = env.get_summary() or {}

        audit = finish_eval_audit(agent, audit_before)
        viable, reasons = is_validation_viable({"summary": summary, "eval_audit": audit})
        eval_result = {
            "validation_protocol_version": "stage04-schema4-v1",
            "validation_viable": viable,
            "validation_rejection_reasons": reasons,
            "eval_audit": audit,
            "summary": summary,
            "scenario_name": scenario_name,
            "algorithm": "Q-Learning",
            "model_path": model_path,
            "seed": seed,
            "overrides": overrides,
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
            "is_valid": not validation_data_errors({"summary": summary, "eval_audit": audit}),
            "status": "INVALID" if validation_data_errors({"summary": summary, "eval_audit": audit}) else summary.get("status", "UNKNOWN"),
        }

        # Tripinfo metrics if available
        for k, v in summary.items():
            if k.startswith("tripinfo_"):
                eval_result[k] = v

        out_json = os.path.join(output_dir, f"{scenario_name}_{algo_key}_eval.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(eval_result, f, indent=2)

        mw_val = eval_result.get("mean_waiting_time")
        mw_str = f"{mw_val:.2f}s" if mw_val is not None else "N/A"
        print(
            f"     Eval Completed in {wall_clock:.2f}s | "
            f"Completed: {eval_result['total_completed_vehicles']} | "
            f"Mean Wait: {mw_str} | "
            f"Mean Queue: {eval_result['mean_queue_length']:.2f} | "
            f"Reward: {eval_result['episode_reward']:.1f}"
        )

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
    parser.add_argument("--interval", type=float, default=None, help="Decision interval (default: from checkpoint or 5.0)")
    parser.add_argument("--min-green", type=float, default=None, help="Min green duration (default: from checkpoint or 5.0)")
    parser.add_argument("--max-green", type=float, default=None, help="Max green duration (default: from checkpoint or 60.0)")
    parser.add_argument("--yellow", type=float, default=None, help="Yellow transition duration (default: from checkpoint or 3.0)")
    parser.add_argument("--all-red", type=float, default=None, help="All-red clearance duration (default: from checkpoint or 1.0)")
    parser.add_argument("--gui", action="store_true", help="Enable SUMO graphical user interface (sumo-gui)")
    parser.add_argument("--gui-delay", type=float, default=0.0, help="Delay in milliseconds between simulation steps in SUMO-GUI")

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
    )
