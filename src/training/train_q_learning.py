"""Training pipeline for Tabular Q-Learning on SUMO traffic networks."""

import argparse
import math
import json
import os
import time
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.config import DEFAULT_SAFETY_CONFIG, load_yaml_config
from src.controllers.q_learning import QLearningController
from src.metrics import BENCHMARK_COMPLETION_THRESHOLD, completion_statistics
from src.evaluation.evaluate_rl import run_greedy_eval_loop
from src.rl.discretization import StateDiscretizer
from src.rl.environment import TrafficSignalRLEnv
from src.rl.rewards import BaseRewardFunction, NegativePressureReward


def validate_completion_threshold(value: float) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("completion threshold must be finite and within [0, 1]")


def is_validation_viable(result: dict, min_completion: float = BENCHMARK_COMPLETION_THRESHOLD) -> tuple[bool, list[str]]:
    """Simple viability check: no collisions/teleports and sufficient vehicle completion rate."""
    validate_completion_threshold(min_completion)
    s = {**result, **result.get("summary", {})}
    reasons = []
    if s.get("collision_count", 0) > 0:
        reasons.append("has_collisions")
    if s.get("teleport_count", 0) > 0:
        reasons.append("has_teleports")

    _, demand, comp_rate = completion_statistics(s, result.get("total_demand"))
    if demand == 0:
        reasons.append("NO_DEMAND")
    elif demand is None:
        comp_rate = result.get("completion_rate")
    if comp_rate is None or not math.isfinite(comp_rate):
        reasons.append("missing_completion")
        return False, reasons

    if comp_rate < min_completion:
        reasons.append(f"low_completion: {comp_rate:.2%}")

    return len(reasons) == 0, reasons


def validation_data_errors(result: dict) -> list[str]:
    """Simple check for recording or measurement errors."""
    s = result.get("summary", result)
    errors = []
    if s.get("is_valid") is False:
        errors.append("data_invalid")
    if s.get("measurement_errors", 0) > 0:
        errors.append("measurement_errors")
    if s.get("tripinfo_required") is not True or s.get("tripinfo_reconciliation_status") != "MATCH":
        errors.append("tripinfo_unverified")
    return errors


class ValidationCheckpointSelector:
    """Selects best checkpoint based on validation reward and completion."""

    def __init__(self, min_completion: float = BENCHMARK_COMPLETION_THRESHOLD):
        validate_completion_threshold(min_completion)
        self.min_completion = min_completion
        self.best_reward = -float("inf")
        self.episode = None

    def is_viable(self, results: list) -> bool:
        return bool(results) and all(
            r.get("reward") is not None and math.isfinite(r["reward"])
            and r.get("summary", r).get("is_valid") is True
            and is_validation_viable(r, self.min_completion)[0] and not validation_data_errors(r)
            for r in results
        )

    def consider(self, episode: int, results: list) -> bool:
        if not self.is_viable(results):
            return False
        avg_rew = float(np.mean([r["reward"] for r in results]))
        if avg_rew > self.best_reward:
            self.best_reward = avg_rew
            self.episode = episode
            return True
        return False


def evaluate_greedy_q_learning(
    agent: QLearningController,
    scenario_name: str,
    decision_interval: float = 5.0,
    min_green: float = DEFAULT_SAFETY_CONFIG["min_green"],
    yellow_duration: float = DEFAULT_SAFETY_CONFIG["yellow_duration"],
    all_red_duration: float = DEFAULT_SAFETY_CONFIG["all_red_duration"],
    max_green: Optional[float] = DEFAULT_SAFETY_CONFIG["max_green"],
    reward_fn: Optional[BaseRewardFunction] = None,
    state_discretizer: Optional[StateDiscretizer] = None,
    max_steps: Optional[int] = 3600,
    seed: int = 42,
    output_dir: Optional[str] = None,
    save_run_details: bool = False,
) -> Dict[str, Any]:
    target_tls = getattr(agent, "target_tls_id", None)
    val_env = TrafficSignalRLEnv(
        scenario_name=scenario_name,
        decision_interval=decision_interval,
        min_green=min_green,
        max_green=max_green,
        yellow_duration=yellow_duration,
        all_red_duration=all_red_duration,
        reward_fn=reward_fn,
        state_discretizer=state_discretizer or getattr(agent, "discretizer", None),
        max_steps=max_steps,
        seed=seed,
        gui=False,
        output_dir=output_dir,
        save_run_details=save_run_details,
        target_tls_id=target_tls,
    )
    try:
        env_actions = getattr(val_env, "num_actions", None)
        agent_actions = getattr(agent, "num_actions", None)
        if env_actions is not None and agent_actions is not None and env_actions != agent_actions:
            raise ValueError(
                f"Action space mismatch: agent expects {agent_actions} actions, but environment target '{val_env.tls_id}' has {env_actions} actions."
            )
        total_reward, summary = run_greedy_eval_loop(agent, val_env, seed=seed)
    finally:
        val_env.close()
    completed, total_demand, completion_rate = completion_statistics(summary)
    collisions = summary.get("collision_count", summary.get("collision_incidents", 0))
    teleports = summary.get("teleport_count", 0)
    mean_wait = summary.get("mean_waiting_time", 0.0)
    sys_delay = summary.get("system_mean_delay", 0.0)

    viable, reasons = is_validation_viable({"summary": summary, "completed_vehicles": completed, "total_demand": total_demand})
    return {
        "validation_viable": viable,
        "validation_rejection_reasons": reasons,
        "reward": round(total_reward, 2),
        "completed_vehicles": completed,
        "total_demand": total_demand,
        "completion_rate": round(completion_rate, 4),
        "collisions": collisions,
        "teleports": teleports,
        "mean_waiting_time": round(mean_wait, 2) if mean_wait is not None else None,
        "system_mean_delay": round(sys_delay, 2) if sys_delay is not None else None,
        "seed": seed,
        "summary": summary,
    }


@dataclass
class TrainingConfig:
    """Configuration settings for Tabular Q-Learning training."""
    scenario_name: str = "ingolstadt1"
    num_episodes: int = 15
    seed: int = 42
    learning_rate: float = 0.1
    gamma: float = 0.95
    epsilon_start: float = 1.0
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.94
    decision_interval: float = 5.0
    min_green: float = DEFAULT_SAFETY_CONFIG["min_green"]
    max_green: Optional[float] = DEFAULT_SAFETY_CONFIG["max_green"]
    yellow_duration: float = DEFAULT_SAFETY_CONFIG["yellow_duration"]
    all_red_duration: float = DEFAULT_SAFETY_CONFIG["all_red_duration"]
    max_steps: Optional[int] = 3600
    val_interval: int = 5
    val_seeds: Optional[List[int]] = None
    min_val_completion: float = BENCHMARK_COMPLETION_THRESHOLD
    reward_scale: float = 1.0
    pending_weight: float = 0.5
    max_effective_green: float = 25.0
    serving_thresholds: Tuple[float, ...] = (2.0, 8.0)
    waiting_thresholds: Tuple[float, ...] = (4.0, 15.0, 35.0)
    model_dir: str = "models/q_learning"
    log_dir: str = "results/training"
    save_run_details: bool = False

    def __post_init__(self) -> None:
        validate_completion_threshold(self.min_val_completion)
        if self.num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive, got {self.num_episodes}")

    @classmethod
    def from_yaml(cls, config_path: str, **overrides: Any) -> "TrainingConfig":
        yaml_cfg = load_yaml_config(config_path)
        names = {item.name for item in fields(cls)}
        params = {key: value for group in (yaml_cfg.get("hyperparameters", {}), yaml_cfg)
                  for key, value in group.items() if key in names}
        rc = yaml_cfg.get("reward", {})
        if "scale" in rc or "reward_scale" in yaml_cfg:
            params["reward_scale"] = float(rc.get("scale", yaml_cfg.get("reward_scale", 1.0)))
        if "pending_weight" in rc:
            params["pending_weight"] = float(rc["pending_weight"])
        dc = yaml_cfg.get("discretization", {})
        if "max_effective_green" in dc:
            params["max_effective_green"] = float(dc["max_effective_green"])
        if "serving_queue_thresholds" in dc:
            params["serving_thresholds"] = tuple(float(x) for x in dc["serving_queue_thresholds"])
        if "waiting_queue_thresholds" in dc:
            params["waiting_thresholds"] = tuple(float(x) for x in dc["waiting_queue_thresholds"])

        params.update(overrides)
        return cls(**params)


def train_q_learning(config: Optional[TrainingConfig] = None, **legacy_options: Any) -> Dict[str, Any]:
    """Train from a config; existing keyword calls remain supported by this adapter."""
    if config is not None and legacy_options:
        raise TypeError("Pass config or training keyword options, not both")
    if config is None:
        config_path = legacy_options.pop("config_path", None)
        max_steps = legacy_options.pop("max_steps_per_episode", None)
        options = {k: v for k, v in legacy_options.items() if v is not None or k in ("max_green", "max_steps")}
        if max_steps is not None:
            options["max_steps"] = max_steps
        config = TrainingConfig.from_yaml(config_path, **options) if config_path else TrainingConfig(**options)
    cfg = config

    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    reward_fn = NegativePressureReward(scale=cfg.reward_scale, pending_weight=cfg.pending_weight)
    state_discretizer = StateDiscretizer(
        min_green=cfg.min_green,
        max_effective_green=cfg.max_effective_green,
        serving_queue_thresholds=cfg.serving_thresholds,
        waiting_queue_thresholds=cfg.waiting_thresholds,
    )

    print(f"\n{'='*56}")
    print(f"Starting Q-Learning Training on [{cfg.scenario_name.upper()}]")
    print(f"Episodes: {cfg.num_episodes} | LR: {cfg.learning_rate} | Gamma: {cfg.gamma} | Decay: {cfg.epsilon_decay}")
    print(f"{'='*56}")

    env_output_dir = os.path.join(cfg.log_dir, f"env_{cfg.scenario_name}") if cfg.save_run_details else None
    timing_options = {key: getattr(cfg, key) for key in (*DEFAULT_SAFETY_CONFIG, "decision_interval")}
    env = TrafficSignalRLEnv(
        scenario_name=cfg.scenario_name,
        **timing_options,
        reward_fn=reward_fn,
        state_discretizer=state_discretizer,
        max_steps=cfg.max_steps,
        seed=cfg.seed,
        output_dir=env_output_dir,
        save_run_details=cfg.save_run_details,
        use_tripinfo=True,
    )

    agent = QLearningController(
        num_actions=env.num_actions,
        target_tls_id=env.tls_id,
        learning_rate=cfg.learning_rate,
        gamma=cfg.gamma,
        epsilon=cfg.epsilon_start,
        epsilon_min=cfg.epsilon_min,
        epsilon_decay=cfg.epsilon_decay,
        discretizer=state_discretizer,
        **timing_options,
        seed=cfg.seed,
        eval_mode=False,
    )

    agent.reward_config = reward_fn.get_config()
    history: List[Dict[str, Any]] = []
    selector = ValidationCheckpointSelector(min_completion=cfg.min_val_completion)
    best_path: Optional[str] = None
    start_time = time.time()

    try:
        for ep in range(1, cfg.num_episodes + 1):
            ep_seed = cfg.seed + ep
            ep_start = time.time()

            _, info = env.reset(seed=ep_seed)
            state = info["discrete_state"]
            valid_actions = info["valid_actions"]

            ep_reward = 0.0
            ep_steps = 0

            while True:
                action = agent.select_action_for_state(
                    state, valid_actions=valid_actions, current_phase=state[0]
                )
                _, reward, terminated, truncated, next_info = env.step(action)
                next_state = next_info["discrete_state"]
                next_valid = next_info["valid_actions"]
                done = bool(terminated or truncated)

                agent.update(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    valid_next_actions=next_valid,
                    terminated=bool(terminated),
                    truncated=bool(truncated),
                )

                ep_reward += reward
                ep_steps += 1
                state, valid_actions = next_state, next_valid
                if done:
                    break

            ep_duration = time.time() - ep_start
            curr_eps = agent.epsilon
            agent.decay_epsilon()

            summary = env.get_summary() or {}
            mean_wait = summary.get("mean_waiting_time", 0.0)
            mean_queue = summary.get("mean_queue_length", 0.0)
            completed_vehs = summary.get("total_completed_vehicles", 0)

            history.append({
                "episode": ep,
                "reward": round(ep_reward, 2),
                "steps": ep_steps,
                "epsilon": round(curr_eps, 4),
                "q_table_size": len(agent.q_table),
                "total_updates": agent.total_updates,
                "mean_waiting_time": round(mean_wait, 2) if mean_wait is not None else None,
                "mean_queue_length": round(mean_queue, 2),
                "completed_vehicles": completed_vehs,
                "duration_seconds": round(ep_duration, 2),
            })

            wait_str = f"{mean_wait:5.2f}s" if mean_wait is not None else "  N/A "
            print(
                f"Ep {ep:2d}/{cfg.num_episodes} | Rew: {ep_reward:8.1f} | Eps: {curr_eps:.3f} | "
                f"Q-States: {len(agent.q_table):3d} | Wait: {wait_str} | "
                f"Queue: {mean_queue:4.2f} | Completed: {completed_vehs:4d} | "
                f"Time: {ep_duration:4.1f}s"
            )

            # Validation checkpointing
            val_int = cfg.val_interval
            if val_int > 0 and (ep % val_int == 0 or ep == cfg.num_episodes):
                v_seeds = cfg.val_seeds or [cfg.seed + 9999, cfg.seed + 10000]
                val_results = [
                    evaluate_greedy_q_learning(
                        agent=agent,
                        scenario_name=cfg.scenario_name,
                        **timing_options,
                        reward_fn=NegativePressureReward(scale=cfg.reward_scale, pending_weight=cfg.pending_weight),
                        state_discretizer=state_discretizer,
                        max_steps=cfg.max_steps,
                        seed=vs,
                        output_dir=os.path.join(cfg.log_dir, "validation", f"ep{ep}_seed{vs}") if cfg.save_run_details else None,
                        save_run_details=cfg.save_run_details,
                    )
                    for vs in v_seeds
                ]

                avg_val_rew = float(np.mean([r["reward"] for r in val_results]))
                avg_val_comp = float(np.mean([r["completion_rate"] for r in val_results]))
                is_viable = selector.is_viable(val_results)

                print(f"    --> [Eval] Ep {ep} | Rew: {avg_val_rew:.1f} | Comp: {avg_val_comp:.1%} | Viable: {is_viable}")

                if selector.consider(ep, val_results):
                    best_path = os.path.join(cfg.model_dir, f"q_learning_{cfg.scenario_name}_best.json")
                    agent.save(best_path)
                    print(f"    --> [BEST] Saved to {best_path}")

            if cfg.save_run_details and (ep % 5 == 0 or ep == cfg.num_episodes):
                agent.save(os.path.join(cfg.model_dir, f"q_learning_{cfg.scenario_name}_ep{ep}.json"))

    finally:
        env.close()

    total_wall_time = time.time() - start_time
    final_model_path = os.path.join(cfg.model_dir, f"q_learning_{cfg.scenario_name}_final.json")
    agent.save(final_model_path)

    log_file = os.path.join(cfg.log_dir, f"q_learning_{cfg.scenario_name}_log.json")
    training_data = {
        "scenario_name": cfg.scenario_name,
        "algorithm": "Q-Learning",
        "num_episodes": cfg.num_episodes,
        "seed": cfg.seed,
        "hyperparameters": {
            k: getattr(cfg, k)
            for k in [
                "learning_rate", "gamma", "epsilon_start", "epsilon_min",
                "epsilon_decay", "decision_interval", "min_green", "max_green",
            ]
        },
        "total_wall_clock_seconds": round(total_wall_time, 2),
        "history": history,
        "final_q_table_size": len(agent.q_table),
        "best_reward": round(selector.best_reward, 2) if selector.episode is not None else None,
        "best_is_viable": selector.episode is not None,
        "best_checkpoint_path": best_path,
        "final_checkpoint_path": final_model_path,
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2)

    print(f"\nTraining completed in {total_wall_time:.2f}s.")
    print(f"Final Model saved to: {final_model_path}")
    print(f"Training Log saved to: {log_file}")
    return training_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tabular Q-Learning on SUMO.", argument_default=argparse.SUPPRESS)
    parser.add_argument("--dataset", dest="scenario_name", choices=["cologne1", "ingolstadt1", "cologne3"])
    for flag, dest, value_type in (
        ("episodes", "num_episodes", int), ("lr", "learning_rate", float),
        ("gamma", "gamma", float), ("decay", "epsilon_decay", float),
        ("steps", "max_steps", int), ("seed", "seed", int),
        ("min-completion", "min_val_completion", float), ("val-interval", "val_interval", int),
    ):
        parser.add_argument(f"--{flag}", dest=dest, type=value_type)
    parser.add_argument("--max-green", type=lambda value: None if value.lower() == "none" else float(value))
    parser.add_argument(
        "--config",
        dest="config_path",
        default="configs/q_learning.yaml" if os.path.isfile("configs/q_learning.yaml") else None,
        help="Path to YAML configuration",
    )
    parser.add_argument("--val-seeds", type=int, nargs="+")
    parser.add_argument("--save-details", dest="save_run_details", action="store_true")
    train_q_learning(**vars(parser.parse_args()))
