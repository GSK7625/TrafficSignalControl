"""Training pipeline for Tabular Q-Learning on SUMO traffic networks."""

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from statistics import fmean
from typing import Any, Dict, List, Optional, Tuple

from src.config import DEFAULT_SAFETY_CONFIG, load_yaml_config, resolve_config_path
from src.controllers.q_learning import QLearningController
from src.evaluation.evaluate_rl import run_greedy_eval_loop
from src.io_utils import write_json_atomic
from src.metrics import BENCHMARK_COMPLETION_THRESHOLD, completion_statistics
from src.rl.discretization import StateDiscretizer
from src.rl.environment import TrafficSignalRLEnv
from src.rl.rewards import NegativePressureReward


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
        avg_rew = float(fmean(r["reward"] for r in results))
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
    reward_fn: Optional[NegativePressureReward] = None,
    state_discretizer: Optional[StateDiscretizer] = None,
    max_steps: Optional[int] = 3600,
    seed: int = 42,
    output_dir: Optional[str] = None,
    save_run_details: bool = False,
    max_red: Optional[float] = None,
) -> Dict[str, Any]:
    """Evaluate an agent greedily on a target scenario environment."""
    val_env = TrafficSignalRLEnv(
        scenario_name=scenario_name,
        decision_interval=decision_interval,
        min_green=min_green,
        max_green=max_green,
        max_red=max_red,
        yellow_duration=yellow_duration,
        all_red_duration=all_red_duration,
        reward_fn=reward_fn,
        state_discretizer=state_discretizer or getattr(agent, "discretizer", None),
        max_steps=max_steps,
        seed=seed,
        gui=False,
        output_dir=output_dir,
        save_run_details=save_run_details,
        target_tls_id=getattr(agent, "target_tls_id", None),
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
    viable, reasons = is_validation_viable({"summary": summary, "completed_vehicles": completed, "total_demand": total_demand})
    mean_wait = summary.get("mean_waiting_time", 0.0)
    sys_delay = summary.get("system_mean_delay", 0.0)

    return {
        "validation_viable": viable,
        "validation_rejection_reasons": reasons,
        "reward": round(total_reward, 2),
        "completed_vehicles": completed,
        "total_demand": total_demand,
        "completion_rate": round(completion_rate, 4),
        "collisions": summary.get("collision_count", summary.get("collision_incidents", 0)),
        "teleports": summary.get("teleport_count", 0),
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
    max_red: Optional[float] = None
    yellow_duration: float = DEFAULT_SAFETY_CONFIG["yellow_duration"]
    all_red_duration: float = DEFAULT_SAFETY_CONFIG["all_red_duration"]
    max_steps: Optional[int] = 3600
    val_interval: int = 5
    val_seeds: Optional[List[int]] = None
    min_val_completion: float = BENCHMARK_COMPLETION_THRESHOLD
    reward_scale: float = 1.0
    pending_weight: float = 0.5
    switch_penalty: float = 0.0
    consecutive_switch_penalty: float = 0.0
    max_effective_green: float = 25.0
    serving_thresholds: Tuple[float, ...] = (2.0, 8.0)
    waiting_thresholds: Tuple[float, ...] = (4.0, 15.0, 35.0)
    group_by: str = "aggregate"
    model_dir: str = "models/q_learning"
    log_dir: str = "results/training"
    save_run_details: bool = False

    def __post_init__(self) -> None:
        validate_completion_threshold(self.min_val_completion)
        if type(self.num_episodes) is not int or self.num_episodes <= 0:
            raise ValueError(f"num_episodes must be positive and an integer, got {self.num_episodes}")
        for key in ("learning_rate", "gamma", "epsilon_start", "epsilon_min", "epsilon_decay"):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be finite and within [0, 1]")
        if type(self.val_interval) is not int or self.val_interval < 0:
            raise ValueError("val_interval must be a non-negative integer")
        if self.max_steps is not None and (type(self.max_steps) is not int or self.max_steps < 0):
            raise ValueError("max_steps must be a non-negative integer or None")
        valid_modes = {"phase_group", "branch", "phase", "aggregate"}
        if self.group_by not in valid_modes:
            raise ValueError(f"group_by must be one of {valid_modes}, got {self.group_by}")

    @classmethod
    def from_yaml(cls, config_path: str, **overrides: Any) -> "TrainingConfig":
        config_path = resolve_config_path(config_path)
        yaml_cfg = load_yaml_config(config_path)
        if "q_learning" in yaml_cfg:
            master = yaml_cfg
            yaml_cfg = dict(master["q_learning"])
            if "safety" in master:
                for k, v in master["safety"].items():
                    yaml_cfg.setdefault(k, v)

        names = {item.name for item in fields(cls)}
        params = {k: v for g in (yaml_cfg.get("hyperparameters", {}), yaml_cfg) for k, v in g.items() if k in names}
        rc = yaml_cfg.get("reward", {})
        dc = yaml_cfg.get("discretization", {})

        mapping = [
            (rc.get("scale") if "scale" in rc else yaml_cfg.get("reward_scale"), "reward_scale", float),
            (rc.get("pending_weight"), "pending_weight", float),
            (rc.get("switch_penalty"), "switch_penalty", float),
            (rc.get("consecutive_switch_penalty"), "consecutive_switch_penalty", float),
            (dc.get("max_effective_green"), "max_effective_green", float),
            (dc.get("serving_queue_thresholds"), "serving_thresholds", lambda v: tuple(float(x) for x in v)),
            (dc.get("waiting_queue_thresholds"), "waiting_thresholds", lambda v: tuple(float(x) for x in v)),
            (dc.get("group_by") if "group_by" in dc else yaml_cfg.get("group_by"), "group_by", str),
        ]
        for src, dst, caster in mapping:
            if src is not None:
                params[dst] = caster(src)

        params.update(overrides)
        return cls(**params)


def _create_reward_fn(cfg: TrainingConfig) -> NegativePressureReward:
    return NegativePressureReward(
        scale=cfg.reward_scale,
        pending_weight=cfg.pending_weight,
    )


def _run_training_episode(env: TrafficSignalRLEnv, agent: QLearningController, ep_seed: int) -> Tuple[float, int]:
    """Execute a single training episode with action selection and Q-table updates."""
    state, info = env.reset(seed=ep_seed)
    if state is None:
        state = info["discrete_state"]
    valid_actions = info["valid_actions"]
    ep_reward, ep_steps = 0.0, 0

    while True:
        action = agent.select_action_for_state(state, valid_actions=valid_actions, current_phase=state[0])
        next_state, reward, terminated, truncated, next_info = env.step(action)
        if next_state is None:
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

    return ep_reward, ep_steps


def _evaluate_checkpoint(
    agent: QLearningController,
    cfg: TrainingConfig,
    ep: int,
    timing_options: Dict[str, Any],
    state_discretizer: StateDiscretizer,
    selector: ValidationCheckpointSelector,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Run greedy validation across configured seeds and update best checkpoint."""
    v_seeds = cfg.val_seeds or [cfg.seed + 9999, cfg.seed + 10000]
    val_results = [
        evaluate_greedy_q_learning(
            agent=agent,
            scenario_name=cfg.scenario_name,
            **timing_options,
            reward_fn=_create_reward_fn(cfg),
            state_discretizer=state_discretizer,
            max_steps=cfg.max_steps,
            seed=vs,
            output_dir=os.path.join(cfg.log_dir, "validation", f"ep{ep}_seed{vs}") if cfg.save_run_details else None,
            save_run_details=cfg.save_run_details,
        )
        for vs in v_seeds
    ]

    avg_rew = float(fmean(r["reward"] for r in val_results))
    avg_comp = float(fmean(r["completion_rate"] for r in val_results))
    is_viable = selector.is_viable(val_results)
    val_records = [
        {k: r.get(k) for k in ("seed", "reward", "completion_rate", "validation_viable", "validation_rejection_reasons")}
        for r in val_results
    ]
    print(f"    --> [Eval] Ep {ep} | Rew: {avg_rew:.1f} | Comp: {avg_comp:.1%} | Viable: {is_viable}")

    best_path = None
    if selector.consider(ep, val_results):
        best_path = os.path.join(cfg.model_dir, f"q_learning_{cfg.scenario_name}_best.json")
        agent.save(best_path)
        print(f"    --> [BEST] Saved to {best_path}")

    return val_records, best_path


def train_q_learning(config: Optional[TrainingConfig] = None, **options: Any) -> Dict[str, Any]:
    """Train from a config or keyword options."""
    if config is not None and options:
        raise TypeError("Pass config or training keyword options, not both")
    if config is None:
        config_path = options.pop("config_path", None)
        max_steps = options.pop("max_steps_per_episode", None)
        cleaned_options = {k: v for k, v in options.items() if v is not None or k in ("max_green", "max_red", "max_steps")}
        if max_steps is not None:
            cleaned_options["max_steps"] = max_steps
        config = TrainingConfig.from_yaml(config_path, **cleaned_options) if config_path else TrainingConfig(**cleaned_options)
    cfg = config

    os.makedirs(cfg.model_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)

    reward_fn = _create_reward_fn(cfg)
    state_discretizer = StateDiscretizer(
        min_green=cfg.min_green,
        max_effective_green=cfg.max_effective_green,
        serving_queue_thresholds=cfg.serving_thresholds,
        waiting_queue_thresholds=cfg.waiting_thresholds,
        group_by=cfg.group_by,
    )

    print(f"\n{'='*56}")
    print(f"Starting Q-Learning Training on [{cfg.scenario_name.upper()}]")
    print(f"Episodes: {cfg.num_episodes} | LR: {cfg.learning_rate} | Gamma: {cfg.gamma} | Decay: {cfg.epsilon_decay}")
    print(f"{'='*56}")

    env_output_dir = os.path.join(cfg.log_dir, f"env_{cfg.scenario_name}") if cfg.save_run_details else None
    timing_options = {k: getattr(cfg, k) for k in (*DEFAULT_SAFETY_CONFIG, "decision_interval", "max_red")}
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
            ep_start = time.time()
            ep_reward, ep_steps = _run_training_episode(env, agent, cfg.seed + ep)
            ep_duration = time.time() - ep_start
            curr_eps = agent.epsilon
            agent.decay_epsilon()

            summary = env.get_summary() or {}
            if summary.get("is_valid") is False or summary.get("measurement_errors", 0):
                raise ValueError(f"Invalid training measurements: {summary.get('measurement_error_messages', [])}")

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

            if cfg.val_interval > 0 and (ep % cfg.val_interval == 0 or ep == cfg.num_episodes):
                val_records, new_best = _evaluate_checkpoint(
                    agent, cfg, ep, timing_options, state_discretizer, selector
                )
                history[-1]["validation"] = val_records
                if new_best:
                    best_path = new_best

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
        "training_config": asdict(cfg),
        "hyperparameters": {
            k: getattr(cfg, k)
            for k in [
                "learning_rate", "gamma", "epsilon_start", "epsilon_min",
                "epsilon_decay", "decision_interval", "min_green", "max_green", "max_red",
            ]
        },
        "total_wall_clock_seconds": round(total_wall_time, 2),
        "history": history,
        "final_q_table_size": len(agent.q_table),
        "best_reward": round(selector.best_reward, 2) if selector.episode is not None else None,
        "best_is_viable": selector.episode is not None,
        "best_episode": selector.episode,
        "best_checkpoint_path": best_path,
        "final_checkpoint_path": final_model_path,
    }

    write_json_atomic(log_file, training_data)
    print(f"\nTraining completed in {total_wall_time:.2f}s.")
    print(f"Final Model saved to: {final_model_path}")
    print(f"Training Log saved to: {log_file}")
    return training_data


def _build_parser() -> argparse.ArgumentParser:
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
    parser.add_argument("--max-red", type=lambda value: None if value.lower() == "none" else float(value))
    parser.add_argument("--group-by", dest="group_by", choices=["aggregate", "branch", "phase", "phase_group"])
    parser.add_argument("--config", dest="config_path", default="configs/config.yaml", help="Path to YAML configuration")
    parser.add_argument("--val-seeds", type=int, nargs="+")
    parser.add_argument("--save-details", dest="save_run_details", action="store_true")
    return parser


if __name__ == "__main__":
    train_q_learning(**vars(_build_parser().parse_args()))
