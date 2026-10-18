"""Training pipeline for Tabular Q-Learning on SUMO traffic networks."""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.config import DEFAULT_SAFETY_CONFIG, load_yaml_config
from src.controllers.q_learning import QLearningController
from src.rl.discretization import StateDiscretizer
from src.rl.environment import TrafficSignalRLEnv
from src.rl.rewards import BaseRewardFunction, NegativePressureReward


def is_validation_viable(result: dict, min_completion: float = 0.95) -> tuple[bool, list[str]]:
    """Simple viability check: no collisions/teleports and sufficient vehicle completion rate."""
    s = result.get("summary", result)
    reasons = []
    if s.get("collision_count", 0) > 0:
        reasons.append("has_collisions")
    if s.get("teleport_count", 0) > 0:
        reasons.append("has_teleports")

    audit = result.get("eval_audit")
    if audit is not None and not audit.get("invariance_pass", True):
        reasons.append("EVALUATION_MUTATED_AGENT")

    demand = s.get("scheduled_demand_in_window") or s.get("total_scenario_demand") or result.get("total_demand")
    completed = s.get("total_completed_vehicles", result.get("completed_vehicles", 0))
    if demand and demand > 0:
        comp_rate = completed / demand
    else:
        comp_rate = result.get("completion_rate", 1.0)

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
    audit = result.get("eval_audit")
    if audit is not None and not audit.get("invariance_pass", True):
        errors.append("EVALUATION_MUTATED_AGENT")
    return errors


class ValidationCheckpointSelector:
    """Selects best checkpoint based on validation reward and completion."""

    def __init__(self, min_completion: float = 0.95):
        self.min_completion = min_completion
        self.best_reward = -float("inf")
        self.episode = None

    def consider(self, episode: int, results: list) -> bool:
        if not results or not all(is_validation_viable(r, self.min_completion)[0] for r in results):
            return False
        avg_rew = float(np.mean([r.get("reward", 0.0) for r in results]))
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
    """Evaluates the Q-Learning agent deterministically (epsilon=0, eval_mode=True)."""
    prev_mode, prev_eps = agent.eval_mode, agent.epsilon
    agent.eval_mode, agent.epsilon = True, 0.0

    val_env = TrafficSignalRLEnv(
        scenario_name=scenario_name,
        decision_interval=decision_interval,
        min_green=min_green,
        max_green=max_green,
        yellow_duration=yellow_duration,
        all_red_duration=all_red_duration,
        reward_fn=reward_fn,
        state_discretizer=state_discretizer,
        max_steps=max_steps,
        seed=seed,
        gui=False,
        output_dir=output_dir,
        save_run_details=save_run_details,
    )
    total_reward = 0.0
    try:
        _, info = val_env.reset(seed=seed)
        discrete_state = info["discrete_state"]
        valid_actions = info["valid_actions"]
        while True:
            action = agent.select_action_for_state(
                discrete_state, valid_actions=valid_actions, current_phase=discrete_state[0]
            )
            _, reward, terminated, truncated, next_info = val_env.step(action)
            total_reward += reward
            discrete_state = next_info["discrete_state"]
            valid_actions = next_info["valid_actions"]
            if terminated or truncated:
                break
    finally:
        val_env.close()
        agent.eval_mode, agent.epsilon = prev_mode, prev_eps

    summary = val_env.get_summary() or {}
    completed = summary.get("total_completed_vehicles", summary.get("tripinfo_completed_vehicles", 0))
    total_demand = summary.get("scheduled_demand_in_window") or summary.get("total_scenario_demand")
    if total_demand is None:
        run_end = summary.get("running_vehicles_at_end", 0) or 0
        pend_end = summary.get("pending_insertion_vehicles", 0) or 0
        calc_demand = completed + run_end + pend_end
        if calc_demand > 0:
            total_demand = calc_demand

    completion_rate = (completed / total_demand) if total_demand and total_demand > 0 else 0.0
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


def train_q_learning(
    scenario_name: str = "ingolstadt1",
    num_episodes: Optional[int] = None,
    seed: Optional[int] = None,
    learning_rate: Optional[float] = None,
    gamma: Optional[float] = None,
    epsilon_start: Optional[float] = None,
    epsilon_min: Optional[float] = None,
    epsilon_decay: Optional[float] = None,
    decision_interval: Optional[float] = None,
    min_green: Optional[float] = None,
    max_green: Optional[float] = None,
    yellow_duration: Optional[float] = None,
    all_red_duration: Optional[float] = None,
    max_steps_per_episode: Optional[int] = None,
    val_interval: Optional[int] = None,
    val_seeds: Optional[List[int]] = None,
    min_val_completion: float = 0.95,
    config_path: Optional[str] = None,
    model_dir: str = "models/q_learning",
    log_dir: str = "results/training",
    save_run_details: bool = False,
) -> Dict[str, Any]:
    """Trains a Tabular Q-Learning agent on a specified SUMO scenario."""
    # 1. Base defaults
    cfg_data = {
        "num_episodes": 15,
        "seed": 42,
        "learning_rate": 0.1,
        "gamma": 0.95,
        "epsilon_start": 1.0,
        "epsilon_min": 0.05,
        "epsilon_decay": 0.85,
        "decision_interval": 5.0,
        "min_green": DEFAULT_SAFETY_CONFIG["min_green"],
        "max_green": DEFAULT_SAFETY_CONFIG["max_green"],
        "yellow_duration": DEFAULT_SAFETY_CONFIG["yellow_duration"],
        "all_red_duration": DEFAULT_SAFETY_CONFIG["all_red_duration"],
        "max_steps": 3600,
        "val_interval": 5,
        "val_seeds": None,
        "reward_scale": 1.0,
        "pending_weight": 0.0,
        "max_effective_green": 25.0,
        "serving_thresholds": (1.0, 5.0),
        "waiting_thresholds": (1.0, 6.0, 15.0),
    }

    # 2. YAML overrides
    if config_path:
        yaml_cfg = load_yaml_config(config_path)
        hp = yaml_cfg.get("hyperparameters", {})
        for k in ["learning_rate", "gamma", "epsilon_start", "epsilon_min", "epsilon_decay", "num_episodes", "seed", "max_green", "val_seeds"]:
            if k in hp:
                cfg_data[k] = hp[k]
        for k in ["decision_interval", "min_green", "max_green", "yellow_duration", "all_red_duration", "max_steps", "val_interval", "val_seeds"]:
            if k in yaml_cfg:
                cfg_data[k] = yaml_cfg[k]
        rc = yaml_cfg.get("reward", {})
        if "scale" in rc or "reward_scale" in yaml_cfg:
            cfg_data["reward_scale"] = float(rc.get("scale", yaml_cfg.get("reward_scale", cfg_data["reward_scale"])))
        if "pending_weight" in rc:
            cfg_data["pending_weight"] = float(rc["pending_weight"])
        dc = yaml_cfg.get("discretization", {})
        if "max_effective_green" in dc:
            cfg_data["max_effective_green"] = float(dc["max_effective_green"])
        if "serving_queue_thresholds" in dc:
            cfg_data["serving_thresholds"] = tuple(float(x) for x in dc["serving_queue_thresholds"])
        if "waiting_queue_thresholds" in dc:
            cfg_data["waiting_thresholds"] = tuple(float(x) for x in dc["waiting_queue_thresholds"])

    # 3. Explicit arguments override
    overrides = {
        "num_episodes": num_episodes,
        "seed": seed,
        "learning_rate": learning_rate,
        "gamma": gamma,
        "epsilon_start": epsilon_start,
        "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay,
        "decision_interval": decision_interval,
        "min_green": min_green,
        "max_green": max_green,
        "yellow_duration": yellow_duration,
        "all_red_duration": all_red_duration,
        "max_steps": max_steps_per_episode,
        "val_interval": val_interval,
        "val_seeds": val_seeds,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg_data[k] = v

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    reward_fn = NegativePressureReward(scale=cfg_data["reward_scale"], pending_weight=cfg_data["pending_weight"])
    state_discretizer = StateDiscretizer(
        min_green=cfg_data["min_green"],
        max_effective_green=cfg_data["max_effective_green"],
        serving_queue_thresholds=cfg_data["serving_thresholds"],
        waiting_queue_thresholds=cfg_data["waiting_thresholds"],
    )

    print(f"\n{'='*56}")
    print(f"Starting Q-Learning Training on [{scenario_name.upper()}]")
    print(f"Episodes: {cfg_data['num_episodes']} | LR: {cfg_data['learning_rate']} | Gamma: {cfg_data['gamma']} | Decay: {cfg_data['epsilon_decay']}")
    print(f"{'='*56}")

    env_output_dir = os.path.join(log_dir, f"env_{scenario_name}") if save_run_details else None
    env = TrafficSignalRLEnv(
        scenario_name=scenario_name,
        decision_interval=cfg_data["decision_interval"],
        min_green=cfg_data["min_green"],
        max_green=cfg_data["max_green"],
        yellow_duration=cfg_data["yellow_duration"],
        all_red_duration=cfg_data["all_red_duration"],
        reward_fn=reward_fn,
        state_discretizer=state_discretizer,
        max_steps=cfg_data["max_steps"],
        seed=cfg_data["seed"],
        output_dir=env_output_dir,
        save_run_details=save_run_details,
        use_tripinfo=True,
    )

    agent = QLearningController(
        num_actions=env.num_actions,
        learning_rate=cfg_data["learning_rate"],
        gamma=cfg_data["gamma"],
        epsilon=cfg_data["epsilon_start"],
        epsilon_min=cfg_data["epsilon_min"],
        epsilon_decay=cfg_data["epsilon_decay"],
        discretizer=state_discretizer,
        decision_interval=cfg_data["decision_interval"],
        min_green=cfg_data["min_green"],
        max_green=cfg_data["max_green"],
        yellow_duration=cfg_data["yellow_duration"],
        all_red_duration=cfg_data["all_red_duration"],
        seed=cfg_data["seed"],
        eval_mode=False,
    )

    history: List[Dict[str, Any]] = []
    best_reward = -float("inf")
    best_is_viable = False
    best_path: Optional[str] = None
    start_time = time.time()

    try:
        for ep in range(1, cfg_data["num_episodes"] + 1):
            ep_seed = cfg_data["seed"] + ep
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
                f"Ep {ep:2d}/{cfg_data['num_episodes']} | Rew: {ep_reward:8.1f} | Eps: {curr_eps:.3f} | "
                f"Q-States: {len(agent.q_table):3d} | Wait: {wait_str} | "
                f"Queue: {mean_queue:4.2f} | Completed: {completed_vehs:4d} | "
                f"Time: {ep_duration:4.1f}s"
            )

            # Validation checkpointing
            val_int = cfg_data["val_interval"]
            if val_int > 0 and (ep % val_int == 0 or ep == cfg_data["num_episodes"]):
                v_seeds = cfg_data["val_seeds"] or [cfg_data["seed"] + 9999, cfg_data["seed"] + 10000]
                val_output_dir = os.path.join(log_dir, "validation", f"ep{ep}_seed{vs}") if save_run_details else None
                val_results = [
                    evaluate_greedy_q_learning(
                        agent=agent,
                        scenario_name=scenario_name,
                        decision_interval=cfg_data["decision_interval"],
                        min_green=cfg_data["min_green"],
                        max_green=cfg_data["max_green"],
                        yellow_duration=cfg_data["yellow_duration"],
                        all_red_duration=cfg_data["all_red_duration"],
                        reward_fn=NegativePressureReward(scale=cfg_data["reward_scale"], pending_weight=cfg_data["pending_weight"]),
                        state_discretizer=state_discretizer,
                        max_steps=cfg_data["max_steps"],
                        seed=vs,
                        output_dir=val_output_dir,
                        save_run_details=save_run_details,
                    )
                    for vs in v_seeds
                ]

                avg_val_rew = float(np.mean([r["reward"] for r in val_results]))
                avg_val_comp = float(np.mean([r["completion_rate"] for r in val_results]))
                is_viable = all(r.get("validation_viable", True) for r in val_results)

                print(f"    --> [Eval] Ep {ep} | Rew: {avg_val_rew:.1f} | Comp: {avg_val_comp:.1%} | Viable: {is_viable}")

                if is_viable and (not best_is_viable or avg_val_rew > best_reward):
                    best_reward = avg_val_rew
                    best_is_viable = True
                    best_path = os.path.join(model_dir, f"q_learning_{scenario_name}_best.json")
                    agent.save(best_path)
                    print(f"    --> [BEST] Saved to {best_path}")

            if save_run_details and (ep % 5 == 0 or ep == cfg_data["num_episodes"]):
                agent.save(os.path.join(model_dir, f"q_learning_{scenario_name}_ep{ep}.json"))

    finally:
        env.close()

    total_wall_time = time.time() - start_time
    final_model_path = os.path.join(model_dir, f"q_learning_{scenario_name}_final.json")
    agent.save(final_model_path)

    log_file = os.path.join(log_dir, f"q_learning_{scenario_name}_log.json")
    training_data = {
        "scenario_name": scenario_name,
        "algorithm": "Q-Learning",
        "num_episodes": cfg_data["num_episodes"],
        "seed": cfg_data["seed"],
        "hyperparameters": {
            k: cfg_data[k]
            for k in [
                "learning_rate", "gamma", "epsilon_start", "epsilon_min",
                "epsilon_decay", "decision_interval", "min_green", "max_green",
            ]
        },
        "total_wall_clock_seconds": round(total_wall_time, 2),
        "history": history,
        "final_q_table_size": len(agent.q_table),
        "best_reward": round(best_reward, 2) if best_is_viable else None,
        "best_is_viable": best_is_viable,
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
    parser = argparse.ArgumentParser(description="Train Tabular Q-Learning on SUMO.")
    parser.add_argument("--dataset", type=str, default="ingolstadt1", choices=["cologne1", "ingolstadt1", "cologne3"])
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--decay", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-green", type=float, default=None, help="Maximum consecutive green duration")
    parser.add_argument("--min-completion", type=float, default=0.95, help="Minimum validation completion rate for best checkpoint")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--val-interval", type=int, default=None, help="Interval of episodes for greedy evaluation")
    parser.add_argument("--val-seeds", type=int, nargs="+", default=None, help="List of validation seeds")
    parser.add_argument("--save-details", action="store_true", help="Save detailed per-episode simulation files and checkpoints")

    args = parser.parse_args()
    train_q_learning(
        scenario_name=args.dataset,
        num_episodes=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.decay,
        max_steps_per_episode=args.steps,
        seed=args.seed,
        max_green=args.max_green,
        min_val_completion=args.min_completion,
        config_path=args.config,
        val_interval=args.val_interval,
        val_seeds=args.val_seeds,
        save_run_details=args.save_details,
    )
