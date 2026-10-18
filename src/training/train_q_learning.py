"""Training pipeline for Tabular Q-Learning on SUMO traffic networks."""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.evaluation.eval_audit import snapshot_agent, finish_eval_audit
from src.config import DEFAULT_SAFETY_CONFIG, load_yaml_config
from src.controllers.q_learning import QLearningController
from src.rl.discretization import StateDiscretizer
from src.rl.environment import TrafficSignalRLEnv
from src.rl.rewards import BaseRewardFunction, NegativeAverageQueueReward


def is_validation_viable(result: dict, min_completion: float = 0.95) -> tuple[bool, list[str]]:
    """Fail closed on missing schema-4 evidence; never infer scheduled demand."""
    import math
    s = result.get("summary", result)
    reasons = []
    required = {"metric_schema_version": "4.0.0", "is_valid": True,
                "data_validity": "VALID", "accounting_valid": True,
                "tripinfo_reconciliation_status": "MATCH", "measurement_errors": 0,
                "conservation_error": 0, "collision_count": 0, "teleport_count": 0}
    for field, expected in required.items():
        if field not in s or s[field] != expected:
            reasons.append(f"{field}: required {expected!r}, got {s.get(field)!r}")
    demand = s.get("scheduled_demand_in_window")
    completed = s.get("total_completed_vehicles")
    def number(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    if not number(demand) or demand < 0:
        reasons.append("UNKNOWN_SCHEDULED_DEMAND")
    elif demand == 0:
        reasons.append("ZERO_DEMAND: diagnostic only")
    elif not number(completed) or not 0 <= completed <= demand:
        reasons.append("total_completed_vehicles: missing or outside scheduled demand")
    elif completed / demand < min_completion:
        reasons.append(f"completion below {min_completion}: {completed / demand}")
    audit = result.get("eval_audit")
    if audit is not None and not audit.get("invariance_pass", False):
        reasons.append("EVALUATION_MUTATED_AGENT")
    return not reasons, reasons


def validation_data_errors(result: dict) -> list[str]:
    """Separate data evidence from the service threshold and zero-demand diagnostic."""
    _, reasons = is_validation_viable(result, min_completion=0.0)
    import math
    s = result.get("summary", result)
    data_reasons = []
    for reason in reasons:
        if reason.startswith("ZERO_DEMAND:"):
            continue
        incident = next((field for field in ("collision_count", "teleport_count") if reason.startswith(field + ":")), None)
        if incident:
            count = s.get(incident)
            if isinstance(count, (int, float)) and math.isfinite(count) and count >= 0:
                continue  # Real safety events do not by themselves corrupt measurements.
        data_reasons.append(reason)
    return data_reasons


class ValidationCheckpointSelector:
    """Selection uses validation only: reward, completion, then earliest episode."""
    def __init__(self, min_completion=0.95):
        self.min_completion = min_completion
        self.episode = None
        self.key = None

    def consider(self, episode, results):
        import math
        if not results or not all(is_validation_viable(r, self.min_completion)[0] for r in results):
            return False
        rewards = [r.get("reward") for r in results]
        if any(not isinstance(r, (int, float)) or not math.isfinite(r) for r in rewards):
            return False
        comp = [r["summary"]["total_completed_vehicles"] / r["summary"]["scheduled_demand_in_window"] for r in results]
        key = (float(np.mean(rewards)), float(np.mean(comp)), -episode)
        if self.key is None or key > self.key:
            self.key, self.episode = key, episode
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
    output_dir: str = "results/rl",
) -> Dict[str, Any]:
    """Evaluates the Q-Learning agent deterministically (epsilon=0, eval_mode=True) with RNG state isolation.

    Returns:
        Dict containing reward, completed_vehicles, completion_rate, collisions, teleports,
        mean_waiting_time, system_mean_delay, and raw summary.
    """
    import random
    prev_py_state = random.getstate()
    prev_np_state = np.random.get_state()

    prev_mode = agent.eval_mode
    prev_eps = agent.epsilon
    agent.eval_mode = True
    agent.epsilon = 0.0

    val_env = None
    total_reward = 0.0
    try:
        before = snapshot_agent(agent)
        random.seed(seed)
        np.random.seed(seed)
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
        )
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
        try:
            if val_env is not None:
                val_env.close()
        finally:
            try:
                if val_env is not None:
                    eval_audit = finish_eval_audit(agent, before)
            finally:
                agent.eval_mode = prev_mode
                agent.epsilon = prev_eps
                random.setstate(prev_py_state)
                np.random.set_state(prev_np_state)

    summary = val_env.get_summary() or {}
    completed = summary.get("total_completed_vehicles", summary.get("tripinfo_completed_vehicles", 0))
    total_demand = summary.get("scheduled_demand_in_window")
    if total_demand is None:
        total_demand = summary.get("total_scenario_demand")
    if total_demand is None:
        run_end = summary.get("running_vehicles_at_end", 0) or 0
        pend_end = summary.get("pending_insertion_vehicles", 0) or 0
        calc_demand = completed + run_end + pend_end
        if calc_demand > 0:
            total_demand = calc_demand
    completion_rate = (completed / total_demand) if isinstance(total_demand, (int, float)) and total_demand > 0 else 0.0
    collisions = summary.get("collision_count", summary.get("collision_incidents", 0))
    teleports = summary.get("teleport_count", 0)
    mean_wait = summary.get("mean_waiting_time", 0.0)
    sys_delay = summary.get("system_mean_delay", 0.0)

    viable, reasons = is_validation_viable({"summary": summary, "eval_audit": eval_audit})
    return {
        "validation_protocol_version": "stage04-schema4-v1",
        "validation_viable": viable,
        "validation_rejection_reasons": reasons,
        "completion_status": "UNKNOWN_DEMAND" if total_demand is None else ("ZERO_DEMAND" if total_demand == 0 else "MEASURED"),
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
        "eval_audit": eval_audit,
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
) -> Dict[str, Any]:
    """Trains a Tabular Q-Learning agent on a specified SUMO scenario."""
    # 1. Base defaults (lowest priority)
    num_episodes_val = 15
    seed_val = 42
    learning_rate_val = 0.1
    gamma_val = 0.95
    epsilon_start_val = 1.0
    epsilon_min_val = 0.05
    epsilon_decay_val = 0.85
    decision_interval_val = 5.0
    min_green_val = DEFAULT_SAFETY_CONFIG["min_green"]
    max_green_val = DEFAULT_SAFETY_CONFIG["max_green"]
    yellow_dur = DEFAULT_SAFETY_CONFIG["yellow_duration"]
    all_red_dur = DEFAULT_SAFETY_CONFIG["all_red_duration"]
    max_steps_val = 3600
    val_interval_val = 5
    reward_scale = 1.0
    pending_weight_val = 0.0
    max_effective_green = 25.0
    serving_thresholds = (1.0, 5.0)
    waiting_thresholds = (1.0, 6.0, 15.0)

    # 2. YAML config (medium priority - overrides base defaults)
    if config_path:
        cfg = load_yaml_config(config_path)
        hp = cfg.get("hyperparameters", {})
        if "learning_rate" in hp:
            learning_rate_val = float(hp["learning_rate"])
        if "gamma" in hp:
            gamma_val = float(hp["gamma"])
        if "epsilon_start" in hp:
            epsilon_start_val = float(hp["epsilon_start"])
        if "epsilon_min" in hp:
            epsilon_min_val = float(hp["epsilon_min"])
        if "epsilon_decay" in hp:
            epsilon_decay_val = float(hp["epsilon_decay"])
        if "num_episodes" in hp:
            num_episodes_val = int(hp["num_episodes"])
        if "seed" in hp:
            seed_val = int(hp["seed"])
        if "decision_interval" in cfg:
            decision_interval_val = float(cfg["decision_interval"])
        if "min_green" in cfg:
            min_green_val = float(cfg["min_green"])
        if "max_green" in cfg:
            max_green_val = float(cfg["max_green"])
        elif "max_green" in hp:
            max_green_val = float(hp["max_green"])
        if "yellow_duration" in cfg:
            yellow_dur = float(cfg["yellow_duration"])
        if "all_red_duration" in cfg:
            all_red_dur = float(cfg["all_red_duration"])
        if "max_steps" in cfg:
            max_steps_val = int(cfg["max_steps"])
        if "val_interval" in cfg:
            val_interval_val = int(cfg["val_interval"])
        if "val_seeds" in cfg:
            val_seeds_cfg = [int(s) for s in cfg["val_seeds"]]
        elif "val_seeds" in hp:
            val_seeds_cfg = [int(s) for s in hp["val_seeds"]]
        else:
            val_seeds_cfg = None

        reward_cfg = cfg.get("reward", {})
        if "scale" in reward_cfg or "reward_scale" in cfg:
            reward_scale = float(reward_cfg.get("scale", cfg.get("reward_scale", reward_scale)))
        if "pending_weight" in reward_cfg:
            pending_weight_val = float(reward_cfg["pending_weight"])

        disc_cfg = cfg.get("discretization", {})
        if "max_effective_green" in disc_cfg:
            max_effective_green = float(disc_cfg["max_effective_green"])
        if "serving_queue_thresholds" in disc_cfg:
            serving_thresholds = tuple(float(x) for x in disc_cfg["serving_queue_thresholds"])
        if "waiting_queue_thresholds" in disc_cfg:
            waiting_thresholds = tuple(float(x) for x in disc_cfg["waiting_queue_thresholds"])

    # 3. Explicit arguments / CLI (highest priority - overrides YAML)
    if num_episodes is not None:
        num_episodes_val = int(num_episodes)
    if seed is not None:
        seed_val = int(seed)
    if learning_rate is not None:
        learning_rate_val = float(learning_rate)
    if gamma is not None:
        gamma_val = float(gamma)
    if epsilon_start is not None:
        epsilon_start_val = float(epsilon_start)
    if epsilon_min is not None:
        epsilon_min_val = float(epsilon_min)
    if epsilon_decay is not None:
        epsilon_decay_val = float(epsilon_decay)
    if decision_interval is not None:
        decision_interval_val = float(decision_interval)
    if min_green is not None:
        min_green_val = float(min_green)
    if max_green is not None:
        max_green_val = float(max_green)
    if yellow_duration is not None:
        yellow_dur = float(yellow_duration)
    if all_red_duration is not None:
        all_red_dur = float(all_red_duration)
    if max_steps_per_episode is not None:
        max_steps_val = int(max_steps_per_episode)
    if val_interval is not None:
        val_interval_val = int(val_interval)
    if val_seeds is None and "val_seeds_cfg" in locals() and val_seeds_cfg is not None:
        val_seeds = val_seeds_cfg

    # Aliases for downstream execution
    num_episodes = num_episodes_val
    seed = seed_val
    learning_rate = learning_rate_val
    gamma = gamma_val
    epsilon_start = epsilon_start_val
    epsilon_min = epsilon_min_val
    epsilon_decay = epsilon_decay_val
    decision_interval = decision_interval_val
    min_green = min_green_val
    max_green = max_green_val
    max_steps_per_episode = max_steps_val
    val_interval = val_interval_val
    pending_weight = pending_weight_val

    reward_fn = NegativeAverageQueueReward(scale=reward_scale, pending_weight=pending_weight)
    state_discretizer = StateDiscretizer(
        min_green=min_green,
        max_effective_green=max_effective_green,
        serving_queue_thresholds=serving_thresholds,
        waiting_queue_thresholds=waiting_thresholds,
    )

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n========================================================")
    print(f"Starting Q-Learning Training on [{scenario_name.upper()}]")
    print(f"Episodes: {num_episodes} | LR: {learning_rate} | Gamma: {gamma} | Decay: {epsilon_decay}")
    print(f"Decision Interval: {decision_interval}s | Seed: {seed} | Yellow: {yellow_dur}s | All-Red: {all_red_dur}s | Max Green: {max_green}s")
    print(f"========================================================")

    # Initialize environment to query intersection parameters
    env = TrafficSignalRLEnv(
        scenario_name=scenario_name,
        decision_interval=decision_interval,
        min_green=min_green,
        max_green=max_green,
        yellow_duration=yellow_dur,
        all_red_duration=all_red_dur,
        reward_fn=reward_fn,
        state_discretizer=state_discretizer,
        max_steps=max_steps_per_episode,
        seed=seed,
        output_dir=os.path.join(log_dir, f"env_{scenario_name}"),
        use_tripinfo=True,
    )

    agent = QLearningController(
        num_actions=env.num_actions,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        discretizer=state_discretizer,
        decision_interval=decision_interval,
        min_green=min_green,
        max_green=max_green,
        yellow_duration=yellow_dur,
        all_red_duration=all_red_dur,
        seed=seed,
        eval_mode=False,
    )

    history: List[Dict[str, Any]] = []
    selector = ValidationCheckpointSelector(min_val_completion)
    best_reward = -float("inf")
    best_is_viable = False
    best_path: Optional[str] = None
    validation_history: List[Dict[str, Any]] = []
    start_total_time = time.time()

    try:
        for ep in range(1, num_episodes + 1):
            ep_seed = seed + ep
            ep_start_time = time.time()

            _, info = env.reset(seed=ep_seed)
            discrete_state = info["discrete_state"]
            valid_actions = info["valid_actions"]

            ep_reward = 0.0
            ep_steps = 0
            ep_td_errors = []

            while True:
                action = agent.select_action_for_state(
                    discrete_state,
                    valid_actions=valid_actions,
                    current_phase=discrete_state[0],
                )

                _, reward, terminated, truncated, next_info = env.step(action)
                next_state = next_info["discrete_state"]
                next_valid = next_info["valid_actions"]

                done = bool(terminated or truncated)
                td_err = agent.update(
                    state=discrete_state,
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
                ep_td_errors.append(td_err)

                discrete_state = next_state
                valid_actions = next_valid

                if done:
                    break

            ep_duration = time.time() - ep_start_time
            curr_eps = agent.epsilon
            agent.decay_epsilon()

            summary = env.get_summary() or {}
            mean_wait = summary.get("mean_waiting_time", 0.0)
            mean_queue = summary.get("mean_queue_length", 0.0)
            completed_vehs = summary.get("total_completed_vehicles", 0)
            running_vehs = summary.get("running_vehicles_at_end", 0)
            pending_vehs = summary.get("pending_insertion_vehicles", 0)
            coll_count = summary.get("collision_count", summary.get("collision_incidents", 0))
            tele_count = summary.get("teleport_count", 0)
            meas_errors = summary.get("measurement_errors", 0)
            avg_td = float(np.mean(ep_td_errors)) if ep_td_errors else 0.0

            ep_record = {
                "episode": ep,
                "seed": ep_seed,
                "reward": round(ep_reward, 2),
                "steps": ep_steps,
                "epsilon": round(curr_eps, 4),
                "q_table_size": len(agent.q_table),
                "total_updates": agent.total_updates,
                "mean_td_error": round(avg_td, 4),
                "mean_waiting_time": round(mean_wait, 2) if mean_wait is not None else None,
                "mean_queue_length": round(mean_queue, 2),
                "completed_vehicles": completed_vehs,
                "running_vehicles_at_end": running_vehs,
                "pending_insertion_vehicles": pending_vehs,
                "collision_count": coll_count,
                "teleport_count": tele_count,
                "measurement_errors": meas_errors,
                "action_counts": agent.action_counts.tolist(),
                "duration_seconds": round(ep_duration, 2),
            }
            history.append(ep_record)

            print(
                f"Ep {ep:2d}/{num_episodes} | Rew: {ep_reward:8.1f} | Eps: {curr_eps:.3f} | "
                f"Q-States: {len(agent.q_table):3d} | Wait: {mean_wait:5.2f}s | "
                f"Queue: {mean_queue:4.2f} | Completed: {completed_vehs:4d} | "
                f"Time: {ep_duration:4.1f}s"
            )

            # Save checkpoint based on greedy evaluation or periodic interval
            if val_interval > 0 and (ep % val_interval == 0 or ep == num_episodes):
                target_val_seeds = list(val_seeds) if val_seeds is not None else [seed + 9999, seed + 10000]
                val_results = []
                for v_seed in target_val_seeds:
                    res = evaluate_greedy_q_learning(
                        agent=agent,
                        scenario_name=scenario_name,
                        decision_interval=decision_interval,
                        min_green=min_green,
                        max_green=max_green,
                        yellow_duration=yellow_dur,
                        all_red_duration=all_red_dur,
                        reward_fn=NegativeAverageQueueReward(scale=reward_scale, pending_weight=pending_weight),
                        state_discretizer=state_discretizer,
                        max_steps=max_steps_per_episode,
                        seed=v_seed,
                        output_dir=os.path.join(log_dir, "validation", f"ep{ep}_seed{v_seed}"),
                    )
                    val_results.append(res)

                avg_val_reward = float(np.mean([r["reward"] for r in val_results]))
                min_val_comp = float(min(r["completion_rate"] for r in val_results))
                avg_val_comp = float(np.mean([r["completion_rate"] for r in val_results]))
                total_val_colls = int(sum(r["collisions"] for r in val_results))
                total_val_teleports = int(sum(r["teleports"] for r in val_results))
                avg_sys_delay = float(np.mean([r["system_mean_delay"] or 0.0 for r in val_results]))

                checks = [is_validation_viable(r, min_val_completion) for r in val_results]
                is_viable = bool(checks) and all(ok for ok, _ in checks)

                print(
                    f"    --> [Greedy Eval] Ep {ep} | Seeds: {target_val_seeds} | "
                    f"Mean Rew: {avg_val_reward:8.1f} | Min Comp: {min_val_comp:6.1%} | "
                    f"Colls: {total_val_colls} | Teleports: {total_val_teleports} | "
                    f"Delay: {avg_sys_delay:5.2f}s | Viable: {is_viable}"
                )

                val_record = {
                    "episode": ep,
                    "target_val_seeds": target_val_seeds,
                    "mean_reward": round(avg_val_reward, 2),
                    "min_completion_rate": round(min_val_comp, 4),
                    "mean_completion_rate": round(avg_val_comp, 4),
                    "total_collisions": total_val_colls,
                    "total_teleports": total_val_teleports,
                    "mean_system_delay": round(avg_sys_delay, 2),
                    "is_viable": is_viable,
                    "rejection_reasons": [],
                    "seed_results": [
                        {
                            "summary": r["summary"],
                            "eval_audit": r.get("eval_audit"),
                            "seed": r.get("seed", v_s),
                            "reward": r["reward"],
                            "completed_vehicles": r["completed_vehicles"],
                            "total_demand": r["total_demand"],
                            "completion_rate": r["completion_rate"],
                            "collisions": r["collisions"],
                            "teleports": r["teleports"],
                            "mean_waiting_time": r["mean_waiting_time"],
                            "system_mean_delay": r["system_mean_delay"],
                        }
                        for v_s, r in zip(target_val_seeds, val_results)
                    ],
                }

                if is_viable:
                    is_better = selector.consider(ep, val_results)
                    if is_better:
                        best_reward = avg_val_reward
                        best_is_viable = True
                        best_path = os.path.join(model_dir, f"q_learning_{scenario_name}_best.json")
                        agent.save(best_path)
                        print(f"    --> [NEW BEST CHECKPOINT] Saved to {best_path} (Rew: {avg_val_reward:.1f}, Comp: {avg_val_comp:.1%})")
                else:
                    reasons = [reason for _, rs in checks for reason in rs]
                    if total_val_colls > 0:
                        reasons.append(f"{total_val_colls} collisions")
                    if total_val_teleports > 0:
                        reasons.append(f"{total_val_teleports} teleports")
                    if min_val_comp < min_val_completion:
                        reasons.append(f"low completion {min_val_comp:.1%} < {min_val_completion:.1%}")
                    val_record["rejection_reasons"] = reasons
                    print(f"    --> [CHECKPOINT REJECTED] Ep {ep} policy is degenerate ({', '.join(reasons)}). Not saved as best checkpoint.")
                validation_history.append(val_record)
            # With validation disabled, checkpoints remain diagnostic only.

            if ep % 5 == 0 or ep == num_episodes:
                ckpt_path = os.path.join(
                    model_dir, f"q_learning_{scenario_name}_ep{ep}.json"
                )
                agent.save(ckpt_path)

    finally:
        env.close()

    total_wall_time = time.time() - start_total_time

    # Save final model
    final_model_path = os.path.join(model_dir, f"q_learning_{scenario_name}_final.json")
    agent.save(final_model_path)

    # Save training log
    log_file = os.path.join(log_dir, f"q_learning_{scenario_name}_log.json")
    training_data = {
        "scenario_name": scenario_name,
        "algorithm": "Q-Learning",
        "num_episodes": num_episodes,
        "seed": seed,
        "hyperparameters": {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "decision_interval": decision_interval,
            "min_green": min_green,
            "max_green": max_green,
            "pending_weight": pending_weight,
            "min_val_completion": min_val_completion,
        },
        "total_wall_clock_seconds": round(total_wall_time, 2),
        "history": history,
        "final_q_table_size": len(agent.q_table),
        "best_reward": round(best_reward, 2) if best_is_viable else None,
        "best_is_viable": best_is_viable,
        "validation_protocol_version": "stage04-schema4-v1",
        "validation_history": validation_history,
        "best_checkpoint_path": best_path if best_is_viable else None,
        "best_checkpoint_status": "BEST_VIABLE" if best_is_viable else "NO_VIABLE_CHECKPOINT",
        "final_checkpoint_path": final_model_path,
        "selected_checkpoint_path": best_path if best_is_viable else None,
        "selected_checkpoint_status": "BEST_VIABLE" if best_is_viable else "NO_VIABLE_CHECKPOINT",
        "selected_checkpoint_reason": (
            f"Viable policy with validation reward {best_reward:.2f}"
            if best_is_viable
            else "NO_VIABLE_CHECKPOINT: No validation checkpoint met viability criteria; final checkpoint is diagnostic only; confirmatory evaluation disabled"
        ),
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2)

    print(f"\nTraining completed in {total_wall_time:.2f}s.")
    print(f"Final Model saved to: {final_model_path}")
    print(f"Training Log saved to: {log_file}")

    return training_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Tabular Q-Learning on SUMO.")
    parser.add_argument("--dataset", type=str, default="ingolstadt1", choices=["cologne1", "ingolstadt1"])
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
    )
