"""Training pipeline for Tabular Q-Learning on SUMO traffic networks."""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.config import load_yaml_config
from src.controllers.q_learning import QLearningController
from src.rl.discretization import StateDiscretizer
from src.rl.environment import TrafficSignalRLEnv
from src.rl.rewards import BaseRewardFunction, NegativeAverageQueueReward


def evaluate_greedy_q_learning(
    agent: QLearningController,
    scenario_name: str,
    decision_interval: float = 5.0,
    min_green: float = 5.0,
    yellow_duration: float = 3.0,
    all_red_duration: float = 0.0,
    reward_fn: Optional[BaseRewardFunction] = None,
    state_discretizer: Optional[StateDiscretizer] = None,
    max_steps: Optional[int] = 3600,
    seed: int = 42,
) -> float:
    """Evaluates the Q-Learning agent deterministically (epsilon=0, eval_mode=True) with RNG state isolation."""
    import random
    prev_py_state = random.getstate()
    prev_np_state = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)

    val_env = TrafficSignalRLEnv(
        scenario_name=scenario_name,
        decision_interval=decision_interval,
        min_green=min_green,
        yellow_duration=yellow_duration,
        all_red_duration=all_red_duration,
        reward_fn=reward_fn,
        state_discretizer=state_discretizer,
        max_steps=max_steps,
        seed=seed,
        gui=False,
    )
    prev_mode = agent.eval_mode
    prev_eps = agent.epsilon
    agent.eval_mode = True
    agent.epsilon = 0.0

    total_reward = 0.0
    try:
        obs, info = val_env.reset(seed=seed)
        discrete_state = info["discrete_state"]
        valid_actions = info["valid_actions"]
        while True:
            action = agent.select_action_for_state(discrete_state, valid_actions=valid_actions)
            next_obs, reward, terminated, truncated, next_info = val_env.step(action)
            total_reward += reward
            discrete_state = next_info["discrete_state"]
            valid_actions = next_info["valid_actions"]
            if terminated or truncated:
                break
    finally:
        val_env.close()
        agent.eval_mode = prev_mode
        agent.epsilon = prev_eps
        random.setstate(prev_py_state)
        np.random.set_state(prev_np_state)

    return total_reward


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
    yellow_duration: Optional[float] = None,
    all_red_duration: Optional[float] = None,
    max_steps_per_episode: Optional[int] = None,
    val_interval: Optional[int] = None,
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
    min_green_val = 5.0
    yellow_dur = 3.0
    all_red_dur = 0.0
    max_steps_val = 3600
    val_interval_val = 5
    reward_scale = 1.0
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
        if "yellow_duration" in cfg:
            yellow_dur = float(cfg["yellow_duration"])
        if "all_red_duration" in cfg:
            all_red_dur = float(cfg["all_red_duration"])
        if "max_steps" in cfg:
            max_steps_val = int(cfg["max_steps"])
        if "val_interval" in cfg:
            val_interval_val = int(cfg["val_interval"])

        reward_cfg = cfg.get("reward", {})
        if "scale" in reward_cfg or "reward_scale" in cfg:
            reward_scale = float(reward_cfg.get("scale", cfg.get("reward_scale", reward_scale)))

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
    if yellow_duration is not None:
        yellow_dur = float(yellow_duration)
    if all_red_duration is not None:
        all_red_dur = float(all_red_duration)
    if max_steps_per_episode is not None:
        max_steps_val = int(max_steps_per_episode)
    if val_interval is not None:
        val_interval_val = int(val_interval)

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
    max_steps_per_episode = max_steps_val
    val_interval = val_interval_val

    reward_fn = NegativeAverageQueueReward(scale=reward_scale)
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
    print(f"Decision Interval: {decision_interval}s | Seed: {seed} | Yellow: {yellow_dur}s | All-Red: {all_red_dur}s")
    print(f"========================================================")

    # Initialize environment to query intersection parameters
    env = TrafficSignalRLEnv(
        scenario_name=scenario_name,
        decision_interval=decision_interval,
        min_green=min_green,
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
        yellow_duration=yellow_dur,
        all_red_duration=all_red_dur,
        seed=seed,
        eval_mode=False,
    )

    history: List[Dict[str, Any]] = []
    best_reward = -float("inf")
    start_total_time = time.time()

    try:
        for ep in range(1, num_episodes + 1):
            ep_seed = seed + ep
            ep_start_time = time.time()

            obs_vec, info = env.reset(seed=ep_seed)
            discrete_state = info["discrete_state"]
            valid_actions = info["valid_actions"]

            ep_reward = 0.0
            ep_steps = 0
            ep_td_errors = []

            while True:
                action = agent.select_action_for_state(
                    discrete_state, valid_actions=valid_actions
                )

                next_obs, reward, terminated, truncated, next_info = env.step(action)
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
            avg_td = float(np.mean(ep_td_errors)) if ep_td_errors else 0.0

            ep_record = {
                "episode": ep,
                "reward": round(ep_reward, 2),
                "steps": ep_steps,
                "epsilon": round(curr_eps, 4),
                "q_table_size": len(agent.q_table),
                "total_updates": agent.total_updates,
                "mean_td_error": round(avg_td, 4),
                "mean_waiting_time": round(mean_wait, 2),
                "mean_queue_length": round(mean_queue, 2),
                "completed_vehicles": completed_vehs,
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
                val_reward = evaluate_greedy_q_learning(
                    agent=agent,
                    scenario_name=scenario_name,
                    decision_interval=decision_interval,
                    min_green=min_green,
                    yellow_duration=yellow_dur,
                    all_red_duration=all_red_dur,
                    reward_fn=NegativeAverageQueueReward(scale=reward_scale),
                    state_discretizer=state_discretizer,
                    max_steps=max_steps_per_episode,
                    seed=seed + 9999,
                )
                print(f"    --> [Greedy Eval] Ep {ep} Validation Reward: {val_reward:8.1f} (Best: {best_reward:8.1f})")
                if val_reward > best_reward:
                    best_reward = val_reward
                    best_path = os.path.join(model_dir, f"q_learning_{scenario_name}_best.json")
                    agent.save(best_path)
                    print(f"    --> [NEW BEST CHECKPOINT] Saved to {best_path}")
            elif val_interval <= 0:
                if ep_reward > best_reward:
                    best_reward = ep_reward
                    best_path = os.path.join(model_dir, f"q_learning_{scenario_name}_best.json")
                    agent.save(best_path)

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
        },
        "total_wall_clock_seconds": round(total_wall_time, 2),
        "history": history,
        "final_q_table_size": len(agent.q_table),
        "best_reward": round(best_reward, 2),
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
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--val-interval", type=int, default=None, help="Interval of episodes for greedy evaluation")

    args = parser.parse_args()
    train_q_learning(
        scenario_name=args.dataset,
        num_episodes=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.decay,
        max_steps_per_episode=args.steps,
        seed=args.seed,
        config_path=args.config,
        val_interval=args.val_interval,
    )
