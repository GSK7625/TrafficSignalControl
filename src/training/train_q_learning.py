"""Training pipeline for Tabular Q-Learning on SUMO traffic networks."""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.controllers.q_learning import QLearningController
from src.rl.environment import TrafficSignalRLEnv


def train_q_learning(
    scenario_name: str = "ingolstadt1",
    num_episodes: int = 15,
    seed: int = 42,
    learning_rate: float = 0.1,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.85,
    decision_interval: float = 5.0,
    min_green: float = 5.0,
    max_steps_per_episode: Optional[int] = 3600,
    model_dir: str = "models/q_learning",
    log_dir: str = "results/training",
) -> Dict[str, Any]:
    """Trains a Tabular Q-Learning agent on a specified SUMO scenario.

    Args:
        scenario_name: Scenario name ('cologne1' or 'ingolstadt1').
        num_episodes: Number of training episodes.
        seed: Base random seed.
        learning_rate: Q-learning update rate alpha.
        gamma: Discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_min: Minimum exploration rate.
        epsilon_decay: Per-episode decay factor for epsilon.
        decision_interval: Seconds between control decisions.
        min_green: Minimum required green duration.
        max_steps_per_episode: Maximum simulation seconds per episode.
        model_dir: Directory to save model checkpoints.
        log_dir: Directory to save training progression logs.

    Returns:
        Dictionary containing training history and final evaluation summary.
    """
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n========================================================")
    print(f"Starting Q-Learning Training on [{scenario_name.upper()}]")
    print(f"Episodes: {num_episodes} | LR: {learning_rate} | Gamma: {gamma} | Decay: {epsilon_decay}")
    print(f"Decision Interval: {decision_interval}s | Seed: {seed}")
    print(f"========================================================")

    # Initialize environment to query intersection parameters
    env = TrafficSignalRLEnv(
        scenario_name=scenario_name,
        decision_interval=decision_interval,
        min_green=min_green,
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

            # Save checkpoint if best episode reward or last episode
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
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--decay", type=float, default=0.85)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train_q_learning(
        scenario_name=args.dataset,
        num_episodes=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.decay,
        max_steps_per_episode=args.steps,
        seed=args.seed,
    )
