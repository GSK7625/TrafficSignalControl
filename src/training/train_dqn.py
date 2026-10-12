"""Training pipeline for Deep Q-Network (DQN) on SUMO traffic networks."""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.controllers.dqn import DQNController
from src.rl.environment import TrafficSignalRLEnv


def train_dqn(
    scenario_name: str = "ingolstadt1",
    num_episodes: int = 15,
    seed: int = 42,
    hidden_dim: int = 128,
    learning_rate: float = 1e-3,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.85,
    buffer_size: int = 20000,
    batch_size: int = 32,
    learning_starts: int = 64,
    target_update_interval: int = 100,
    max_grad_norm: float = 10.0,
    decision_interval: float = 5.0,
    min_green: float = 5.0,
    max_steps_per_episode: Optional[int] = 3600,
    model_dir: str = "models/dqn",
    log_dir: str = "results/training",
) -> Dict[str, Any]:
    """Trains a Deep Q-Network agent on a specified SUMO scenario.

    Args:
        scenario_name: Scenario name ('cologne1' or 'ingolstadt1').
        num_episodes: Number of training episodes.
        seed: Base random seed.
        hidden_dim: Number of neurons per hidden layer.
        learning_rate: Optimizer learning rate.
        gamma: Discount factor.
        epsilon_start: Initial exploration rate.
        epsilon_min: Minimum exploration rate.
        epsilon_decay: Per-episode decay factor for epsilon.
        buffer_size: Maximum replay buffer transitions.
        batch_size: Minibatch sampling size.
        learning_starts: Minimum transitions before starting gradient updates.
        target_update_interval: Frequency (in updates) to sync target network.
        max_grad_norm: Maximum gradient norm for clipping.
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
    print(f"Starting DQN Training on [{scenario_name.upper()}]")
    print(f"Episodes: {num_episodes} | LR: {learning_rate} | Gamma: {gamma} | Hidden: {hidden_dim}")
    print(f"Buffer: {buffer_size} | Batch: {batch_size} | Target Update: every {target_update_interval}")
    print(f"Decision Interval: {decision_interval}s | Seed: {seed}")
    print(f"========================================================")

    env = TrafficSignalRLEnv(
        scenario_name=scenario_name,
        decision_interval=decision_interval,
        min_green=min_green,
        max_steps=max_steps_per_episode,
        seed=seed,
        output_dir=os.path.join(log_dir, f"env_dqn_{scenario_name}"),
        use_tripinfo=True,
    )

    agent = DQNController(
        observation_dim=env.observation_dim,
        action_dim=env.num_actions,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_starts=learning_starts,
        target_update_interval=target_update_interval,
        max_grad_norm=max_grad_norm,
        obs_builder=env.obs_builder,
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
            valid_actions = info["valid_actions"]

            ep_reward = 0.0
            ep_steps = 0
            ep_losses: List[float] = []

            while True:
                action = agent.select_action_for_state(
                    obs_vec, valid_actions=valid_actions
                )

                next_obs, reward, terminated, truncated, next_info = env.step(action)
                done = bool(terminated or truncated)
                next_mask = next_info["action_mask"]

                # Push to replay buffer
                agent.replay_buffer.push(
                    state=obs_vec,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    done=done,
                    valid_next_mask=next_mask,
                )

                # Gradient update step
                loss_val = agent.train_step()
                if loss_val is not None:
                    ep_losses.append(loss_val)

                ep_reward += reward
                ep_steps += 1

                obs_vec = next_obs
                valid_actions = next_info["valid_actions"]

                if done:
                    break

            ep_duration = time.time() - ep_start_time
            curr_eps = agent.epsilon
            agent.decay_epsilon()

            summary = env.get_summary() or {}
            mean_wait = summary.get("mean_waiting_time", 0.0)
            mean_queue = summary.get("mean_queue_length", 0.0)
            completed_vehs = summary.get("total_completed_vehicles", 0)
            avg_loss = float(np.mean(ep_losses)) if ep_losses else 0.0

            ep_record = {
                "episode": ep,
                "reward": round(ep_reward, 2),
                "steps": ep_steps,
                "epsilon": round(curr_eps, 4),
                "mean_loss": round(avg_loss, 4),
                "total_updates": agent.total_updates,
                "buffer_size": len(agent.replay_buffer),
                "mean_waiting_time": round(mean_wait, 2),
                "mean_queue_length": round(mean_queue, 2),
                "completed_vehicles": completed_vehs,
                "duration_seconds": round(ep_duration, 2),
            }
            history.append(ep_record)

            print(
                f"Ep {ep:2d}/{num_episodes} | Rew: {ep_reward:8.1f} | Loss: {avg_loss:6.4f} | "
                f"Eps: {curr_eps:.3f} | Updates: {agent.total_updates:4d} | "
                f"Wait: {mean_wait:5.2f}s | Queue: {mean_queue:4.2f} | Completed: {completed_vehs:4d} | "
                f"Time: {ep_duration:4.1f}s"
            )

            # Save checkpoint if best episode reward or periodic
            if ep_reward > best_reward:
                best_reward = ep_reward
                best_path = os.path.join(model_dir, f"dqn_{scenario_name}_best.pt")
                agent.save(best_path)

            if ep % 5 == 0 or ep == num_episodes:
                ckpt_path = os.path.join(model_dir, f"dqn_{scenario_name}_ep{ep}.pt")
                agent.save(ckpt_path)

    finally:
        env.close()

    total_wall_time = time.time() - start_total_time

    # Save final model
    final_model_path = os.path.join(model_dir, f"dqn_{scenario_name}_final.pt")
    agent.save(final_model_path)

    # Save training log
    log_file = os.path.join(log_dir, f"dqn_{scenario_name}_log.json")
    training_data = {
        "scenario_name": scenario_name,
        "algorithm": "DQN",
        "num_episodes": num_episodes,
        "seed": seed,
        "hyperparameters": {
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_min": epsilon_min,
            "epsilon_decay": epsilon_decay,
            "buffer_size": buffer_size,
            "batch_size": batch_size,
            "learning_starts": learning_starts,
            "target_update_interval": target_update_interval,
            "max_grad_norm": max_grad_norm,
            "decision_interval": decision_interval,
            "min_green": min_green,
        },
        "total_wall_clock_seconds": round(total_wall_time, 2),
        "history": history,
        "final_updates": agent.total_updates,
        "best_reward": round(best_reward, 2),
    }

    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(training_data, f, indent=2)

    print(f"\nTraining completed in {total_wall_time:.2f}s.")
    print(f"Final Model saved to: {final_model_path}")
    print(f"Training Log saved to: {log_file}")

    return training_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN on SUMO.")
    parser.add_argument("--dataset", type=str, default="ingolstadt1", choices=["cologne1", "ingolstadt1"])
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--decay", type=float, default=0.85)
    parser.add_argument("--steps", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train_dqn(
        scenario_name=args.dataset,
        num_episodes=args.episodes,
        learning_rate=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.decay,
        max_steps_per_episode=args.steps,
        seed=args.seed,
    )
