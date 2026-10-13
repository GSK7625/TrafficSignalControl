"""Training pipeline for Deep Q-Network (DQN) on SUMO traffic networks."""

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np

from src.config import load_yaml_config
from src.controllers.dqn import DQNController
from src.rl.environment import TrafficSignalRLEnv
from src.rl.rewards import BaseRewardFunction, NegativeAverageQueueReward


def evaluate_greedy_dqn(
    agent: DQNController,
    scenario_name: str,
    decision_interval: float = 5.0,
    min_green: float = 5.0,
    yellow_duration: float = 3.0,
    all_red_duration: float = 0.0,
    reward_fn: Optional[BaseRewardFunction] = None,
    max_queue_scale: float = 20.0,
    max_time_scale: float = 60.0,
    max_steps: Optional[int] = 3600,
    seed: int = 42,
) -> float:
    """Evaluates the DQN agent deterministically (epsilon=0, eval_mode=True) with RNG state isolation."""
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
        max_queue_scale=max_queue_scale,
        max_time_scale=max_time_scale,
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
        valid_actions = info["valid_actions"]
        while True:
            action = agent.select_action_for_state(obs, valid_actions=valid_actions)
            next_obs, reward, terminated, truncated, next_info = val_env.step(action)
            total_reward += reward
            obs = next_obs
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


def train_dqn(
    scenario_name: str = "ingolstadt1",
    num_episodes: Optional[int] = None,
    seed: Optional[int] = None,
    hidden_dim: Optional[int] = None,
    learning_rate: Optional[float] = None,
    gamma: Optional[float] = None,
    epsilon_start: Optional[float] = None,
    epsilon_min: Optional[float] = None,
    epsilon_decay: Optional[float] = None,
    buffer_size: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_starts: Optional[int] = None,
    target_update_interval: Optional[int] = None,
    max_grad_norm: Optional[float] = None,
    decision_interval: Optional[float] = None,
    min_green: Optional[float] = None,
    yellow_duration: Optional[float] = None,
    all_red_duration: Optional[float] = None,
    max_steps_per_episode: Optional[int] = None,
    val_interval: Optional[int] = None,
    config_path: Optional[str] = None,
    model_dir: str = "models/dqn",
    log_dir: str = "results/training",
) -> Dict[str, Any]:
    """Trains a Deep Q-Network agent on a specified SUMO scenario."""
    # 1. Base defaults (lowest priority)
    num_episodes_val = 15
    seed_val = 42
    hidden_dim_val = 128
    learning_rate_val = 1e-3
    gamma_val = 0.95
    epsilon_start_val = 1.0
    epsilon_min_val = 0.05
    epsilon_decay_val = 0.85
    buffer_size_val = 20000
    batch_size_val = 32
    learning_starts_val = 64
    target_update_interval_val = 100
    max_grad_norm_val = 10.0
    decision_interval_val = 5.0
    min_green_val = 5.0
    yellow_dur = 3.0
    all_red_dur = 0.0
    max_steps_val = 3600
    val_interval_val = 5
    max_q_scale = 20.0
    max_t_scale = 60.0
    reward_scale = 1.0

    # 2. YAML config (medium priority - overrides base defaults)
    if config_path:
        cfg = load_yaml_config(config_path)
        hp = cfg.get("hyperparameters", {})
        net = cfg.get("network", {})
        if "hidden_dim" in net:
            hidden_dim_val = int(net["hidden_dim"])
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
        if "buffer_size" in hp:
            buffer_size_val = int(hp["buffer_size"])
        if "batch_size" in hp:
            batch_size_val = int(hp["batch_size"])
        if "learning_starts" in hp:
            learning_starts_val = int(hp["learning_starts"])
        if "target_update_interval" in hp:
            target_update_interval_val = int(hp["target_update_interval"])
        if "max_grad_norm" in hp:
            max_grad_norm_val = float(hp["max_grad_norm"])
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

        obs_cfg = cfg.get("observation", {})
        if "max_queue_scale" in obs_cfg or "max_queue_scale" in cfg:
            max_q_scale = float(obs_cfg.get("max_queue_scale", cfg.get("max_queue_scale", max_q_scale)))
        if "max_time_scale" in obs_cfg or "max_time_scale" in cfg:
            max_t_scale = float(obs_cfg.get("max_time_scale", cfg.get("max_time_scale", max_t_scale)))

        reward_cfg = cfg.get("reward", {})
        if "scale" in reward_cfg or "reward_scale" in cfg:
            reward_scale = float(reward_cfg.get("scale", cfg.get("reward_scale", reward_scale)))

    # 3. Explicit arguments / CLI (highest priority - overrides YAML)
    if num_episodes is not None:
        num_episodes_val = int(num_episodes)
    if seed is not None:
        seed_val = int(seed)
    if hidden_dim is not None:
        hidden_dim_val = int(hidden_dim)
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
    if buffer_size is not None:
        buffer_size_val = int(buffer_size)
    if batch_size is not None:
        batch_size_val = int(batch_size)
    if learning_starts is not None:
        learning_starts_val = int(learning_starts)
    if target_update_interval is not None:
        target_update_interval_val = int(target_update_interval)
    if max_grad_norm is not None:
        max_grad_norm_val = float(max_grad_norm)
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
    hidden_dim = hidden_dim_val
    learning_rate = learning_rate_val
    gamma = gamma_val
    epsilon_start = epsilon_start_val
    epsilon_min = epsilon_min_val
    epsilon_decay = epsilon_decay_val
    buffer_size = buffer_size_val
    batch_size = batch_size_val
    learning_starts = learning_starts_val
    target_update_interval = target_update_interval_val
    max_grad_norm = max_grad_norm_val
    decision_interval = decision_interval_val
    min_green = min_green_val
    max_steps_per_episode = max_steps_val
    val_interval = val_interval_val

    reward_fn = NegativeAverageQueueReward(scale=reward_scale)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"\n========================================================")
    print(f"Starting DQN Training on [{scenario_name.upper()}]")
    print(f"Episodes: {num_episodes} | LR: {learning_rate} | Gamma: {gamma} | Hidden: {hidden_dim}")
    print(f"Buffer: {buffer_size} | Batch: {batch_size} | Target Update: every {target_update_interval}")
    print(f"Decision Interval: {decision_interval}s | Seed: {seed} | Yellow: {yellow_dur}s | All-Red: {all_red_dur}s")
    print(f"========================================================")

    env = TrafficSignalRLEnv(
        scenario_name=scenario_name,
        decision_interval=decision_interval,
        min_green=min_green,
        yellow_duration=yellow_dur,
        all_red_duration=all_red_dur,
        reward_fn=reward_fn,
        max_queue_scale=max_q_scale,
        max_time_scale=max_t_scale,
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
                    terminated=bool(terminated),
                    truncated=bool(truncated),
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

            # Save checkpoint based on greedy evaluation or periodic interval
            if val_interval > 0 and (ep % val_interval == 0 or ep == num_episodes):
                val_reward = evaluate_greedy_dqn(
                    agent=agent,
                    scenario_name=scenario_name,
                    decision_interval=decision_interval,
                    min_green=min_green,
                    yellow_duration=yellow_dur,
                    all_red_duration=all_red_dur,
                    reward_fn=NegativeAverageQueueReward(scale=reward_scale),
                    max_queue_scale=max_q_scale,
                    max_time_scale=max_t_scale,
                    max_steps=max_steps_per_episode,
                    seed=seed + 9999,
                )
                print(f"    --> [Greedy Eval] Ep {ep} Validation Reward: {val_reward:8.1f} (Best: {best_reward:8.1f})")
                if val_reward > best_reward:
                    best_reward = val_reward
                    best_path = os.path.join(model_dir, f"dqn_{scenario_name}_best.pt")
                    agent.save(best_path)
                    print(f"    --> [NEW BEST CHECKPOINT] Saved to {best_path}")
            elif val_interval <= 0:
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
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--decay", type=float, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--val-interval", type=int, default=None, help="Interval of episodes for greedy evaluation")

    args = parser.parse_args()
    train_dqn(
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
