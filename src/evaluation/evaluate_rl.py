"""Evaluation session and loop utilities for trained Q-Learning agents."""

from typing import Any, Dict, Optional, Tuple

from src.config import DEFAULT_SAFETY_CONFIG
from src.controllers.q_learning import QLearningController
from src.rl.environment import TrafficSignalRLEnv
from src.rl.rewards import MetricsAlignedReward, NegativePressureReward


def create_evaluation_session(
    scenario_name: str,
    model_path: Optional[str] = None,
    seed: int = 42,
    timing_overrides: Optional[Dict[str, Any]] = None,
    **env_options: Any,
) -> Tuple[QLearningController, TrafficSignalRLEnv]:
    """Creates a compatible agent/environment pair for evaluation.

    Explicit timing options override checkpoint metadata. An explicit max_green=None
    disables the limit; an absent key inherits checkpoint or project defaults.
    The caller owns the returned environment and must close it.
    """
    if not model_path:
        raise FileNotFoundError("Evaluation requires a trained Q-Learning checkpoint")
    agent = QLearningController.from_checkpoint(model_path, seed=seed, **(timing_overrides or {}))
    config = agent.get_config()
    params = {key: config[key] for key in (*DEFAULT_SAFETY_CONFIG, "max_red", "decision_interval", "target_tls_id")}
    params["state_discretizer"] = agent.discretizer
    reward = dict(agent.reward_config or {})
    reward_name = reward.pop("name", "NegativePressureReward")
    if reward_name == "NegativePressureReward":
        reward_kwargs = {k: v for k, v in reward.items() if k in ("scale", "pending_weight")}
        env_options.setdefault("reward_fn", NegativePressureReward(**reward_kwargs))
    elif reward_name == "MetricsAlignedReward":
        valid_keys = ("scale", "w_completed", "w_queue", "w_waiting", "w_switch", "w_pending")
        reward_kwargs = {k: v for k, v in reward.items() if k in valid_keys}
        env_options.setdefault("reward_fn", MetricsAlignedReward(**reward_kwargs))
    else:
        raise ValueError(f"Unsupported checkpoint reward function: {reward_name}")
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
        discrete_state, info = env.reset(seed=seed)
        valid_actions = info["valid_actions"]
        if discrete_state is None:
            discrete_state = info["discrete_state"]

        while True:
            curr_phase = discrete_state[0]
            action = agent.select_action_for_state(
                discrete_state, valid_actions=valid_actions, current_phase=curr_phase
            )
            next_state, reward, terminated, truncated, next_info = env.step(action)
            total_reward += reward
            valid_actions = next_info["valid_actions"]
            discrete_state = next_state if next_state is not None else next_info.get("discrete_state")
            if terminated or truncated:
                break
    finally:
        agent.eval_mode = prev_mode
        agent.epsilon = prev_eps

    summary = env.get_summary() or {}
    return float(round(total_reward, 2)), summary
