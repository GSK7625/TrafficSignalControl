"""Evaluation session and loop utilities for trained Q-Learning agents."""

from typing import Any, Dict, Optional, Tuple

from src.config import DEFAULT_SAFETY_CONFIG
from src.controllers.q_learning import QLearningController
from src.rl.discretization import StateDiscretizer
from src.rl.environment import TrafficSignalRLEnv
from src.rl.rewards import NegativePressureReward


def load_checkpoint_env_params(model_path: Optional[str]) -> Dict[str, Any]:
    """Inspects checkpoint metadata to extract environment and discretizer configuration."""
    if model_path is None:
        return {}
    return checkpoint_env_params(QLearningController.from_checkpoint(model_path).get_config())


def checkpoint_env_params(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Maps controller metadata to environment options in one place."""
    params: Dict[str, Any] = {}
    if "num_actions" in cfg:
        params["num_actions"] = int(cfg["num_actions"])
    if "target_tls_id" in cfg and cfg["target_tls_id"] is not None:
        params["target_tls_id"] = str(cfg["target_tls_id"])

    params["decision_interval"] = float(cfg.get("decision_interval", 5.0))
    params["yellow_duration"] = float(cfg.get("yellow_duration", DEFAULT_SAFETY_CONFIG["yellow_duration"]))
    params["all_red_duration"] = float(cfg.get("all_red_duration", DEFAULT_SAFETY_CONFIG["all_red_duration"]))
    params["min_green"] = float(cfg.get("min_green", DEFAULT_SAFETY_CONFIG["min_green"]))
    max_green = cfg.get("max_green", DEFAULT_SAFETY_CONFIG["max_green"])
    params["max_green"] = float(max_green) if max_green is not None else None

    d_cfg = cfg.get("discretizer_config", {})
    if isinstance(d_cfg, dict) and d_cfg:
        params["state_discretizer"] = StateDiscretizer.from_config(d_cfg, params["min_green"])
    return params


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
    params = checkpoint_env_params(agent.get_config())
    params.pop("num_actions", None)
    params.update(timing_overrides or {})
    params["state_discretizer"] = agent.discretizer
    reward = dict(agent.reward_config or {})
    if reward.pop("name", "NegativePressureReward") != "NegativePressureReward":
        raise ValueError("Unsupported checkpoint reward function")
    env_options.setdefault("reward_fn", NegativePressureReward(**reward))
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
        _, info = env.reset(seed=seed)
        valid_actions = info["valid_actions"]
        discrete_state = info["discrete_state"]

        while True:
            curr_phase = discrete_state[0]
            action = agent.select_action_for_state(
                discrete_state, valid_actions=valid_actions, current_phase=curr_phase
            )
            _, reward, terminated, truncated, next_info = env.step(action)
            total_reward += reward
            valid_actions = next_info["valid_actions"]
            discrete_state = next_info["discrete_state"]
            if terminated or truncated:
                break
    finally:
        agent.eval_mode = prev_mode
        agent.epsilon = prev_eps

    summary = env.get_summary() or {}
    return float(round(total_reward, 2)), summary
