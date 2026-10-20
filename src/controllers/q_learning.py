"""Tabular Q-Learning Traffic Signal Controller."""

import ast
import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.config import DEFAULT_SAFETY_CONFIG
from src.controllers.base import BaseTrafficController, IntersectionObservation
from src.rl.discretization import StateDiscretizer
from src.io_utils import write_json_atomic


class QLearningController(BaseTrafficController):
    """Tabular Q-Learning agent for traffic signal control."""

    PARAM_TYPES = {
        "num_actions": int, "learning_rate": float, "gamma": float,
        "epsilon": float, "epsilon_min": float, "epsilon_decay": float,
        "decision_interval": float, "min_green": float, "max_green": float,
        "max_red": float, "yellow_duration": float,
        "all_red_duration": float, "tie_break_strategy": str,
    }

    def __init__(
        self,
        num_actions: int,
        learning_rate: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.98,
        discretizer: Optional[StateDiscretizer] = None,
        target_tls_id: Optional[str] = None,
        decision_interval: float = 5.0,
        min_green: float = DEFAULT_SAFETY_CONFIG["min_green"],
        max_green: Optional[float] = DEFAULT_SAFETY_CONFIG["max_green"],
        max_red: Optional[float] = None,
        yellow_duration: float = DEFAULT_SAFETY_CONFIG["yellow_duration"],
        all_red_duration: float = DEFAULT_SAFETY_CONFIG["all_red_duration"],
        seed: Optional[int] = None,
        eval_mode: bool = False,
        tie_break_strategy: str = "first",
    ):
        super().__init__(name="QLearning")
        if isinstance(num_actions, bool) or not isinstance(num_actions, (int, np.integer)) or num_actions <= 0:
            raise ValueError(f"num_actions must be positive and an integer, got {num_actions}")

        self.num_actions = int(num_actions)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.discretizer = discretizer or StateDiscretizer(min_green=min_green)
        self.target_tls_id = target_tls_id
        self.decision_interval = float(decision_interval)
        self.min_green = float(min_green)
        self.max_green = float(max_green) if max_green is not None else None
        self.max_red = float(max_red) if max_red is not None else None
        self.yellow_duration = float(yellow_duration)
        self.all_red_duration = float(all_red_duration)
        self.eval_mode = bool(eval_mode)
        self.tie_break_strategy = str(tie_break_strategy).lower()
        self._validate_parameters()

        self._rng = random.Random(seed)

        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}
        self.action_counts: np.ndarray = np.zeros(self.num_actions, dtype=int)
        self.total_updates: int = 0
        self.reward_config = None

    def _validate_parameters(self) -> None:
        for key in ("learning_rate", "gamma", "epsilon", "epsilon_min", "epsilon_decay"):
            value = getattr(self, key)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{key} must be finite and within [0, 1]")
        for key in ("decision_interval", "yellow_duration"):
            value = getattr(self, key)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("min_green", "all_red_duration"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be non-negative and finite")
        for key in ("max_green", "max_red"):
            value = getattr(self, key)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be positive and finite or None")
        if self.max_green is not None and self.max_green < self.min_green:
            raise ValueError("max_green must be >= min_green")
        if self.tie_break_strategy not in ("first", "random"):
            raise ValueError("tie_break_strategy must be first or random")

    def reset(self) -> None:
        """Resets transient execution state."""
        pass

    def get_q_values(self, state: Tuple[int, ...]) -> np.ndarray:
        """Retrieves or initializes Q-values for a given discrete state."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions, dtype=np.float64)
        return self.q_table[state]

    def select_action_for_state(
        self,
        state: Tuple[int, ...],
        valid_actions: Optional[Sequence[int]] = None,
        current_phase: Optional[int] = None,
    ) -> int:
        """Selects action via epsilon-greedy strategy with anti-chatter tie breaking."""
        candidates = list(range(self.num_actions)) if valid_actions is None else list(valid_actions)
        if not candidates or any(not isinstance(action, (int, np.integer)) or not 0 <= action < self.num_actions for action in candidates):
            raise ValueError("Action mask must contain valid action indices")

        if not self.eval_mode and self._rng.random() < self.epsilon:
            chosen = self._rng.choice(candidates)
        else:
            q_vals = (self.q_table.get(state, np.zeros(self.num_actions))
                      if self.eval_mode else self.get_q_values(state))
            cand_q = [q_vals[a] for a in candidates]
            max_q = max(cand_q)
            best = [a for a, q in zip(candidates, cand_q) if abs(q - max_q) <= 1e-8]
            if current_phase is not None and current_phase in best:
                chosen = current_phase
            elif self.tie_break_strategy == "random" and not self.eval_mode:
                chosen = self._rng.choice(best)
            else:
                chosen = best[0]

        if not self.eval_mode:
            self.action_counts[chosen] += 1
        return chosen

    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        done: bool = False,
        valid_next_actions: Optional[Sequence[int]] = None,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
    ) -> float:
        """Applies Bellman Q-learning update."""
        if self.eval_mode:
            return 0.0
        if not isinstance(action, (int, np.integer)) or not 0 <= action < self.num_actions:
            raise ValueError("Action index is out of range")
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")

        is_term = terminated if terminated is not None else done
        q_curr = self.get_q_values(state)

        if is_term:
            target = float(reward)
        else:
            next_q = self.get_q_values(next_state)
            cand = list(range(self.num_actions)) if valid_next_actions is None else list(valid_next_actions)
            if not cand or any(not 0 <= a < self.num_actions for a in cand):
                raise ValueError("Next action mask must contain valid action indices")
            target = float(reward + self.gamma * max(next_q[a] for a in cand))

        td_error = target - q_curr[action]
        q_curr[action] += self.learning_rate * td_error
        self.total_updates += 1
        return abs(td_error)

    def decay_epsilon(self) -> float:
        """Decays exploration rate down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def select_action(
        self, observations: Dict[str, IntersectionObservation]
    ) -> Dict[str, int]:
        """Maps observations to target phase actions."""
        actions: Dict[str, int] = {}
        for tls_id, obs in observations.items():
            if self.target_tls_id and tls_id != self.target_tls_id:
                continue
            state = self.discretizer.discretize(obs)
            actions[tls_id] = self.select_action_for_state(
                state, valid_actions=obs.valid_actions, current_phase=obs.current_phase
            )
        return actions

    def validate_compatibility(self, min_green=None, max_green=None, num_phases_per_tls=None) -> List[str]:
        """Check the checkpoint action space once, before opening SUMO."""
        phases = num_phases_per_tls or {}
        targets = [self.target_tls_id] if self.target_tls_id else list(phases)
        return [f"Checkpoint actions ({self.num_actions}) do not match TLS {tls} phases ({phases.get(tls)})"
                for tls in targets if phases.get(tls) != self.num_actions]

    def get_config(self) -> Dict[str, Any]:
        """Returns controller configuration dictionary."""
        config_keys = (
            "name", "num_actions", "learning_rate", "gamma", "epsilon",
            "epsilon_min", "epsilon_decay", "eval_mode", "target_tls_id",
            "decision_interval", "min_green", "max_green", "max_red", "yellow_duration",
            "all_red_duration", "tie_break_strategy",
        )
        cfg = {k: getattr(self, k) for k in config_keys}
        cfg["num_states_in_table"] = len(self.q_table)
        cfg["discretizer_config"] = self.discretizer.get_config()
        return cfg

    def save(self, filepath: str) -> None:
        """Saves Q-table and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "config": self.get_config(),
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "action_counts": self.action_counts.tolist(),
            "reward_config": self.reward_config,
            "total_updates": self.total_updates,
        }
        write_json_atomic(filepath, data)

    def load(self, filepath: str) -> None:
        """Loads Q-table and configuration from disk."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._load_checkpoint_data(data)

    def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
        """Restores an already decoded checkpoint without reopening the file."""
        if not isinstance(data, dict) or not isinstance(data.get("config", {}), dict) or not isinstance(data.get("q_table", {}), dict):
            raise ValueError("Checkpoint config and q_table must be dictionaries")
        cfg = data.get("config", {})
        if "num_actions" in cfg and (type(cfg["num_actions"]) is not int or cfg["num_actions"] <= 0):
            raise ValueError("Checkpoint num_actions must be a positive integer")
        for attr, convert in self.PARAM_TYPES.items():
            if attr in cfg and cfg[attr] is not None:
                try:
                    setattr(self, attr, convert(cfg[attr]))
                except (ValueError, TypeError, OverflowError):
                    raise ValueError(f"Invalid value for '{attr}' in checkpoint: {cfg[attr]}")

        if "target_tls_id" in cfg:
            self.target_tls_id = str(cfg["target_tls_id"]) if cfg["target_tls_id"] is not None else None
        for attr in ("max_green", "max_red"):
            if attr in cfg:
                setattr(self, attr, float(cfg[attr]) if cfg[attr] is not None else None)
        self._validate_parameters()

        self.reward_config = data.get("reward_config")
        self.discretizer = StateDiscretizer.from_config(
            cfg.get("discretizer_config") or {}, min_green=self.min_green,
        )

        self.q_table.clear()
        for k_str, v_list in data.get("q_table", {}).items():
            k_tuple = ast.literal_eval(k_str)
            values = np.asarray(v_list, dtype=np.float64)
            if not isinstance(k_tuple, tuple) or len(k_tuple) < 4 or any(type(x) is not int or x < 0 for x in k_tuple):
                raise ValueError("Checkpoint state must contain non-negative integers")
            if values.shape != (self.num_actions,) or not np.isfinite(values).all():
                raise ValueError("Checkpoint Q-vector shape or values are invalid")
            self.q_table[k_tuple] = values

        self.action_counts = np.array(data.get("action_counts", [0] * self.num_actions), dtype=int)
        if self.action_counts.shape != (self.num_actions,) or (self.action_counts < 0).any():
            raise ValueError("Checkpoint action_counts shape or values are invalid")
        self.total_updates = int(data.get("total_updates", 0))

    @classmethod
    def from_checkpoint(
        cls,
        filepath: str,
        eval_mode: bool = True,
        seed: Optional[int] = None,
        **override_kwargs: Any,
    ) -> "QLearningController":
        """Loads a QLearningController directly from a JSON checkpoint file."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = data.get("config", {})
        if not isinstance(cfg, dict):
            raise ValueError("Checkpoint config must be a dictionary")
        raw_actions = cfg.get("num_actions", 2)
        try:
            num_actions = int(raw_actions)
        except (ValueError, TypeError, OverflowError):
            raise ValueError(f"Invalid num_actions in checkpoint: {raw_actions}")

        inst = cls(num_actions=num_actions, seed=seed, eval_mode=eval_mode)
        inst._load_checkpoint_data(data)
        if eval_mode:
            inst.eval_mode = True
            inst.epsilon = 0.0

        allowed = {*DEFAULT_SAFETY_CONFIG, "decision_interval", "target_tls_id", "max_red"}
        if override_kwargs.keys() - allowed:
            raise TypeError(f"Unknown checkpoint overrides: {override_kwargs.keys() - allowed}")
        if override_kwargs.get("min_green", inst.min_green) != inst.discretizer.min_green:
            raise ValueError("min_green override changes the learned discretizer state encoding")
        for key, value in override_kwargs.items():
            setattr(inst, key, value)
        inst._validate_parameters()

        return inst
