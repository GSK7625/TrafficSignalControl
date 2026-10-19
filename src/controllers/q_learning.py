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


def validate_q_learning_hyperparameters(
    num_actions: int,
    learning_rate: float,
    gamma: float,
    epsilon: float = 1.0,
    epsilon_min: float = 0.01,
    epsilon_decay: float = 0.98,
    decision_interval: float = 5.0,
    min_green: float = 5.0,
    max_green: Optional[float] = None,
    yellow_duration: float = 3.0,
    all_red_duration: float = 1.0,
) -> None:
    """Validates domain constraints for Q-Learning hyperparameters."""
    if num_actions <= 0:
        raise ValueError(f"num_actions must be positive, got {num_actions}")
    if not (math.isfinite(learning_rate) and 0.0 < learning_rate <= 1.0):
        raise ValueError(f"learning_rate must be in (0, 1] and finite, got {learning_rate}")
    if not (math.isfinite(gamma) and 0.0 <= gamma <= 1.0):
        raise ValueError(f"gamma must be in [0, 1] and finite, got {gamma}")
    if not (math.isfinite(epsilon) and 0.0 <= epsilon <= 1.0):
        raise ValueError(f"epsilon must be in [0, 1] and finite, got {epsilon}")
    if not (math.isfinite(epsilon_min) and 0.0 <= epsilon_min <= 1.0):
        raise ValueError(f"epsilon_min must be in [0, 1] and finite, got {epsilon_min}")
    if not (math.isfinite(epsilon_decay) and 0.0 <= epsilon_decay <= 1.0):
        raise ValueError(f"epsilon_decay must be in [0, 1] and finite, got {epsilon_decay}")
    if not (math.isfinite(decision_interval) and decision_interval > 0):
        raise ValueError(f"decision_interval must be positive and finite, got {decision_interval}")
    if not (math.isfinite(min_green) and min_green >= 0):
        raise ValueError(f"min_green must be non-negative and finite, got {min_green}")
    if max_green is not None:
        if not (math.isfinite(max_green) and max_green > 0):
            raise ValueError(f"max_green must be positive and finite, got {max_green}")
        if max_green < min_green:
            raise ValueError(f"max_green must be >= min_green ({max_green} < {min_green})")
    if not (math.isfinite(yellow_duration) and yellow_duration > 0):
        raise ValueError(f"yellow_duration must be positive and finite, got {yellow_duration}")
    if not (math.isfinite(all_red_duration) and all_red_duration >= 0):
        raise ValueError(f"all_red_duration must be non-negative and finite, got {all_red_duration}")


class QLearningController(BaseTrafficController):
    """Tabular Q-Learning agent for traffic signal control."""

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
        yellow_duration: float = DEFAULT_SAFETY_CONFIG["yellow_duration"],
        all_red_duration: float = DEFAULT_SAFETY_CONFIG["all_red_duration"],
        seed: Optional[int] = None,
        eval_mode: bool = False,
        tie_break_strategy: str = "first",
    ):
        super().__init__(name="QLearning")
        validate_q_learning_hyperparameters(
            num_actions=num_actions,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            decision_interval=decision_interval,
            min_green=min_green,
            max_green=max_green,
            yellow_duration=yellow_duration,
            all_red_duration=all_red_duration,
        )

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
        self.yellow_duration = float(yellow_duration)
        self.all_red_duration = float(all_red_duration)
        self.eval_mode = bool(eval_mode)
        self.tie_break_strategy = str(tie_break_strategy).lower()

        if seed is not None:
            self.set_seed(seed)

        self.q_table: Dict[Tuple[int, ...], np.ndarray] = {}
        self.action_counts: np.ndarray = np.zeros(self.num_actions, dtype=int)
        self.total_updates: int = 0
        self.reward_config = None

    def set_seed(self, seed: int) -> None:
        """Sets random seed for reproducible exploration."""
        random.seed(seed)
        np.random.seed(seed)

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
        if not candidates or any(not 0 <= action < self.num_actions for action in candidates):
            raise ValueError("Action mask must contain valid action indices")

        if not self.eval_mode and random.random() < self.epsilon:
            chosen = random.choice(candidates)
        else:
            # Evaluation must not insert unseen states into the learned Q-table.
            q_vals = (self.q_table.get(state, np.zeros(self.num_actions))
                      if self.eval_mode else self.get_q_values(state))
            cand_q = [q_vals[a] for a in candidates]
            max_q = max(cand_q)
            best = [a for a, q in zip(candidates, cand_q) if abs(q - max_q) <= 1e-8]
            if current_phase is not None and current_phase in best:
                chosen = current_phase
            elif self.tie_break_strategy == "random" and not self.eval_mode:
                chosen = random.choice(best)
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
        return {
            **{key: getattr(self, key) for key in (
                "name", "num_actions", "learning_rate", "gamma", "epsilon",
                "epsilon_min", "epsilon_decay", "eval_mode", "target_tls_id",
                "decision_interval", "min_green", "max_green", "yellow_duration",
                "all_red_duration", "tie_break_strategy",
            )},
            "num_states_in_table": len(self.q_table),
            "discretizer_config": self.discretizer.get_config(),
        }

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
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Loads Q-table and configuration from disk."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._load_checkpoint_data(data)

    def _load_checkpoint_data(self, data: Dict[str, Any]) -> None:
        """Restores an already decoded checkpoint without reopening the file."""
        cfg = data.get("config", {})
        converters = {
            "num_actions": int, "learning_rate": float, "gamma": float,
            "epsilon": float, "epsilon_min": float, "epsilon_decay": float,
            "decision_interval": float, "min_green": float, "yellow_duration": float,
            "all_red_duration": float, "tie_break_strategy": str,
        }
        parsed: Dict[str, Any] = {}
        for attr, convert in converters.items():
            if attr in cfg and cfg[attr] is not None:
                try:
                    parsed[attr] = convert(cfg[attr])
                except (ValueError, TypeError, OverflowError):
                    raise ValueError(f"Invalid value for '{attr}' in checkpoint: {cfg[attr]}")
        if "target_tls_id" in cfg:
            target_tls_id = str(cfg["target_tls_id"]) if cfg["target_tls_id"] is not None else None
        else:
            target_tls_id = getattr(self, "target_tls_id", None)

        if "max_green" in cfg:
            max_green = float(cfg["max_green"]) if cfg["max_green"] is not None else None
        else:
            max_green = getattr(self, "max_green", None)

        merged = {
            "num_actions": parsed.get("num_actions", getattr(self, "num_actions", 2)),
            "learning_rate": parsed.get("learning_rate", getattr(self, "learning_rate", 0.1)),
            "gamma": parsed.get("gamma", getattr(self, "gamma", 0.95)),
            "epsilon": parsed.get("epsilon", getattr(self, "epsilon", 1.0)),
            "epsilon_min": parsed.get("epsilon_min", getattr(self, "epsilon_min", 0.01)),
            "epsilon_decay": parsed.get("epsilon_decay", getattr(self, "epsilon_decay", 0.98)),
            "decision_interval": parsed.get("decision_interval", getattr(self, "decision_interval", 5.0)),
            "min_green": parsed.get("min_green", getattr(self, "min_green", 5.0)),
            "max_green": max_green,
            "yellow_duration": parsed.get("yellow_duration", getattr(self, "yellow_duration", 3.0)),
            "all_red_duration": parsed.get("all_red_duration", getattr(self, "all_red_duration", 1.0)),
        }
        validate_q_learning_hyperparameters(**merged)

        self.reward_config = data.get("reward_config")
        for attr, val in parsed.items():
            setattr(self, attr, val)
        if "target_tls_id" in cfg:
            self.target_tls_id = target_tls_id
        if "max_green" in cfg:
            self.max_green = max_green

        self.discretizer = StateDiscretizer.from_config(
            cfg.get("discretizer_config") or {}, min_green=self.min_green,
        )

        self.q_table.clear()
        for k_str, v_list in data.get("q_table", {}).items():
            k_tuple = ast.literal_eval(k_str)
            values = np.asarray(v_list, dtype=np.float64)
            if not isinstance(k_tuple, tuple) or len(k_tuple) != 4 or any(type(x) is not int or x < 0 for x in k_tuple):
                raise ValueError("Checkpoint state must contain four non-negative integers")
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

        allowed = {*DEFAULT_SAFETY_CONFIG, "decision_interval", "target_tls_id"}
        if override_kwargs.keys() - allowed:
            raise TypeError(f"Unknown checkpoint overrides: {override_kwargs.keys() - allowed}")
        if override_kwargs.get("min_green", inst.min_green) != inst.discretizer.min_green:
            raise ValueError("min_green override changes the learned discretizer state encoding")
        for key, value in override_kwargs.items():
            setattr(inst, key, value)
        validate_q_learning_hyperparameters(
            num_actions=inst.num_actions,
            learning_rate=inst.learning_rate,
            gamma=inst.gamma,
            epsilon=inst.epsilon,
            epsilon_min=inst.epsilon_min,
            epsilon_decay=inst.epsilon_decay,
            decision_interval=inst.decision_interval,
            min_green=inst.min_green,
            max_green=inst.max_green,
            yellow_duration=inst.yellow_duration,
            all_red_duration=inst.all_red_duration,
        )

        return inst
