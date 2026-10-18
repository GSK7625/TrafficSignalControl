"""Tabular Q-Learning Traffic Signal Controller."""

import json
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.controllers.base import BaseTrafficController, IntersectionObservation
from src.rl.discretization import StateDiscretizer


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
        min_green: float = 5.0,
        max_green: Optional[float] = 60.0,
        yellow_duration: float = 3.0,
        all_red_duration: float = 1.0,
        seed: Optional[int] = None,
        eval_mode: bool = False,
        tie_break_strategy: str = "first",
    ):
        super().__init__(name="QLearning")
        if num_actions <= 0:
            raise ValueError(f"num_actions must be positive, got {num_actions}")
        if not (0.0 < learning_rate <= 1.0):
            raise ValueError(f"learning_rate must be in (0, 1], got {learning_rate}")
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")

        self.num_actions = int(num_actions)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.discretizer = discretizer or StateDiscretizer()
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
        candidates = [a for a in valid_actions if 0 <= a < self.num_actions] if valid_actions else list(range(self.num_actions))
        if not candidates:
            candidates = list(range(self.num_actions))

        if not self.eval_mode and random.random() < self.epsilon:
            chosen = random.choice(candidates)
        else:
            q_vals = self.get_q_values(state)
            cand_q = [q_vals[a] for a in candidates]
            max_q = max(cand_q)
            best = [a for a, q in zip(candidates, cand_q) if abs(q - max_q) <= 1e-8]
            if current_phase is not None and current_phase in best:
                chosen = current_phase
            elif self.tie_break_strategy == "random" and not self.eval_mode:
                chosen = random.choice(best)
            else:
                chosen = best[0]

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
            cand = [a for a in valid_next_actions if 0 <= a < self.num_actions] if valid_next_actions else range(self.num_actions)
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
                state, current_phase=obs.current_phase
            )
        return actions

    def get_config(self) -> Dict[str, Any]:
        """Returns controller configuration dictionary."""
        return {
            "name": self.name,
            "num_actions": self.num_actions,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "eval_mode": self.eval_mode,
            "target_tls_id": self.target_tls_id,
            "decision_interval": self.decision_interval,
            "min_green": self.min_green,
            "max_green": self.max_green,
            "yellow_duration": self.yellow_duration,
            "all_red_duration": self.all_red_duration,
            "tie_break_strategy": self.tie_break_strategy,
            "num_states_in_table": len(self.q_table),
            "discretizer_config": self.discretizer.get_config() if hasattr(self.discretizer, "get_config") else {},
        }

    def save(self, filepath: str) -> None:
        """Saves Q-table and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        data = {
            "config": self.get_config(),
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "action_counts": self.action_counts.tolist(),
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

        cfg = data.get("config", {})
        for attr in (
            "num_actions", "learning_rate", "gamma", "epsilon", "epsilon_min",
            "epsilon_decay", "decision_interval", "min_green", "yellow_duration",
            "all_red_duration", "tie_break_strategy", "target_tls_id"
        ):
            if attr in cfg and cfg[attr] is not None:
                setattr(self, attr, type(getattr(self, attr, cfg[attr]))(cfg[attr]))
        if "max_green" in cfg:
            self.max_green = float(cfg["max_green"]) if cfg["max_green"] is not None else None

        if "discretizer_config" in cfg and hasattr(self, "discretizer"):
            d_cfg = cfg["discretizer_config"]
            self.discretizer = StateDiscretizer(
                min_green=d_cfg.get("min_green", 5.0),
                max_effective_green=d_cfg.get("max_effective_green", 25.0),
                serving_queue_thresholds=tuple(d_cfg.get("serving_thresholds", [1.0, 5.0])),
                waiting_queue_thresholds=tuple(d_cfg.get("waiting_thresholds", [1.0, 6.0, 15.0])),
            )

        self.q_table.clear()
        for k_str, v_list in data.get("q_table", {}).items():
            k_tuple = tuple(int(x.strip()) for x in k_str.strip("() ").split(","))
            self.q_table[k_tuple] = np.array(v_list, dtype=np.float64)

        if "action_counts" in data:
            self.action_counts = np.array(data["action_counts"], dtype=int)
        if "total_updates" in data:
            self.total_updates = data["total_updates"]
