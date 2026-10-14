"""Tabular Q-Learning Traffic Signal Controller."""

import json
import os
import pickle
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from src.controllers.base import BaseTrafficController, IntersectionObservation
from src.rl.discretization import StateDiscretizer


class QLearningController(BaseTrafficController):
    """Tác tử Tabular Q-Learning dùng cho điều khiển tín hiệu đèn giao thông.

    Công thức cập nhật:
        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_{a'} Q(s', a') - Q(s, a)]
        Đối với các trạng thái kết thúc (done=True):
        Q(s, a) <- Q(s, a) + alpha * [r - Q(s, a)]
    """


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
        yellow_duration: float = 3.0,
        all_red_duration: float = 0.0,
        seed: Optional[int] = None,
        eval_mode: bool = False,
        tie_break_strategy: str = "random",
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
        self.discretizer = discretizer if discretizer is not None else StateDiscretizer()
        self.target_tls_id = target_tls_id
        self.decision_interval = float(decision_interval)
        self.min_green = float(min_green)
        self.yellow_duration = float(yellow_duration)
        self.all_red_duration = float(all_red_duration)
        self.eval_mode = bool(eval_mode)
        self.tie_break_strategy = str(tie_break_strategy).lower()

        if seed is not None:
            self.set_seed(seed)

        # Q-table: dictionary mapping discrete state tuple -> np.ndarray of shape (num_actions,)
        self.q_table: Dict[Tuple[int, int, int, int], np.ndarray] = {}

        # Tracking metrics
        self.total_updates: int = 0
        self.action_counts: np.ndarray = np.zeros(self.num_actions, dtype=int)
        self.episode_rewards: List[float] = []

    def set_seed(self, seed: int) -> None:
        """Sets random seed for reproducible exploration."""
        random.seed(seed)
        np.random.seed(seed)

    def reset(self) -> None:
        """Resets transient execution state."""
        self.history.clear()

    def get_q_values(self, state: Tuple[int, int, int, int]) -> np.ndarray:
        """Retrieves or initializes Q-values for a given discrete state."""
        if state not in self.q_table:
            if getattr(self, "eval_mode", False):
                return np.zeros(self.num_actions, dtype=np.float64)
            self.q_table[state] = np.zeros(self.num_actions, dtype=np.float64)
        return self.q_table[state]

    def select_action_for_state(
        self,
        state: Tuple[int, int, int, int],
        valid_actions: Optional[Sequence[int]] = None,
    ) -> int:
        """Selects an action using epsilon-greedy exploration respecting valid actions.

        Args:
            state: Discrete state tuple.
            valid_actions: Optional list of eligible actions. If None, all actions are valid.

        Returns:
            Selected action index.
        """
        if valid_actions is None or len(valid_actions) == 0:
            candidates = list(range(self.num_actions))
        else:
            candidates = [a for a in valid_actions if 0 <= a < self.num_actions]
            if not candidates:
                candidates = list(range(self.num_actions))

        # In evaluation mode or when random check fails, exploit argmax Q
        if self.eval_mode or random.random() >= self.epsilon:
            q_vals = self.get_q_values(state)
            candidate_q = [q_vals[a] for a in candidates]
            max_q = max(candidate_q)
            # Break ties among candidates achieving maximum Q
            best_candidates = [
                a for a, q in zip(candidates, candidate_q) if abs(q - max_q) <= 1e-8
            ]
            if self.eval_mode and self.tie_break_strategy == "first":
                chosen_action = best_candidates[0]
            else:
                chosen_action = random.choice(best_candidates)
        else:
            # Explore: uniform random selection among valid actions
            chosen_action = random.choice(candidates)

        self.action_counts[chosen_action] += 1
        return chosen_action

    def update(
        self,
        state: Tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int],
        done: bool = False,
        valid_next_actions: Optional[Sequence[int]] = None,
        terminated: Optional[bool] = None,
        truncated: Optional[bool] = None,
    ) -> float:
        """Applies Bellman Q-learning update.

        Args:
            state: Current discrete state.
            action: Action taken.
            reward: Immediate scalar reward.
            next_state: Next discrete state.
            done: Whether episode terminated or truncated (for backward compatibility).
            valid_next_actions: Valid actions available in next_state.
            terminated: True if episode ended at true terminal absorbing state.
            truncated: True if episode ended due to time limit / horizon.

        Returns:
            Absolute Bellman TD error.
        """
        if self.eval_mode:
            return 0.0

        q_current = self.get_q_values(state)

        # True terminal state cuts off bootstrapping; time-limit truncation continues bootstrapping
        is_true_terminal = terminated if terminated is not None else done

        if is_true_terminal:
            target = float(reward)
        else:
            next_q = self.get_q_values(next_state)
            if valid_next_actions is not None and len(valid_next_actions) > 0:
                valid_candidates = [a for a in valid_next_actions if 0 <= a < self.num_actions]
                max_next_q = max([next_q[a] for a in valid_candidates]) if valid_candidates else 0.0
            else:
                max_next_q = float(np.max(next_q))

            target = float(reward + self.gamma * max_next_q)

        td_error = target - q_current[action]
        q_current[action] += self.learning_rate * td_error
        self.total_updates += 1

        return abs(td_error)

    def decay_epsilon(self) -> float:
        """Decays exploration rate epsilon by decay factor down to epsilon_min."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return self.epsilon

    def select_action(
        self, observations: Dict[str, IntersectionObservation]
    ) -> Dict[str, int]:
        """BaseTrafficController interface method: maps observations to target actions."""
        actions: Dict[str, int] = {}
        for tls_id, obs in observations.items():
            if self.target_tls_id is not None and tls_id != self.target_tls_id:
                continue

            state = self.discretizer.discretize(obs)

            # Determine valid actions from observation constraints
            if obs.is_in_transition:
                valid_actions = [obs.current_phase]
            elif obs.time_in_phase < self.discretizer.min_green:
                valid_actions = [obs.current_phase]
            else:
                valid_actions = list(range(len(obs.green_phases)))

            act = self.select_action_for_state(state, valid_actions=valid_actions)
            actions[tls_id] = act

            self.log_decision(
                tls_id=tls_id,
                sim_time=obs.sim_time,
                step=obs.step,
                action=act,
                details={
                    "state": state,
                    "epsilon": self.epsilon,
                    "eval_mode": self.eval_mode,
                },
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
            "yellow_duration": self.yellow_duration,
            "all_red_duration": self.all_red_duration,
            "tie_break_strategy": self.tie_break_strategy,
            "num_states_in_table": len(self.q_table),
            "discretizer_config": self.discretizer.get_config(),
        }

    def save(self, filepath: str) -> None:
        """Saves Q-table and metadata to disk."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        # Convert Q-table tuple keys to string keys for JSON serialization, or use pickle
        serializable_data = {
            "config": self.get_config(),
            "q_table": {str(k): v.tolist() for k, v in self.q_table.items()},
            "action_counts": self.action_counts.tolist(),
            "total_updates": self.total_updates,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Loads Q-table and configuration from disk."""
        if not os.path.isfile(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        cfg = data["config"]
        self.num_actions = cfg["num_actions"]
        self.learning_rate = cfg["learning_rate"]
        self.gamma = cfg["gamma"]
        self.epsilon = cfg["epsilon"]
        self.epsilon_min = cfg["epsilon_min"]
        self.epsilon_decay = cfg["epsilon_decay"]
        self.decision_interval = cfg.get("decision_interval", getattr(self, "decision_interval", 5.0))
        self.min_green = cfg.get("min_green", getattr(self, "min_green", 5.0))
        self.yellow_duration = cfg.get("yellow_duration", getattr(self, "yellow_duration", 3.0))
        self.all_red_duration = cfg.get("all_red_duration", getattr(self, "all_red_duration", 0.0))
        self.tie_break_strategy = cfg.get("tie_break_strategy", getattr(self, "tie_break_strategy", "random"))

        # Restore discretizer if config present
        if "discretizer_config" in cfg:
            d_cfg = cfg["discretizer_config"]
            self.discretizer = StateDiscretizer(
                min_green=d_cfg.get("min_green", 5.0),
                max_effective_green=d_cfg.get("max_effective_green", 25.0),
                serving_queue_thresholds=tuple(d_cfg.get("serving_thresholds", [1.0, 5.0])),
                waiting_queue_thresholds=tuple(d_cfg.get("waiting_thresholds", [1.0, 6.0, 15.0])),
            )

        if "target_tls_id" in cfg and self.target_tls_id is None:
            self.target_tls_id = cfg["target_tls_id"]

        # Restore Q-table
        self.q_table.clear()
        for k_str, v_list in data["q_table"].items():
            # Parse tuple from string representation, e.g. "(0, 1, 2, 1)"
            cleaned = k_str.strip("() ")
            k_tuple = tuple(int(x.strip()) for x in cleaned.split(","))
            self.q_table[k_tuple] = np.array(v_list, dtype=np.float64)

        if "action_counts" in data:
            self.action_counts = np.array(data["action_counts"], dtype=int)
        if "total_updates" in data:
            self.total_updates = data["total_updates"]
