"""Tabular Q-Learning Traffic Signal Controller with Compact State Discretization.

Optimized to minimize waiting time, travel time, queue length, and time loss,
while maximizing network throughput and enforcing safety bounds:
  - min_green (default 20s): prevents rapid phase flickering
  - yellow_time (default 3s): clearance interval between opposing green phases
  - max_green (default 45s): prevents movement starvation
"""

import ast
import bisect
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.controllers.base import BaseController, register_controller

__all__ = ["CompactStateDiscretizer", "QLearningController"]


class CompactStateDiscretizer:
    """Discretizes continuous intersection observations into compact 4-element discrete state tuples.

    Produces a compact 4-element state tuple:
        (curr_phase, serving_q_bin, max_competing_phase, competing_q_bin)

    By eliminating redundant time bins (handled safely by min_green/max_green bounds) and
    simplifying queue pressure into binary indicators, the state space is reduced
    from 4,096 to ~256 states, allowing rapid and stable Q-table convergence.
    """

    def __init__(
        self,
        serving_threshold: float = 2.0,
        competing_threshold: float = 6.0,
        time_thresholds: Optional[Sequence[float]] = None,
        queue_thresholds: Optional[Sequence[float]] = None,
    ) -> None:
        if queue_thresholds is not None and len(queue_thresholds) >= 2:
            self.serving_threshold = float(queue_thresholds[0])
            self.competing_threshold = float(queue_thresholds[1])
        elif queue_thresholds is not None and len(queue_thresholds) == 1:
            self.serving_threshold = float(queue_thresholds[0])
            self.competing_threshold = float(competing_threshold)
        else:
            self.serving_threshold = float(serving_threshold)
            self.competing_threshold = float(competing_threshold)
        self.time_thresholds = tuple(float(x) for x in time_thresholds) if time_thresholds else ()

    def discretize(
        self,
        curr_phase: int,
        time_in_phase: float,
        phase_queues: Dict[int, float],
    ) -> Tuple[int, int, int, int]:
        """Discretizes current traffic condition into a compact 4-element tuple."""
        serving_q = float(phase_queues.get(curr_phase, 0.0))
        serving_bin = 0 if serving_q < self.serving_threshold else 1

        competing_phases = [p for p in phase_queues.keys() if p != curr_phase]
        if competing_phases:
            strongest_competing = max(competing_phases, key=lambda p: phase_queues.get(p, 0.0))
        else:
            strongest_competing = curr_phase

        competing_q = float(phase_queues.get(strongest_competing, 0.0))
        competing_bin = 0 if competing_q < self.competing_threshold else 1

        return (int(curr_phase), int(serving_bin), int(strongest_competing), int(competing_bin))


@register_controller("q_learning")
class QLearningController(BaseController):
    """Tabular Q-Learning Traffic Signal Controller with traffic safety bounds."""

    def __init__(
        self,
        green_phases: Sequence[Any],
        movements: Sequence[Any],
        decision_interval: float = 5.0,
        min_green: float = 20.0,
        max_green: float = 45.0,
        yellow_time: float = 3.0,
        all_red_time: float = 1.0,
        max_red: float = 90.0,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        tls_id: Optional[str] = None,
        lane_vehicle_fn: Optional[Callable[[str], int]] = None,
        discretizer: Optional[CompactStateDiscretizer] = None,
    ) -> None:
        if min_green <= 0:
            raise ValueError(f"min_green must be positive, got {min_green}.")
        if max_green < min_green:
            raise ValueError(f"max_green ({max_green}) must be >= min_green ({min_green}).")

        self.movements: List[Any] = list(movements) if movements is not None else []
        self.decision_interval: float = float(decision_interval)
        self.min_green: float = float(min_green)
        self.max_green: float = float(max_green)
        self.lr: float = float(lr)
        self.gamma: float = float(gamma)
        self.epsilon: float = float(epsilon)
        self.discretizer: CompactStateDiscretizer = discretizer or CompactStateDiscretizer()

        super().__init__(
            green_phases=green_phases,
            yellow_time=yellow_time,
            all_red_time=all_red_time,
            max_red=max_red,
            tls_id=tls_id,
            lane_vehicle_fn=lane_vehicle_fn,
        )

        # Precompute incoming lanes per green phase
        self._phase_incoming_lanes: List[List[str]] = []
        for p_state in self.green_phases:
            lanes: List[str] = []
            for m in self.movements:
                idx = getattr(m, "link_index", None) if not isinstance(m, dict) else m.get("link_index")
                if idx is not None and 0 <= idx < len(p_state):
                    if p_state[idx] in ("G", "g"):
                        from_l = getattr(m, "from_lane", "") if not isinstance(m, dict) else m.get("from_lane", "")
                        if from_l and from_l not in lanes:
                            lanes.append(from_l)
            self._phase_incoming_lanes.append(lanes)

        # Q-table: Dict[Tuple[int, ...], List[float]]
        self.q_table: Dict[Tuple[int, ...], List[float]] = {}
        self.action_counts: Dict[Tuple[Tuple[int, ...], int], int] = {}
        self.explore_mode: bool = False

    def reset(self) -> None:
        """Resets dynamic execution state without wiping learned Q-table."""
        super().reset()
        self._time_since_last_decision: float = 0.0
        self.last_decision: Optional[Tuple[Tuple[int, ...], int]] = None

    def calculate_phase_queues(self) -> Dict[int, float]:
        """Calculates total vehicle queue/demand per green phase."""
        queues: Dict[int, float] = {}
        for p_idx, lanes in enumerate(self._phase_incoming_lanes):
            total_q = sum(self._get_vehicle_count(l) for l in lanes)
            queues[p_idx] = float(total_q)
        return queues

    def get_state(self) -> Tuple[int, int, int, int]:
        """Observes the current intersection state and discretizes it."""
        phase_queues = self.calculate_phase_queues()
        return self.discretizer.discretize(
            curr_phase=self._current_phase_idx,
            time_in_phase=self._time_in_phase,
            phase_queues=phase_queues,
        )

    def _get_q_entry(self, state: Tuple[int, ...]) -> List[float]:
        try:
            return self.q_table[state]
        except KeyError:
            val = [0.0] * self.num_phases
            self.q_table[state] = val
            return val

    def choose_action(self, state: Tuple[int, ...], explore: bool = False) -> int:
        """Selects next action according to epsilon-greedy policy with action masking."""
        queues = self.calculate_phase_queues()
        curr_q = queues.get(self._current_phase_idx, 0.0)

        # Exploration (epsilon-greedy)
        if explore and random.random() < self.epsilon:
            eligible = [p for p in range(self.num_phases) if p == self._current_phase_idx or queues.get(p, 0.0) > 0]
            if not eligible:
                eligible = list(range(self.num_phases))
            # 50% random exploration, 50% count-guided exploration to discover under-visited actions
            if random.random() < 0.5:
                return random.choice(eligible)
            return min(eligible, key=lambda a: self.action_counts.get((state, a), 0))

        q_vals = self._get_q_entry(state)

        # In greedy mode: determine eligible actions via action masking
        eligible_actions: List[int] = []
        for a in range(self.num_phases):
            if a == self._current_phase_idx:
                eligible_actions.append(a)
            elif queues.get(a, 0.0) > 0 or curr_q == 0:
                eligible_actions.append(a)

        if not eligible_actions:
            eligible_actions = [self._current_phase_idx]

        best_score = -float("inf")
        best_actions: List[int] = []

        for a in eligible_actions:
            q = q_vals[a]
            count = self.action_counts.get((state, a), 0)
            adjusted_q = q
            if count == 0 and q == 0.0:
                if a != self._current_phase_idx:
                    adjusted_q = -float("inf") if curr_q > 0 else -1e-3

            if adjusted_q > best_score:
                best_score = adjusted_q
                best_actions = [a]
            elif adjusted_q == best_score:
                best_actions.append(a)

        # Tie-breaker: prioritize keeping current phase to minimize yellow clearance delay
        if self._current_phase_idx in best_actions:
            return self._current_phase_idx

        # If still tied, pick action with highest queue
        return max(best_actions, key=lambda a: queues.get(a, 0.0))

    def update_q(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
    ) -> float:
        """Applies Bellman Q-Learning update: Q(s, a) <- Q + lr * [r + gamma * max Q(s', a') - Q]."""
        q_vals = self._get_q_entry(state)
        next_q_vals = self._get_q_entry(next_state)
        target = float(reward) + self.gamma * max(next_q_vals)
        td_error = target - q_vals[action]
        q_vals[action] += self.lr * td_error
        self.action_counts[(state, action)] = self.action_counts.get((state, action), 0) + 1
        return td_error

    def step(self, step_duration: float = 1.0) -> Tuple[str, bool]:
        """Advances controller state by step_duration seconds."""
        step_duration = float(step_duration)
        self.last_decision = None

        if self._is_yellow:
            self._time_in_phase += step_duration
            self._total_time += step_duration
            self._step_count += 1
            self._update_red_timers(step_duration)
            if self._time_in_phase >= self.yellow_time - 1e-6:
                self._time_in_phase = 0.0
                self._is_yellow = False
                if self.all_red_time > 0:
                    self._is_all_red = True
                    return self.get_current_state(), True
                else:
                    self._is_all_red = False
                    self._current_phase_idx = self._next_phase_idx
                    self._time_since_last_decision = 0.0
                    return self.get_current_state(), True
            return self.get_current_state(), False

        if self._is_all_red:
            self._time_in_phase += step_duration
            self._total_time += step_duration
            self._step_count += 1
            self._update_red_timers(step_duration)
            if self._time_in_phase >= self.all_red_time - 1e-6:
                self._time_in_phase = 0.0
                self._is_all_red = False
                self._current_phase_idx = self._next_phase_idx
                self._time_since_last_decision = 0.0
                return self.get_current_state(), True
            return self.get_current_state(), False

        # In green phase
        self._time_in_phase += step_duration
        self._total_time += step_duration
        self._step_count += 1
        self._time_since_last_decision += step_duration
        self._update_red_timers(step_duration)

        # 1. Check max_red constraint (Starvation Watchdog)
        if self._time_in_phase >= self.min_green - 1e-6:
            starved = [
                p for p in range(self.num_phases)
                if p != self._current_phase_idx and self.get_phase_red_time(p) >= self.max_red - 1e-6
            ]
            if starved:
                queues = self.calculate_phase_queues()
                chosen = max(starved, key=lambda p: (self.get_phase_red_time(p), queues.get(p, 0.0)))
                state = self.get_state()
                self.last_decision = (state, chosen)
                self._time_since_last_decision = 0.0
                if chosen != self._current_phase_idx:
                    self._next_phase_idx = chosen
                    self._time_in_phase = 0.0
                    if self.yellow_time > 0:
                        self._is_yellow = True
                    elif self.all_red_time > 0:
                        self._is_all_red = True
                    else:
                        self._current_phase_idx = chosen
                    return self.get_current_state(), True

        # 2. Check max_green constraint (forced switch)
        if self._time_in_phase >= self.max_green - 1e-6:
            state = self.get_state()
            alt_actions = [p for p in range(self.num_phases) if p != self._current_phase_idx]
            if alt_actions:
                q_vals = self._get_q_entry(state)
                chosen = max(alt_actions, key=lambda a: q_vals[a])
            else:
                chosen = self._current_phase_idx

            self.last_decision = (state, chosen)
            self._time_since_last_decision = 0.0

            if chosen != self._current_phase_idx:
                self._next_phase_idx = chosen
                self._time_in_phase = 0.0
                if self.yellow_time > 0:
                    self._is_yellow = True
                elif self.all_red_time > 0:
                    self._is_all_red = True
                else:
                    self._current_phase_idx = chosen
                return self.get_current_state(), True

        # 3. Check RL decision interval (only allowed when time_in_phase >= min_green)
        if (
            self._time_since_last_decision >= self.decision_interval - 1e-6
            and self._time_in_phase >= self.min_green - 1e-6
        ):
            state = self.get_state()
            chosen = self.choose_action(state, explore=self.explore_mode)
            self._time_since_last_decision = 0.0
            self.last_decision = (state, chosen)

            if chosen != self._current_phase_idx:
                self._next_phase_idx = chosen
                self._time_in_phase = 0.0
                if self.yellow_time > 0:
                    self._is_yellow = True
                elif self.all_red_time > 0:
                    self._is_all_red = True
                else:
                    self._current_phase_idx = chosen
                return self.get_current_state(), True

        return self.get_current_state(), False

    def save(self, filepath: str) -> None:
        """Serializes learned Q-table and hyperparameters to a JSON file."""
        abs_path = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        data = {
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "min_green": self.min_green,
            "max_green": self.max_green,
            "yellow_time": self.yellow_time,
            "all_red_time": self.all_red_time,
            "max_red": self.max_red,
            "decision_interval": self.decision_interval,
            "q_table": {str(k): v for k, v in self.q_table.items()},
            "action_counts": {str(k): v for k, v in self.action_counts.items()},
        }
        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str) -> None:
        """Loads learned Q-table and hyperparameters from a JSON file."""
        abs_path = os.path.abspath(filepath)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Model file not found: {abs_path}")

        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.lr = float(data.get("lr", self.lr))
        self.gamma = float(data.get("gamma", self.gamma))
        self.epsilon = float(data.get("epsilon", self.epsilon))
        self.min_green = float(data.get("min_green", self.min_green))
        self.max_green = float(data.get("max_green", self.max_green))
        self.yellow_time = float(data.get("yellow_time", self.yellow_time))
        self.all_red_time = float(data.get("all_red_time", self.all_red_time))
        self.max_red = float(data.get("max_red", self.max_red))
        self.decision_interval = float(data.get("decision_interval", self.decision_interval))

        raw_q = data.get("q_table", {})
        parsed_q: Dict[Tuple[int, ...], List[float]] = {}
        for k_str, vals in raw_q.items():
            try:
                k_tuple = ast.literal_eval(k_str)
                parsed_q[tuple(int(x) for x in k_tuple)] = [float(v) for v in vals]
            except Exception:
                pass
        self.q_table = parsed_q

        raw_counts = data.get("action_counts", {})
        parsed_counts: Dict[Tuple[Tuple[int, ...], int], int] = {}
        for k_str, c_val in raw_counts.items():
            try:
                k_parsed = ast.literal_eval(k_str)
                state_tuple = tuple(int(x) for x in k_parsed[0])
                act_int = int(k_parsed[1])
                parsed_counts[(state_tuple, act_int)] = int(c_val)
            except Exception:
                pass
        self.action_counts = parsed_counts
