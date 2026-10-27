"""Bộ điều khiển tín hiệu đèn giao thông Tabular Q-Learning với không gian trạng thái rời rạc thu gọn.

Tối ưu hóa để giảm thiểu thời gian chờ, thời gian di chuyển, độ dài hàng đợi và tổn thất thời gian,
đồng thời tối đa hóa thông lượng mạng lưới và đảm bảo các ràng buộc an toàn giao thông:
  - min_green (mặc định 20s): ngăn chặn chuyển pha đèn quá nhanh gây mất an toàn
  - yellow_time (mặc định 3s): khoảng dọn nút giao giữa các pha xanh đối nghịch
  - max_green (mặc định 45s): ngăn chặn nghẽn đói các luồng giao thông đối diện
"""

import ast
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.controllers.base import BaseController, get_movement_attr, register_controller

__all__ = ["CompactStateDiscretizer", "QLearningController"]


class CompactStateDiscretizer:
    """Rời rạc hóa các quan sát liên tục tại nút giao thành bộ tuple 4 phần tử trạng thái thu gọn.

    Tạo bộ tuple trạng thái 4 phần tử:
        (curr_phase, serving_q_bin, max_competing_phase, competing_q_bin)
    """

    def __init__(
        self,
        serving_threshold: float = 2.0,
        competing_threshold: float = 6.0,
        time_thresholds: Optional[Sequence[float]] = None,
        queue_thresholds: Optional[Sequence[float]] = None,
    ) -> None:
        q_th = list(queue_thresholds) if queue_thresholds else []
        self.serving_threshold = float(q_th[0]) if q_th else float(serving_threshold)
        self.competing_threshold = float(q_th[1]) if len(q_th) >= 2 else float(competing_threshold)

    def discretize(
        self,
        curr_phase: int,
        time_in_phase: float,
        phase_queues: Dict[int, float],
    ) -> Tuple[int, int, int, int]:
        """Rời rạc hóa tình trạng giao thông hiện tại thành tuple 4 phần tử thu gọn."""
        serving_bin = int(phase_queues.get(curr_phase, 0.0) >= self.serving_threshold)
        competing_phases = [p for p in phase_queues if p != curr_phase]
        strongest = max(competing_phases, key=lambda p: phase_queues.get(p, 0.0)) if competing_phases else curr_phase
        competing_bin = int(phase_queues.get(strongest, 0.0) >= self.competing_threshold)
        return (int(curr_phase), serving_bin, int(strongest), competing_bin)


@register_controller("q_learning")
class QLearningController(BaseController):
    """Bộ điều khiển tín hiệu giao thông Tabular Q-Learning với các ràng buộc an toàn giao thông."""

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
        lane_halting_fn: Optional[Callable[[str], int]] = None,
        use_halting_for_state: bool = True,
    ) -> None:
        if min_green <= 0:
            raise ValueError(f"min_green must be positive, got {min_green}.")
        if max_green < min_green:
            raise ValueError(f"max_green ({max_green}) must be >= min_green ({min_green}).")

        super().__init__(
            green_phases=green_phases,
            yellow_time=yellow_time,
            all_red_time=all_red_time,
            max_red=max_red,
            tls_id=tls_id,
            lane_vehicle_fn=lane_vehicle_fn,
            lane_halting_fn=lane_halting_fn,
        )

        self.movements: List[Any] = list(movements) if movements is not None else []
        self.decision_interval: float = float(decision_interval)
        self.min_green: float = float(min_green)
        self.max_green: float = float(max_green)
        self.lr: float = float(lr)
        self.gamma: float = float(gamma)
        self.epsilon: float = float(epsilon)
        self.use_halting_for_state: bool = bool(use_halting_for_state)
        self.discretizer: CompactStateDiscretizer = CompactStateDiscretizer()

        # Tính toán trước các làn xe đi vào cho từng pha xanh
        self._phase_incoming_lanes: List[List[str]] = []
        for p_state in self.green_phases:
            lanes: List[str] = []
            for m in self.movements:
                idx = get_movement_attr(m, "link_index")
                if idx is not None and 0 <= idx < len(p_state) and p_state[idx] in ("G", "g"):
                    from_l = get_movement_attr(m, "from_lane", "")
                    if from_l and from_l not in lanes:
                        lanes.append(from_l)
            self._phase_incoming_lanes.append(lanes)

        self.q_table: Dict[Tuple[int, ...], List[float]] = {}
        self.action_counts: Dict[Tuple[Tuple[int, ...], int], int] = {}
        self.explore_mode: bool = False

    def reset(self) -> None:
        """Đặt lại trạng thái thực thi động của controller mà không xóa bảng Q-table đã học."""
        super().reset()
        self._time_since_last_decision: float = 0.0
        self.last_decision: Optional[Tuple[Tuple[int, ...], int]] = None

    def calculate_phase_queues(self) -> Dict[int, float]:
        """Tính toán tổng hàng đợi/nhu cầu xe dừng cho từng pha đèn xanh."""
        count_fn = self._get_halting_count if self.use_halting_for_state else self._get_vehicle_count
        return {
            p: float(sum(count_fn(l) for l in lanes))
            for p, lanes in enumerate(self._phase_incoming_lanes)
        }

    def get_state(self) -> Tuple[int, int, int, int]:
        """Quan sát trạng thái nút giao hiện tại và rời rạc hóa thành tuple."""
        return self.discretizer.discretize(
            curr_phase=self._current_phase_idx,
            time_in_phase=self._time_in_phase,
            phase_queues=self.calculate_phase_queues(),
        )

    def _get_q_entry(self, state: Tuple[int, ...]) -> List[float]:
        try:
            return self.q_table[state]
        except KeyError:
            val = [0.0] * self.num_phases
            self.q_table[state] = val
            return val

    def choose_action(self, state: Tuple[int, ...], explore: bool = False) -> int:
        """Lựa chọn hành động tiếp theo theo chính sách epsilon-greedy kết hợp mặt nạ hành động (Action Masking)."""
        queues = self.calculate_phase_queues()
        curr_q = queues.get(self._current_phase_idx, 0.0)

        # Chế độ khám phá (Exploration - epsilon-greedy)
        if explore and random.random() < self.epsilon:
            eligible = [p for p in range(self.num_phases) if p == self._current_phase_idx or queues.get(p, 0.0) > 0]
            if not eligible:
                eligible = list(range(self.num_phases))
            if random.random() < 0.5:
                return random.choice(eligible)
            return min(eligible, key=lambda a: self.action_counts.get((state, a), 0))

        # Chế độ khai thác (Greedy) kết hợp mặt nạ hành động (Action Masking)
        q_vals = self._get_q_entry(state)
        eligible = [
            a for a in range(self.num_phases)
            if a == self._current_phase_idx or queues.get(a, 0.0) > 0 or curr_q == 0
        ] or [self._current_phase_idx]

        def get_score(a: int) -> float:
            q = q_vals[a]
            if self.action_counts.get((state, a), 0) == 0 and q == 0.0 and a != self._current_phase_idx:
                return -float("inf") if curr_q > 0 else -1e-3
            return q

        best_score = max(get_score(a) for a in eligible)
        best_actions = [a for a in eligible if get_score(a) == best_score]

        # Phá vỡ thế hòa: Ưu tiên giữ pha hiện tại, sau đó chọn pha có hàng đợi lớn nhất
        if self._current_phase_idx in best_actions:
            return self._current_phase_idx
        return max(best_actions, key=lambda a: queues.get(a, 0.0))

    def update_q(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Optional[Tuple[int, ...]] = None,
        done: bool = False,
    ) -> float:
        """Cập nhật giá trị Q-Learning theo phương trình Bellman:
        - Nếu trạng thái tiếp theo chưa kết thúc (done=False): Q(s, a) <- Q + lr * [r + gamma * max Q(s', a') - Q]
        - Nếu là trạng thái kết thúc (done=True hoặc next_state is None): Q(s, a) <- Q + lr * [r - Q]
        """
        q_vals = self._get_q_entry(state)
        if done or next_state is None:
            target = float(reward)
        else:
            target = float(reward) + self.gamma * max(self._get_q_entry(next_state))
        td_error = target - q_vals[action]
        q_vals[action] += self.lr * td_error
        self.action_counts[(state, action)] = self.action_counts.get((state, action), 0) + 1
        return td_error

    def _switch_to_phase(self, chosen: int, state: Optional[Tuple[int, ...]] = None) -> Tuple[str, bool]:
        """Thực hiện chuyển pha an toàn sang pha đèn xanh mục tiêu và ghi nhận điểm quyết định."""
        self.last_decision = (state if state is not None else self.get_state(), chosen)
        self._time_since_last_decision = 0.0
        if chosen != self._current_phase_idx:
            self._trigger_switch(chosen)
            return self.get_current_state(), True
        return self.get_current_state(), False

    def step(self, step_duration: float = 1.0) -> Tuple[str, bool]:
        """Tiến bộ điều khiển thêm step_duration giây và trả về (trạng_thái_đèn, cờ_đổi_pha)."""
        step_duration = float(step_duration)
        self.last_decision = None
        self._time_in_phase += step_duration
        self._total_time += step_duration
        self._step_count += 1
        self._update_red_timers(step_duration)

        # 1. Khoảng thời gian dọn nút giao đèn vàng
        if self._is_yellow:
            if self._time_in_phase >= self.yellow_time - 1e-6:
                self._time_in_phase = 0.0
                self._is_yellow = False
                if self.all_red_time > 0:
                    self._is_all_red = True
                else:
                    self._current_phase_idx = self._next_phase_idx
                    self._time_since_last_decision = 0.0
                return self.get_current_state(), True
            return self.get_current_state(), False

        # 2. Khoảng thời gian dọn nút giao đỏ toàn phần
        if self._is_all_red:
            if self._time_in_phase >= self.all_red_time - 1e-6:
                self._time_in_phase = 0.0
                self._is_all_red = False
                self._current_phase_idx = self._next_phase_idx
                self._time_since_last_decision = 0.0
                return self.get_current_state(), True
            return self.get_current_state(), False

        # 3. Pha đèn xanh đang hoạt động
        self._time_since_last_decision += step_duration

        # Ưu tiên 1: Giám sát nghẽn đói pha (ràng buộc max_red)
        if self._time_in_phase >= self.min_green - 1e-6:
            starved = [
                p for p in range(self.num_phases)
                if p != self._current_phase_idx and self.get_phase_red_time(p) >= self.max_red - 1e-6
            ]
            if starved:
                queues = self.calculate_phase_queues()
                chosen = max(starved, key=lambda p: (self.get_phase_red_time(p), queues.get(p, 0.0)))
                return self._switch_to_phase(chosen)

        # Ưu tiên 2: Ràng buộc thời gian xanh tối đa (max_green)
        if self._time_in_phase >= self.max_green - 1e-6:
            state = self.get_state()
            alt = [p for p in range(self.num_phases) if p != self._current_phase_idx]
            chosen = max(alt, key=lambda a: self._get_q_entry(state)[a]) if alt else self._current_phase_idx
            return self._switch_to_phase(chosen, state)

        # Ưu tiên 3: Điểm quyết định định kỳ của thuật toán RL
        if (
            self._time_since_last_decision >= self.decision_interval - 1e-6
            and self._time_in_phase >= self.min_green - 1e-6
        ):
            state = self.get_state()
            chosen = self.choose_action(state, explore=self.explore_mode)
            return self._switch_to_phase(chosen, state)

        return self.get_current_state(), False

    def save(self, filepath: str) -> None:
        """Lưu bảng Q-table và siêu tham số đã học vào file JSON."""
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
        """Tải bảng Q-table và siêu tham số đã học từ file JSON."""
        abs_path = os.path.abspath(filepath)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Model file not found: {abs_path}")

        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for key in (
            "lr", "gamma", "epsilon", "min_green", "max_green",
            "yellow_time", "all_red_time", "max_red", "decision_interval",
        ):
            if key in data:
                setattr(self, key, float(data[key]))

        parsed_q: Dict[Tuple[int, ...], List[float]] = {}
        for k_str, vals in data.get("q_table", {}).items():
            try:
                parsed_q[tuple(int(x) for x in ast.literal_eval(k_str))] = [float(v) for v in vals]
            except Exception:
                pass
        self.q_table = parsed_q

        parsed_counts: Dict[Tuple[Tuple[int, ...], int], int] = {}
        for k_str, c_val in data.get("action_counts", {}).items():
            try:
                k_parsed = ast.literal_eval(k_str)
                state_tuple = tuple(int(x) for x in k_parsed[0])
                act_int = int(k_parsed[1])
                parsed_counts[(state_tuple, act_int)] = int(c_val)
            except Exception:
                pass
        self.action_counts = parsed_counts
