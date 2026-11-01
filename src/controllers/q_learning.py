"""Bộ điều khiển đèn giao thông Tabular Q-Learning theo đặc tả đã duyệt.

Bao gồm:
  1. Mã hóa trạng thái rời rạc thu gọn: s = (p_curr, t_bin, q_serv, q_comp)
  2. Không gian hành động nhị phân: a in {0: KEEP, 1: SWITCH}
  3. Lọc hành động hợp lệ (Action Masking) theo ràng buộc an toàn (20s - 45s - 90s)
  4. Lựa chọn hành động theo chính sách epsilon-greedy
  5. Cập nhật phương trình Bellman Q-Learning:
     Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_{a'} Q(s', a') - Q(s, a)]
  6. Lưu và tải Q-table đầy đủ siêu tham số để tái lập 100%.
"""

import ast
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.controllers.fixed_time import build_all_red_state, build_yellow_state


class QLearningController:
    """Bộ điều khiển tín hiệu đèn giao thông Tabular Q-Learning."""

    def __init__(
        self,
        green_phases: List[str],
        movements: List[Dict[str, Any]],
        decision_interval: float = 5.0,
        min_green: float = 20.0,
        max_green: float = 45.0,
        yellow_time: float = 3.0,
        all_red_time: float = 1.0,
        max_red: float = 90.0,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        learning_enabled: bool = True,
        tls_id: Optional[str] = None,
        halting_count_fn: Optional[Callable[[str], int]] = None,
    ) -> None:
        """Khởi tạo bộ điều khiển Q-Learning.

        Tham số:
            green_phases: Danh sách các chuỗi trạng thái pha xanh hợp lệ.
            movements: Danh sách kết nối chuyển động tại nút giao.
            decision_interval: Chu kỳ ra quyết định (giây).
            min_green: Thời gian xanh tối thiểu (giây).
            max_green: Thời gian xanh tối đa (giây).
            yellow_time: Thời gian dọn đèn vàng (giây).
            all_red_time: Thời gian dọn đỏ toàn phần (giây).
            max_red: Ngưỡng thời gian chờ đỏ tối đa chống đói pha (giây).
            alpha: Tốc độ học (Learning Rate).
            gamma: Hệ số chiết khấu tương lai (Discount Factor).
            epsilon: Xác suất thăm dò ngẫu nhiên (Epsilon).
            learning_enabled: True nếu đang huấn luyện (cập nhật Q), False nếu đánh giá.
            tls_id: Mã định danh cụm đèn tín hiệu.
            halting_count_fn: Hàm lấy số lượng xe dừng trên làn (nếu None, dùng TraCI).
        """
        if not green_phases:
            raise ValueError("green_phases phải chứa ít nhất một pha xanh.")
        if min_green <= 0:
            raise ValueError(f"min_green phải dương, nhận được: {min_green}")
        if max_green < min_green:
            raise ValueError(f"max_green ({max_green}) phải >= min_green ({min_green})")

        self.green_phases: List[str] = list(green_phases)
        self.movements: List[Dict[str, Any]] = list(movements)
        self.decision_interval: float = float(decision_interval)
        self.min_green: float = float(min_green)
        self.max_green: float = float(max_green)
        self.yellow_time: float = float(yellow_time)
        self.all_red_time: float = float(all_red_time)
        self.max_red: float = float(max_red)
        self.alpha: float = float(alpha)
        self.gamma: float = float(gamma)
        self.epsilon: float = float(epsilon)
        self.learning_enabled: bool = bool(learning_enabled)
        self.tls_id: Optional[str] = tls_id
        self._halting_count_fn: Optional[Callable[[str], int]] = halting_count_fn

        # Bảng Q-Table: mapping từ state tuple (p_curr, t_bin, q_serv, q_comp) sang list [Q_keep, Q_switch]
        self.q_table: Dict[Tuple[int, int, int, int], List[float]] = {}

        # Trạng thái điều khiển thời gian thực
        self.current_phase_idx: int = 0
        self.next_phase_idx: int = 0
        self.time_in_phase: float = 0.0
        self.time_since_decision: float = 0.0
        self.is_yellow: bool = False
        self.is_all_red: bool = False
        self.step_count: int = 0

        # Bộ đếm thời gian chờ đỏ của các pha
        self.phase_red_timers: List[float] = [0.0] * len(self.green_phases)

        # Lưu chuyển tiếp gần nhất (s, a, interval_halting_sum, interval_steps) để cập nhật Q
        self.last_state: Optional[Tuple[int, int, int, int]] = None
        self.last_action: Optional[int] = None
        self.accumulated_queue_sum: float = 0.0
        self.accumulated_queue_steps: int = 0

        # Thống kê cập nhật
        self.update_count: int = 0

        # Ánh xạ làn vào cho từng pha xanh
        self._phase_incoming_lanes: List[List[str]] = []
        for p_state in self.green_phases:
            lanes_in_p = set()
            for m in self.movements:
                idx = m.get("link_index", -1)
                if 0 <= idx < len(p_state) and p_state[idx] in ("G", "g", "s"):
                    f_lane = m.get("from_lane", "")
                    if f_lane:
                        lanes_in_p.add(f_lane)
            self._phase_incoming_lanes.append(sorted(list(lanes_in_p)))

    @property
    def num_phases(self) -> int:
        return len(self.green_phases)

    def _get_lane_halting_count(self, lane_id: str) -> int:
        """Lấy số lượng xe dừng chờ trên một làn đường."""
        if not lane_id:
            return 0
        if self._halting_count_fn is not None:
            try:
                return int(self._halting_count_fn(lane_id))
            except Exception:
                return 0
        try:
            import traci
            if traci.isLoaded():
                return int(traci.lane.getLastStepHaltingNumber(lane_id))
        except Exception:
            pass
        return 0

    def calculate_phase_queues(self) -> Dict[int, int]:
        """Tính tổng số xe dừng chờ cho từng pha xanh."""
        queues: Dict[int, int] = {}
        for p_idx, lanes in enumerate(self._phase_incoming_lanes):
            total_q = sum(self._get_lane_halting_count(l) for l in lanes)
            queues[p_idx] = total_q
        return queues

    def discretize_state(self) -> Tuple[int, int, int, int]:
        """Mã hóa quan sát thực tế thành bộ tuple trạng thái 4 phần tử (s).

        Định dạng: (p_curr, t_bin, q_serving, q_competing)
        """
        p_curr = self.current_phase_idx

        # 1. t_bin: 0 nếu t < 35s, 1 nếu t >= 35s
        t_bin = 1 if self.time_in_phase >= 35.0 else 0

        # 2. q_serving & q_competing
        phase_queues = self.calculate_phase_queues()
        q_serv_val = phase_queues.get(p_curr, 0)
        q_serv = 1 if q_serv_val >= 3 else 0

        competing_queues = [q for p, q in phase_queues.items() if p != p_curr]
        max_competing = max(competing_queues) if competing_queues else 0
        q_comp = 1 if max_competing >= 6 else 0

        return (p_curr, t_bin, q_serv, q_comp)

    def get_valid_actions(self, time_in_phase: Optional[float] = None) -> List[int]:
        """Xác định danh sách các hành động hợp lệ tuân thủ các ràng buộc an toàn giao thông.

        Hành động 0: KEEP (giữ pha)
        Hành động 1: SWITCH (đổi pha kế tiếp)
        """
        t = self.time_in_phase if time_in_phase is None else float(time_in_phase)

        # 1. Chưa đủ thời gian xanh tối thiểu -> bắt buộc KEEP
        if t < self.min_green - 1e-6:
            return [0]

        # 2. Có pha bị đói đèn đỏ >= 90s -> bắt buộc SWITCH
        has_starved = any(
            p != self.current_phase_idx and self.phase_red_timers[p] >= self.max_red - 1e-6
            for p in range(self.num_phases)
        )
        if has_starved:
            return [1]

        # 3. Đã chạm trần thời gian xanh tối đa -> bắt buộc SWITCH
        if t >= self.max_green - 1e-6:
            return [1]

        # 4. Khoảng thời gian tự do ra quyết định [20s, 45s)
        return [0, 1]

    def _get_q_values(self, state: Tuple[int, int, int, int]) -> List[float]:
        """Lấy giá trị Q-values cho một trạng thái, khởi tạo [0.0, 0.0] nếu chưa gặp."""
        if state not in self.q_table:
            self.q_table[state] = [0.0, 0.0]
        return self.q_table[state]

    def choose_action(self, state: Tuple[int, int, int, int]) -> int:
        """Lựa chọn hành động theo chính sách epsilon-greedy kết hợp Action Masking."""
        valid_actions = self.get_valid_actions()
        if len(valid_actions) == 1:
            return valid_actions[0]

        # Thăm dò ngẫu nhiên (Exploration) nếu trong chế độ học và trúng xác suất epsilon
        if self.learning_enabled and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Khai thác giá trị Q lớn nhất (Exploitation)
        q_vals = self._get_q_values(state)
        # Chỉ lấy argmax trên các hành động hợp lệ
        best_action = max(valid_actions, key=lambda a: q_vals[a])
        return best_action

    def update_q_value(
        self,
        state: Tuple[int, int, int, int],
        action: int,
        reward: float,
        next_state: Tuple[int, int, int, int],
        next_valid_actions: List[int],
    ) -> float:
        """Cập nhật giá trị Q theo phương trình Bellman:

        Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_{a'} Q(s', a') - Q(s, a)]
        """
        old_q = self._get_q_values(state)[action]

        # Max Q tương lai chỉ xét trên các hành động hợp lệ tại next_state
        next_q_vals = self._get_q_values(next_state)
        if next_valid_actions:
            max_next_q = max(next_q_vals[a] for a in next_valid_actions)
        else:
            max_next_q = 0.0

        td_target = reward + self.gamma * max_next_q
        td_error = td_target - old_q
        new_q = old_q + self.alpha * td_error

        self.q_table[state][action] = round(new_q, 4)
        self.update_count += 1
        return td_error

    def get_current_state(self) -> str:
        """Lấy chuỗi trạng thái tín hiệu đèn hiện tại (tương thích giao diện controller)."""
        return self.get_current_state_signal()

    def get_current_state_signal(self) -> str:
        """Lấy chuỗi trạng thái tín hiệu đèn hiện tại."""
        curr_green = self.green_phases[self.current_phase_idx]
        if self.is_yellow:
            next_green = self.green_phases[self.next_phase_idx]
            return build_yellow_state(curr_green, next_green)
        if self.is_all_red:
            return build_all_red_state(curr_green)
        return curr_green

    def step(self, step_duration: float = 1.0) -> Tuple[str, bool, Optional[str], Optional[Dict[str, Any]]]:
        """Tiến bộ điều khiển thêm một bước thời gian step_duration giây."""
        event_name: Optional[str] = None
        phase_changed: bool = False
        decision_info: Optional[Dict[str, Any]] = None

        # 1. Cập nhật bộ đếm thời gian đỏ
        for p in range(self.num_phases):
            if p == self.current_phase_idx and not (self.is_yellow or self.is_all_red):
                self.phase_red_timers[p] = 0.0
            else:
                self.phase_red_timers[p] += step_duration

        # 2. Xử lý dọn nút giao đèn vàng
        if self.is_yellow:
            if self.time_in_phase >= self.yellow_time - 1e-6:
                self.is_yellow = False
                self.time_in_phase = 0.0
                if self.all_red_time > 0:
                    self.is_all_red = True
                    event_name = "enter_all_red"
                else:
                    self.current_phase_idx = self.next_phase_idx
                    self.time_since_decision = 0.0
                    event_name = "enter_green"
                phase_changed = True

        # 3. Xử lý dọn nút giao đỏ toàn phần
        elif self.is_all_red:
            if self.time_in_phase >= self.all_red_time - 1e-6:
                self.is_all_red = False
                self.current_phase_idx = self.next_phase_idx
                self.time_in_phase = 0.0
                self.time_since_decision = 0.0
                event_name = "enter_green"
                phase_changed = True

        # 4. Đang trong pha xanh: ra quyết định RL định kỳ mỗi decision_interval giây
        else:
            self.time_since_decision += step_duration
            is_decision_step = (
                self.time_since_decision >= self.decision_interval - 1e-6
                and self.time_in_phase >= self.min_green - 1e-6
            ) or (self.time_in_phase >= self.max_green - 1e-6)

            if is_decision_step:
                current_state = self.discretize_state()

                # Nếu trước đó đã có một cặp (last_state, last_action), thực hiện cập nhật Q
                if self.learning_enabled and self.last_state is not None and self.last_action is not None:
                    # Tính reward là âm hàng đợi trung bình tích lũy trong khoảng vừa qua
                    avg_q = (
                        (self.accumulated_queue_sum / self.accumulated_queue_steps)
                        if self.accumulated_queue_steps > 0 else 0.0
                    )
                    reward = -round(avg_q, 2)

                    valid_next = self.get_valid_actions()
                    self.update_q_value(
                        state=self.last_state,
                        action=self.last_action,
                        reward=reward,
                        next_state=current_state,
                        next_valid_actions=valid_next,
                    )

                # Chọn hành động cho bước tiếp theo
                action = self.choose_action(current_state)

                decision_info = {
                    "state": current_state,
                    "action": action,
                    "action_name": "KEEP" if action == 0 else "SWITCH",
                    "time_in_phase": self.time_in_phase,
                    "q_values": list(self._get_q_values(current_state)),
                }

                # Lưu trạng thái và đặt lại bộ đếm tích lũy reward
                self.last_state = current_state
                self.last_action = action
                self.accumulated_queue_sum = 0.0
                self.accumulated_queue_steps = 0
                self.time_since_decision = 0.0

                # Thực thi hành động
                if action == 1:  # SWITCH
                    self.next_phase_idx = (self.current_phase_idx + 1) % self.num_phases
                    self.time_in_phase = 0.0
                    if self.yellow_time > 0:
                        self.is_yellow = True
                        event_name = "enter_yellow"
                    elif self.all_red_time > 0:
                        self.is_all_red = True
                        event_name = "enter_all_red"
                    else:
                        self.current_phase_idx = self.next_phase_idx
                        event_name = "enter_green"
                    phase_changed = True

        # Tích lũy hàng đợi tức thời để tính reward cho bước tiếp theo
        current_step_queue = sum(self.calculate_phase_queues().values())
        self.accumulated_queue_sum += current_step_queue
        self.accumulated_queue_steps += 1

        current_signal = self.get_current_state_signal()
        self.time_in_phase += step_duration
        self.step_count += 1

        return current_signal, phase_changed, event_name, decision_info

    def save(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Lưu toàn bộ Q-Table cùng cấu hình và siêu tham số vào file JSON."""
        abs_path = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        # Chuyển key tuple sang string để tương thích với JSON chuẩn
        serializable_q = {str(k): [round(v, 4) for v in vals] for k, vals in self.q_table.items()}

        data: Dict[str, Any] = {
            "algorithm": "Tabular_Q_Learning",
            "hyperparameters": {
                "alpha": self.alpha,
                "gamma": self.gamma,
                "epsilon": self.epsilon,
                "min_green": self.min_green,
                "max_green": self.max_green,
                "yellow_time": self.yellow_time,
                "all_red_time": self.all_red_time,
                "max_red": self.max_red,
                "decision_interval": self.decision_interval,
            },
            "num_states_visited": len(self.q_table),
            "total_updates": self.update_count,
            "metadata": metadata or {},
            "q_table": serializable_q,
        }

        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load(self, filepath: str) -> None:
        """Tải Q-Table và siêu tham số từ file JSON."""
        abs_path = os.path.abspath(filepath)
        if not os.path.isfile(abs_path):
            raise FileNotFoundError(f"Không tìm thấy file model: {abs_path}")

        with open(abs_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        params = data.get("hyperparameters", {})
        if "alpha" in params:
            self.alpha = float(params["alpha"])
        if "gamma" in params:
            self.gamma = float(params["gamma"])
        if "epsilon" in params:
            self.epsilon = float(params["epsilon"])

        # Phục hồi Q-table từ string key sang tuple
        parsed_q: Dict[Tuple[int, int, int, int], List[float]] = {}
        for k_str, vals in data.get("q_table", {}).items():
            try:
                parsed_key = tuple(int(x) for x in ast.literal_eval(k_str))
                parsed_q[parsed_key] = [float(v) for v in vals]
            except Exception:
                continue

        self.q_table = parsed_q
        self.update_count = int(data.get("total_updates", 0))

    def reset(self) -> None:
        """Đặt lại bộ điều khiển cho một episode mới."""
        self.current_phase_idx = 0
        self.next_phase_idx = 0
        self.time_in_phase = 0.0
        self.time_since_decision = 0.0
        self.is_yellow = False
        self.is_all_red = False
        self.step_count = 0
        self.phase_red_timers = [0.0] * len(self.green_phases)
        self.last_state = None
        self.last_action = None
        self.accumulated_queue_sum = 0.0
        self.accumulated_queue_steps = 0
