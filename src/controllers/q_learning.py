"""Bộ điều khiển đèn giao thông Tabular Q-Learning theo kiến trúc PressLight / MPLight.

Bao gồm:
  1. Mã hóa trạng thái rời rạc thu gọn: s = (p_curr, t_bin, q_serv, q_comp)
  2. Không gian hành động lựa chọn pha trực tiếp: a in {0, 1, ..., N-1}
  3. Phần thưởng áp suất âm (Negative Pressure - PressLight): r_t = - sum_m |w_m| / dt
  4. Lọc hành động hợp lệ (Action Masking) kết hợp cơ chế chống đói pha linh hoạt:
     - min_green (20s): Bắt buộc giữ pha hiện tại
     - max_green (45s): Bắt buộc đổi sang pha khác
     - max_red (90s): Chỉ can thiệp chống đói khi pha đó thực sự có xe đang chờ quá 90s
  5. Lựa chọn hành động theo chính sách epsilon-greedy trên các hành động hợp lệ
  6. Cập nhật phương trình Bellman Q-Learning (SMDP intra-option discounting)
  7. Lưu và tải Q-table đầy đủ siêu tham số để tái lập 100%.
"""

import ast
import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.controllers.fixed_time import build_all_red_state, build_yellow_state


class QLearningController:
    """Bộ điều khiển tín hiệu đèn giao thông Tabular Q-Learning với Negative Pressure."""

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
        beta_wait: float = 0.05,
        switch_cost: float = 1.0,
        tls_id: Optional[str] = None,
        halting_count_fn: Optional[Callable[[str], int]] = None,
        vehicle_count_fn: Optional[Callable[[str], int]] = None,
        include_comp_phase_id: bool = False,
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
            vehicle_count_fn: Hàm lấy số lượng xe trên làn (nếu None, dùng TraCI).
            include_comp_phase_id: Nếu True, state bổ sung ID của pha đối thủ có hàng đợi lớn nhất (5-tuple),
                                   giúp phân biệt chính xác pha nào đang cần phục vụ.
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
        self.beta_wait: float = float(beta_wait)
        self.switch_cost: float = float(switch_cost)
        self.tls_id: Optional[str] = tls_id
        self._halting_count_fn: Optional[Callable[[str], int]] = halting_count_fn
        self._vehicle_count_fn: Optional[Callable[[str], int]] = vehicle_count_fn
        self.include_comp_phase_id: bool = bool(include_comp_phase_id)

        # Định danh checkpoint nạp vào để đối chuẩn tái lập
        self.model_path: Optional[str] = None
        self.model_hash: Optional[str] = None
        self.model_metadata: Dict[str, Any] = {}

        # Bảng Q-Table: mapping từ state tuple sang list Q-values cho từng pha [Q_0, ..., Q_{N-1}]
        self.q_table: Dict[Tuple[int, ...], List[float]] = {}

        # Trạng thái điều khiển thời gian thực
        self.current_phase_idx: int = 0
        self.next_phase_idx: int = 0
        self.time_in_phase: float = 0.0
        self.time_since_decision: float = 0.0
        self.is_yellow: bool = False
        self.is_all_red: bool = False
        self.step_count: int = 0

        # Bộ đếm thời gian chờ đỏ của các pha (chỉ tích lũy khi có xe dừng chờ)
        self.phase_red_timers: List[float] = [0.0] * len(self.green_phases)

        # Lưu chuyển tiếp gần nhất (s, a, accumulated_reward, interval_steps) để cập nhật Q
        self.last_state: Optional[Tuple[int, int, int, int]] = None
        self.last_action: Optional[int] = None
        self.accumulated_reward: float = 0.0
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

        # Tập hợp tất cả các làn vào duy nhất của nút giao
        incoming_set = set()
        for m in self.movements:
            f_lane = m.get("from_lane", "")
            if f_lane:
                incoming_set.add(f_lane)
        if not incoming_set:
            for lanes in self._phase_incoming_lanes:
                incoming_set.update(lanes)
        self._all_incoming_lanes: List[str] = sorted(list(incoming_set))

        # Tiền tính toán bậc (degrees) cho các làn vào và làn ra để chuẩn hóa áp suất nút giao
        self._from_lane_degrees: Dict[str, int] = {}
        self._to_lane_degrees: Dict[str, int] = {}
        for m in self.movements:
            u = m.get("from_lane", "")
            d = m.get("to_lane", "")
            if u:
                self._from_lane_degrees[u] = self._from_lane_degrees.get(u, 0) + 1
            if d:
                self._to_lane_degrees[d] = self._to_lane_degrees.get(d, 0) + 1

    @property
    def num_phases(self) -> int:
        return len(self.green_phases)

    def calculate_total_intersection_queue(self) -> int:
        """Tính tổng số xe dừng chờ thực tế trên toàn bộ các làn vào của nút giao (không trùng lặp)."""
        return sum(self._get_lane_halting_count(l) for l in self._all_incoming_lanes)

    def _get_backend(self) -> Any:
        """Lấy backend mô phỏng hiện hành (libsumo hoặc traci)."""
        try:
            import libsumo
            if libsumo.isLoaded():
                return libsumo
        except Exception:
            pass
        try:
            import traci
            if traci.isLoaded():
                return traci
        except Exception:
            pass
        return None

    def _get_lane_halting_count(self, lane_id: str) -> int:
        """Lấy số lượng xe dừng chờ trên một làn đường qua hàm tùy biến hoặc backend mô phỏng."""
        if not lane_id:
            return 0
        if self._halting_count_fn is not None:
            return int(self._halting_count_fn(lane_id))
        if self._vehicle_count_fn is not None:
            return int(self._vehicle_count_fn(lane_id))

        backend = self._get_backend()
        if backend is not None:
            return int(backend.lane.getLastStepHaltingNumber(lane_id))

        return 0

    def _get_lane_vehicle_count(self, lane_id: str) -> int:
        """Lấy số lượng xe trên một làn đường qua hàm tùy biến hoặc backend mô phỏng."""
        if not lane_id:
            return 0
        if self._vehicle_count_fn is not None:
            return int(self._vehicle_count_fn(lane_id))
        if self._halting_count_fn is not None:
            return int(self._halting_count_fn(lane_id))

        backend = self._get_backend()
        if backend is not None:
            return int(backend.lane.getLastStepVehicleNumber(lane_id))

        return 0

    def calculate_phase_queues(self) -> Dict[int, int]:
        """Tính tổng số xe dừng chờ cho từng pha xanh."""
        queues: Dict[int, int] = {}
        for p_idx, lanes in enumerate(self._phase_incoming_lanes):
            total_q = sum(self._get_lane_halting_count(l) for l in lanes)
            queues[p_idx] = total_q
        return queues

    def calculate_intersection_pressure(self) -> float:
        """Tính tổng áp suất nút giao theo biến thể áp suất phân bổ theo bậc kết nối làn (Degree-Weighted Pressure),
        lấy cảm hứng từ lý thuyết PressLight (Wei et al. 2019) / MPLight:

        Pressure = sum_{m} |w_m|
        với w_m = (x_in / deg_in) - (x_out / deg_out) của từng chuyển động m.

        LƯU Ý VỀ TÍNH CHÍNH XÁC LÝ THUYẾT SO VỚI NGUYÊN BẢN:
        - Trong bài báo PressLight gốc (Wei et al., KDD 2019), công thức chuẩn hóa số xe x_l bằng
          sức chứa tối đa của làn c_l: w_m = (x_in / c_in) - (x_out / c_out).
        - Triển khai này sử dụng bậc kết nối (degrees: số movement dùng chung làn vào deg_in và làn ra deg_out)
          để phân bổ lượng xe, tránh tính trùng lặp số xe khi nhiều luồng rẽ cùng xuất phát từ một làn đường.
        - Đây là một biến thể thích nghi phục vụ nghiên cứu mô phỏng nút giao phức tạp, thay vì khẳng định
          triển khai nguyên bản 100% của PressLight.
        """
        if not self.movements:
            return float(self.calculate_total_intersection_queue())

        lane_counts: Dict[str, int] = {}

        def get_count(lane: str) -> int:
            if not lane:
                return 0
            if lane not in lane_counts:
                lane_counts[lane] = self._get_lane_vehicle_count(lane)
            return lane_counts[lane]

        total_pressure = 0.0
        for m in self.movements:
            u = m.get("from_lane", "")
            d = m.get("to_lane", "")
            u_deg = self._from_lane_degrees.get(u, 1)
            d_deg = self._to_lane_degrees.get(d, 1)
            u_cnt = get_count(u)
            d_cnt = get_count(d)

            # w_m = x_in - x_out theo từng chuyển động m (biến thể phân bổ theo bậc kết nối)
            w_m = (u_cnt / float(u_deg)) - (d_cnt / float(d_deg))
            total_pressure += abs(w_m)

        return round(total_pressure, 2)

    def discretize_state(self) -> Tuple[int, ...]:
        """Mã hóa quan sát thực tế thành bộ tuple trạng thái rời rạc (s).

        - Nếu include_comp_phase_id=False (mặc định):
            (p_curr, t_bin, q_serving, q_competing)
        - Nếu include_comp_phase_id=True:
            (p_curr, t_bin, q_serving, q_competing, max_comp_phase)
            trong đó max_comp_phase là ID của pha đối thủ đang có hàng đợi lớn nhất
            (-1 nếu mọi pha đối thủ đều không có hàng đợi). Điều này giải quyết triệt để
            hạn chế mất thông tin pha đối thủ nào đang cần phục vụ.
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

        if self.include_comp_phase_id:
            if max_competing == 0:
                max_comp_phase = -1
            else:
                competing_pairs = [(p, q) for p, q in phase_queues.items() if p != p_curr]
                max_comp_phase = max(competing_pairs, key=lambda item: (item[1], -item[0]))[0]
            return (p_curr, t_bin, q_serv, q_comp, max_comp_phase)

        return (p_curr, t_bin, q_serv, q_comp)

    def get_valid_actions(self, time_in_phase: Optional[float] = None) -> List[int]:
        """Xác định danh sách các hành động hợp lệ tuân thủ các ràng buộc an toàn giao thông.

        Không gian hành động Phase Selection: a in {0, 1, ..., N-1}
        - a == current_phase_idx: KEEP (giữ pha hiện tại)
        - a != current_phase_idx: SWITCH (chuyển trực tiếp sang pha a)
        """
        t = self.time_in_phase if time_in_phase is None else float(time_in_phase)

        # 1. Chưa đủ thời gian xanh tối thiểu -> bắt buộc KEEP pha hiện tại
        if t < self.min_green - 1e-6:
            return [self.current_phase_idx]

        # 2. Cơ chế giám sát chống đói pha linh hoạt:
        # Chỉ can thiệp nếu pha đó thực sự có xe đang chờ bị quá max_red
        phase_queues = self.calculate_phase_queues()
        starved_phases = [
            p
            for p in range(self.num_phases)
            if p != self.current_phase_idx
            and self.phase_red_timers[p] >= self.max_red - 1e-6
            and phase_queues.get(p, 0) > 0
        ]
        if starved_phases:
            return starved_phases

        # 3. Đã chạm trần thời gian xanh tối đa -> bắt buộc SWITCH sang pha khác
        if t >= self.max_green - 1e-6:
            return [p for p in range(self.num_phases) if p != self.current_phase_idx]

        # 4. Khoảng thời gian tự do ra quyết định [min_green, max_green): cho phép chọn bất kỳ pha nào
        return list(range(self.num_phases))

    def _get_q_values(self, state: Tuple[int, ...]) -> List[float]:
        """Lấy giá trị Q-values cho một trạng thái, khởi tạo [0.0] * num_phases nếu chưa gặp."""
        if state not in self.q_table:
            self.q_table[state] = [0.0] * self.num_phases
        return self.q_table[state]

    def choose_action(self, state: Tuple[int, ...]) -> int:
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
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        next_valid_actions: List[int],
        duration_steps: Optional[float] = None,
    ) -> float:
        """Cập nhật giá trị Q theo phương trình Bellman (hỗ trợ SMDP với thời lượng bước chuyển biến đổi):

        Q(s, a) <- Q(s, a) + alpha * [r + (gamma ** (dt / dt_0)) * max_{a'} Q(s', a') - Q(s, a)]
        """
        old_q = self._get_q_values(state)[action]

        # Max Q tương lai chỉ xét trên các hành động hợp lệ tại next_state
        next_q_vals = self._get_q_values(next_state)
        if next_valid_actions:
            max_next_q = max(next_q_vals[a] for a in next_valid_actions)
        else:
            max_next_q = 0.0

        # Chiết khấu SMDP lũy thừa theo thời gian trôi qua giữa hai bước quyết định liên tiếp
        if duration_steps is not None and self.decision_interval > 0:
            effective_gamma = self.gamma ** (float(duration_steps) / self.decision_interval)
        else:
            effective_gamma = self.gamma

        td_target = reward + effective_gamma * max_next_q
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
            next_green = self.green_phases[self.next_phase_idx]
            return build_all_red_state(curr_green, next_green)
        return curr_green

    def observe_step(
        self,
        step_queue: Optional[int] = None,
        step_pressure: Optional[float] = None,
        step_waiting_time: Optional[float] = None,
    ) -> None:
        """Thu thập quan sát và tích lũy phần thưởng SMDP dựa trên Delay-Aware Pressure (DAP-Reward):
            r_k = - [ (Pressure_k / decision_interval) + beta_wait * (WaitingTime_k / (decision_interval * N_lanes)) ]
            R <- R + (gamma_step ** k) * r_k
        với gamma_step = gamma ** (1.0 / decision_interval).
        """
        if step_pressure is None:
            if self.movements:
                step_pressure = self.calculate_intersection_pressure()
            elif step_queue is not None:
                step_pressure = float(step_queue)
            else:
                step_pressure = float(self.calculate_total_intersection_queue())

        if step_queue is None:
            step_queue = self.calculate_total_intersection_queue()

        if step_waiting_time is None:
            backend = self._get_backend()
            if backend is not None:
                try:
                    step_waiting_time = sum(
                        float(backend.lane.getWaitingTime(l)) for l in self._all_incoming_lanes
                    )
                except Exception:
                    step_waiting_time = 0.0
            else:
                step_waiting_time = 0.0

        n_lanes = max(1, len(self._all_incoming_lanes))
        gamma_step = self.gamma ** (1.0 / self.decision_interval) if self.decision_interval > 0 else 1.0

        # Công thức DAP-Reward chuẩn hóa
        pressure_term = float(step_pressure) / self.decision_interval
        waiting_term = (self.beta_wait * float(step_waiting_time)) / (self.decision_interval * n_lanes)
        step_reward = - (pressure_term + waiting_term)

        discount_factor = gamma_step ** self.accumulated_queue_steps

        self.accumulated_reward += discount_factor * step_reward
        self.accumulated_queue_sum += float(step_queue)
        self.accumulated_queue_steps += 1

    def step(self, step_duration: float = 1.0) -> Tuple[str, bool, Optional[str], Optional[Dict[str, Any]]]:
        """Tiến bộ điều khiển thêm một bước thời gian step_duration giây."""
        event_name: Optional[str] = None
        phase_changed: bool = False
        decision_info: Optional[Dict[str, Any]] = None

        # 1. Cập nhật bộ đếm thời gian đỏ có giám sát xe chờ chống đói pha linh hoạt
        phase_queues = self.calculate_phase_queues()
        for p in range(self.num_phases):
            if p == self.current_phase_idx and not (self.is_yellow or self.is_all_red):
                self.phase_red_timers[p] = 0.0
            else:
                # Tích lũy thời gian đỏ nếu pha đó có xe đang dừng chờ hoặc có xe tồn tại trên làn vào
                has_vehicles = phase_queues.get(p, 0) > 0
                if not has_vehicles:
                    lanes = self._phase_incoming_lanes[p] if p < len(self._phase_incoming_lanes) else []
                    has_vehicles = any(self._get_lane_vehicle_count(l) > 0 for l in lanes)

                if has_vehicles:
                    self.phase_red_timers[p] += step_duration
                else:
                    self.phase_red_timers[p] = 0.0

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
            starved_phases = [
                p
                for p in range(self.num_phases)
                if p != self.current_phase_idx
                and self.phase_red_timers[p] >= self.max_red - 1e-6
                and phase_queues.get(p, 0) > 0
            ]
            has_starved = len(starved_phases) > 0

            is_decision_step = (
                (
                    self.time_since_decision >= self.decision_interval - 1e-6
                    and self.time_in_phase >= self.min_green - 1e-6
                )
                or (self.time_in_phase >= self.max_green - 1e-6)
                or (has_starved and self.time_in_phase >= self.min_green - 1e-6)
            )

            if is_decision_step:
                current_state = self.discretize_state()
                valid_next = self.get_valid_actions()

                last_reward = None
                last_duration = None

                # Nếu trước đó đã có một cặp (last_state, last_action), thực hiện cập nhật Q (SMDP formulation)
                if self.learning_enabled and self.last_state is not None and self.last_action is not None and self.accumulated_queue_steps > 0:
                    dt = float(self.accumulated_queue_steps)
                    reward = round(self.accumulated_reward, 4)
                    last_reward = reward
                    last_duration = dt

                    self.update_q_value(
                        state=self.last_state,
                        action=self.last_action,
                        reward=reward,
                        next_state=current_state,
                        next_valid_actions=valid_next,
                        duration_steps=dt,
                    )

                # Chọn hành động (Pha mục tiêu a in {0, ..., N-1})
                action = self.choose_action(current_state)

                # Phân tích nguyên nhân cưỡng chế Action Masking nếu chỉ có 1 hành động hợp lệ
                is_forced = (len(valid_next) == 1)
                forced_reason = None
                if is_forced:
                    if self.time_in_phase < self.min_green - 1e-6:
                        forced_reason = "min_green"
                    elif starved_phases:
                        forced_reason = "max_red_starvation"
                    elif self.time_in_phase >= self.max_green - 1e-6:
                        forced_reason = "max_green"

                decision_info = {
                    "phase_index": self.current_phase_idx,
                    "state": current_state,
                    "action": action,
                    "target_phase": action,
                    "action_name": "KEEP" if action == self.current_phase_idx else "SWITCH",
                    "valid_actions": valid_next,
                    "is_forced": is_forced,
                    "forced_reason": forced_reason,
                    "time_in_phase": self.time_in_phase,
                    "q_values": list(self._get_q_values(current_state)),
                    "prev_reward": last_reward,
                    "prev_duration_steps": last_duration,
                }

                # Lưu trạng thái và đặt lại bộ đếm tích lũy reward
                self.last_state = current_state
                self.last_action = action
                is_switch = (action != self.current_phase_idx)
                self.accumulated_reward = - float(self.switch_cost) if is_switch else 0.0
                self.accumulated_queue_sum = 0.0
                self.accumulated_queue_steps = 0
                self.time_since_decision = 0.0

                # Thực thi hành động lựa chọn pha trực tiếp
                if action != self.current_phase_idx:  # SWITCH trực tiếp sang pha action
                    self.next_phase_idx = action
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

        current_signal = self.get_current_state_signal()
        # Bộ đếm thời gian nhất quán với các bước thực thi trong pha xanh
        if not self.is_yellow and not self.is_all_red:
            self.time_since_decision += step_duration
        self.time_in_phase += step_duration
        self.step_count += 1

        return current_signal, phase_changed, event_name, decision_info

    def finalize_episode(self) -> None:
        """Xử lý chuyển tiếp cuối cùng khi episode kết thúc (Time-limit Truncation).

        Xử lý chuẩn xác theo lý thuyết SMDP / Options:
        - Chỉ cập nhật khi action trước đó đã hoàn tất và kết thúc tại ranh giới quyết định hợp lệ
          (đèn xanh, time_in_phase >= min_green, time_since_decision >= decision_interval hoặc đã thu đủ số bước quyết định).
        - Nếu episode bị cắt ngang dở dang giữa một macro-action (đang đèn vàng, đỏ dọn nút, hoặc chưa đủ min_green),
          trạng thái kế tiếp s' không tồn tại trong không gian quyết định. Bỏ qua (discard) bước dở dang này
          để không làm sai lệch Q-table / Replay Buffer.
        """
        decision_time_satisfied = (
            self.time_since_decision >= self.decision_interval - 1e-6
            or self.accumulated_queue_steps >= int(round(self.decision_interval))
        )
        is_decision_boundary = (
            not self.is_yellow
            and not self.is_all_red
            and (self.time_in_phase >= self.min_green - 1e-6)
            and decision_time_satisfied
        )

        if is_decision_boundary and self.last_state is not None and self.last_action is not None and self.accumulated_queue_steps > 0:
            dt = float(self.accumulated_queue_steps)
            reward = round(self.accumulated_reward, 4)
            final_state = self.discretize_state()
            valid_next = self.get_valid_actions()
            if self.learning_enabled:
                self.update_q_value(
                    state=self.last_state,
                    action=self.last_action,
                    reward=reward,
                    next_state=final_state,
                    next_valid_actions=valid_next,
                    duration_steps=dt,
                )

        # Đặt lại các biến trạng thái kết thúc episode
        self.last_state = None
        self.last_action = None
        self.accumulated_reward = 0.0
        self.accumulated_queue_sum = 0.0
        self.accumulated_queue_steps = 0
        self.time_since_decision = 0.0

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
                "include_comp_phase_id": self.include_comp_phase_id,
                "beta_wait": self.beta_wait,
                "switch_cost": self.switch_cost,
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

        self.model_path = filepath
        self.model_metadata = data.get("metadata", {})
        try:
            import hashlib
            with open(abs_path, "rb") as f_bin:
                self.model_hash = hashlib.sha256(f_bin.read()).hexdigest()[:16]
        except Exception:
            self.model_hash = None

        params = data.get("hyperparameters", {})
        if "alpha" in params:
            self.alpha = float(params["alpha"])
        if "gamma" in params:
            self.gamma = float(params["gamma"])
        if "epsilon" in params:
            self.epsilon = float(params["epsilon"])
        if "min_green" in params:
            self.min_green = float(params["min_green"])
        if "max_green" in params:
            self.max_green = float(params["max_green"])
        if "yellow_time" in params:
            self.yellow_time = float(params["yellow_time"])
        if "all_red_time" in params:
            self.all_red_time = float(params["all_red_time"])
        if "max_red" in params:
            self.max_red = float(params["max_red"])
        if "decision_interval" in params:
            self.decision_interval = float(params["decision_interval"])
        if "include_comp_phase_id" in params:
            self.include_comp_phase_id = bool(params["include_comp_phase_id"])
        if "beta_wait" in params:
            self.beta_wait = float(params["beta_wait"])
        if "switch_cost" in params:
            self.switch_cost = float(params["switch_cost"])

        # Phục hồi Q-table từ string key sang tuple
        parsed_q: Dict[Tuple[int, ...], List[float]] = {}
        detected_key_len: Optional[int] = None
        for k_str, vals in data.get("q_table", {}).items():
            try:
                parsed_key = tuple(int(x) for x in ast.literal_eval(k_str))
                parsed_q[parsed_key] = [float(v) for v in vals]
                if detected_key_len is None:
                    detected_key_len = len(parsed_key)
            except Exception:
                continue

        self.q_table = parsed_q
        self.update_count = int(data.get("total_updates", 0))

        # Tự động đồng bộ include_comp_phase_id nếu file checkpoint chứa dạng 5-tuple
        if detected_key_len == 5:
            self.include_comp_phase_id = True
        elif detected_key_len == 4:
            self.include_comp_phase_id = False

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
        self.accumulated_reward = 0.0
        self.accumulated_queue_sum = 0.0
        self.accumulated_queue_steps = 0
