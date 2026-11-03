"""Bộ điều khiển đèn giao thông Deep Q-Network (DQN) - Kiến trúc PressLight / MPLight.

Tuân thủ thiết kế và quy tắc AGENT.md:
  1. Biểu diễn hàm giá trị Q bằng mạng neural MLP (Online Network và Target Network).
  2. Không gian hành động lựa chọn pha trực tiếp: a in {0, 1, ..., N-1}.
  3. Phần thưởng áp suất âm (Negative Pressure - PressLight): r_t = - sum_m |w_m| / dt.
  4. Experience Replay Buffer lưu trữ và lấy mẫu ngẫu nhiên mini-batch.
  5. Action Masking cưỡng chế ràng buộc an toàn (min_green = 20s, max_green = 45s, max_red = 90s)
     kết hợp cơ chế chống đói pha linh hoạt (chỉ can thiệp khi thực sự có xe chờ quá 90s).
  6. Smooth L1 (Huber Loss) và Optimizer Adam.
  7. Đồng bộ Target Network định kỳ (Periodic Hard Update).
  8. Tương thích 100% với hạ tầng mô phỏng, chuyển pha (vàng 3s + đỏ 1s) và metrics dùng chung.
"""

import collections
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.controllers.fixed_time import build_all_red_state, build_yellow_state


class DQNNetwork(nn.Module):
    """Mạng nơ-ron MLP nhỏ xấp xỉ Q(s, a).

    Kiến trúc: Linear(state_dim -> 64) -> ReLU -> Linear(64 -> 64) -> ReLU -> Linear(64 -> action_dim)
    """

    def __init__(self, state_dim: int, action_dim: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Lan truyền thuận tính Q-values cho từng hành động."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayBuffer:
    """Bộ đệm trải nghiệm (Experience Replay Buffer) lưu trữ các transition."""

    def __init__(self, capacity: int = 10000):
        self.capacity = int(capacity)
        self.buffer = collections.deque(maxlen=self.capacity)

    def clear(self) -> None:
        """Xóa toàn bộ trải nghiệm trong bộ đệm."""
        self.buffer.clear()

    def push(
        self,
        state: List[float],
        action: int,
        reward: float,
        next_state: List[float],
        terminated: bool,
        valid_actions_next: List[int],
        gamma: float = 0.95,
    ) -> None:
        """Thêm một transition mới vào hàng đợi."""
        self.buffer.append(
            (state, action, reward, next_state, terminated, valid_actions_next, float(gamma))
        )

    def sample(
        self, batch_size: int
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[List[int]],
    ]:
        """Lấy mẫu ngẫu nhiên đồng đều một mini-batch."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, terminateds, valid_actions_next, gammas = zip(
            *batch
        )

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(terminateds, dtype=torch.float32),
            torch.tensor(gammas, dtype=torch.float32),
            list(valid_actions_next),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNController:
    """Bộ điều khiển đèn giao thông sử dụng mạng Deep Q-Network."""

    def __init__(
        self,
        green_phases: List[str],
        movements: List[Dict[str, Any]],
        state_dim: int,
        q_max: float,
        decision_interval: float = 5.0,
        min_green: float = 20.0,
        max_green: float = 45.0,
        yellow_time: float = 3.0,
        all_red_time: float = 1.0,
        max_red: float = 90.0,
        gamma: float = 0.95,
        lr: float = 0.0005,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9,
        buffer_capacity: int = 10000,
        batch_size: int = 32,
        min_replay_size: int = 200,
        target_update_freq: int = 200,
        learning_enabled: bool = True,
        beta_wait: float = 0.05,
        switch_cost: float = 1.0,
        halting_count_fn: Optional[Callable[[str], int]] = None,
        vehicle_count_fn: Optional[Callable[[str], int]] = None,
        device: str = "cpu",
        per_phase_queues: Optional[bool] = None,
    ):
        self.green_phases = list(green_phases)
        self.num_phases = len(green_phases)
        self.movements = list(movements)
        self.state_dim = state_dim
        self.q_max = float(q_max)
        self.beta_wait = float(beta_wait)
        self.switch_cost = float(switch_cost)

        # Cấu hình biểu diễn trạng thái hàng đợi từng pha
        if per_phase_queues is not None:
            self.per_phase_queues = bool(per_phase_queues)
        else:
            self.per_phase_queues = (state_dim == 2 * self.num_phases + 1)

        # Định danh model checkpoint để đối chuẩn tái lập
        self.model_path: Optional[str] = None
        self.model_hash: Optional[str] = None
        self.model_metadata: Dict[str, Any] = {}

        # Ràng buộc thời gian và an toàn
        self.decision_interval = float(decision_interval)
        self.min_green = float(min_green)
        self.max_green = float(max_green)
        self.yellow_time = float(yellow_time)
        self.all_red_time = float(all_red_time)
        self.max_red = float(max_red)

        # Siêu tham số học tăng cường
        self.gamma = float(gamma)
        self.lr = float(lr)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = int(batch_size)
        self.min_replay_size = int(min_replay_size)
        self.target_update_freq = int(target_update_freq)
        self.learning_enabled = bool(learning_enabled)
        self._halting_count_fn = halting_count_fn
        self._vehicle_count_fn = vehicle_count_fn
        self.device = torch.device(device)

        # Ánh xạ làn đường vào cho từng pha xanh
        self._phase_incoming_lanes: List[List[str]] = []
        for p_str in self.green_phases:
            lanes_in_p = set()
            for m in self.movements:
                idx = m.get("link_index")
                f_lane = m.get("from_lane")
                if idx is not None and f_lane and idx < len(p_str):
                    if p_str[idx] in ("G", "g", "s"):
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

        # Tiền tính toán bậc cho các làn vào và làn ra để chuẩn hóa áp suất
        self._from_lane_degrees: Dict[str, int] = {}
        self._to_lane_degrees: Dict[str, int] = {}
        for m in self.movements:
            u = m.get("from_lane", "")
            d = m.get("to_lane", "")
            if u:
                self._from_lane_degrees[u] = self._from_lane_degrees.get(u, 0) + 1
            if d:
                self._to_lane_degrees[d] = self._to_lane_degrees.get(d, 0) + 1

        # Khởi tạo Online Network và Target Network với action_dim = num_phases (Phase Selection)
        self.online_net = DQNNetwork(state_dim=self.state_dim, action_dim=self.num_phases).to(self.device)
        self.target_net = DQNNetwork(state_dim=self.state_dim, action_dim=self.num_phases).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

        # Optimizer và Loss Function
        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss()  # Huber Loss

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Thống kê huấn luyện
        self.update_count = 0
        self.step_count = 0
        self.recent_losses: List[float] = []

        # Trạng thái điều khiển đèn nội bộ
        self.current_phase_idx = 0
        self.time_in_phase = 0.0
        self.is_yellow = False
        self.is_all_red = False
        self.next_phase_idx = 0
        self.phase_red_timers: List[float] = [0.0] * self.num_phases

        # Quản lý quyết định và phần thưởng theo chu kỳ decision_interval
        self.time_since_decision = 0.0
        self.accumulated_reward = 0.0
        self.accumulated_queue_sum = 0.0
        self.accumulated_queue_steps = 0
        self.last_state: Optional[List[float]] = None
        self.last_action: Optional[int] = None

    def reset(self, clear_buffer: bool = False) -> None:
        """Đặt lại trạng thái nội bộ trước mỗi lượt chạy (episode)."""
        self.current_phase_idx = 0
        self.time_in_phase = 0.0
        self.is_yellow = False
        self.is_all_red = False
        self.next_phase_idx = 0
        self.phase_red_timers = [0.0] * self.num_phases
        self.time_since_decision = 0.0
        self.accumulated_reward = 0.0
        self.accumulated_queue_sum = 0.0
        self.accumulated_queue_steps = 0
        self.last_state = None
        self.last_action = None
        self.step_count = 0
        if clear_buffer:
            self.replay_buffer.clear()

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

    def calculate_total_intersection_queue(self) -> int:
        """Tính tổng số xe dừng chờ thực tế trên toàn bộ các làn vào của nút giao (không trùng lặp)."""
        return sum(self._get_lane_halting_count(l) for l in self._all_incoming_lanes)

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

    def get_valid_actions(self, time_in_phase: Optional[float] = None) -> List[int]:
        """Xác định danh sách các hành động an toàn và hợp lệ (Phase Selection a in {0, ..., N-1})."""
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

    def encode_state(self) -> List[float]:
        """Mã hóa vector trạng thái liên tục và chuẩn hóa về đoạn [0.0, 1.0]."""
        p_curr = self.current_phase_idx

        # 1. One-hot encoding của pha hiện tại (P chiều)
        phase_one_hot = [0.0] * self.num_phases
        phase_one_hot[p_curr] = 1.0

        # 2. Thời gian xanh đã chuẩn hóa
        time_norm = min(1.0, round(self.time_in_phase / self.max_green, 4))

        # 3. Hàng đợi xe dừng pha phục vụ và các pha cạnh tranh
        phase_queues = self.calculate_phase_queues()

        if self.per_phase_queues:
            # Nhận diện hàng đợi riêng biệt từng pha (P chiều):
            # Cung cấp đầy đủ thông tin pha đối thủ nào đang cần phục vụ,
            # khắc phục triệt để việc triệt tiêu thông tin khi chỉ tính tổng q_comp.
            q_phase_norms = [
                min(1.0, round(phase_queues.get(p, 0) / self.q_max, 4))
                for p in range(self.num_phases)
            ]
            state_vector = phase_one_hot + [time_norm] + q_phase_norms
        else:
            # Chế độ kế thừa cũ (Legacy): q_serv và tổng hàng đợi các pha cạnh tranh
            q_serv_raw = phase_queues.get(p_curr, 0)
            competing_queues = [q for p, q in phase_queues.items() if p != p_curr]
            q_comp_raw = sum(competing_queues)

            q_serv_norm = min(1.0, round(q_serv_raw / self.q_max, 4))
            q_comp_norm = min(1.0, round(q_comp_raw / self.q_max, 4))

            state_vector = phase_one_hot + [time_norm, q_serv_norm, q_comp_norm]

        return state_vector

    def choose_action(
        self, state_vector: List[float], valid_actions: List[int]
    ) -> int:
        """Lựa chọn hành động bằng epsilon-greedy có kết hợp Action Masking."""
        if len(valid_actions) == 1:
            return valid_actions[0]

        # 1. Khám phá (Exploration)
        if self.learning_enabled and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # 2. Khai thác (Exploitation): dự đoán bằng Online Network
        self.online_net.eval()
        with torch.no_grad():
            state_tensor = torch.tensor(
                state_vector, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            q_values = self.online_net(state_tensor).squeeze(0)  # shape (num_phases,)

            # Áp dụng Action Masking: gán giá trị âm vô cùng cho action không hợp lệ
            for a in range(self.num_phases):
                if a not in valid_actions:
                    q_values[a] = -1e9

            action = int(torch.argmax(q_values).item())
            return action

    def update_network(self) -> Optional[float]:
        """Lấy mini-batch từ Replay Buffer, tính target Bellman và cập nhật Online Network."""
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        self.online_net.train()

        states, actions, rewards, next_states, terminateds, gammas, valid_actions_next = (
            self.replay_buffer.sample(self.batch_size)
        )
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terminateds = terminateds.to(self.device)
        gammas = gammas.to(self.device)

        # 1. Dự đoán Q(s, a; theta) từ Online Network
        q_predictions = self.online_net(states)  # (batch_size, num_phases)
        q_chosen = q_predictions.gather(1, actions.unsqueeze(1)).squeeze(1)

        # 2. Tính Target Bellman y = r + gamma_eff * (1 - terminated) * max_{a'} Q_target(s', a'; theta^-)
        with torch.no_grad():
            next_q_values = self.target_net(next_states)  # (batch_size, num_phases)

            # Áp dụng Action Masking trên trạng thái kế tiếp
            for i, valid_acts in enumerate(valid_actions_next):
                for a in range(self.num_phases):
                    if a not in valid_acts:
                        next_q_values[i, a] = -1e9

            max_next_q, _ = torch.max(next_q_values, dim=1)
            target = rewards + gammas * (1.0 - terminateds) * max_next_q

        # 3. Tính Huber Loss và tối ưu hóa
        loss = self.loss_fn(q_chosen, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_count += 1
        loss_val = float(loss.item())
        self.recent_losses.append(loss_val)
        if len(self.recent_losses) > 100:
            self.recent_losses.pop(0)

        # 4. Đồng bộ Target Network định kỳ (Hard update)
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss_val

    def get_current_state(self) -> str:
        """Lấy chuỗi trạng thái tín hiệu đèn hiện tại (tương thích runner)."""
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

        # Công thức DAP-Reward: kết hợp áp suất và thời gian chờ chuẩn hóa theo số làn vào
        pressure_term = float(step_pressure) / self.decision_interval
        waiting_term = (self.beta_wait * float(step_waiting_time)) / (self.decision_interval * n_lanes)
        step_reward = - (pressure_term + waiting_term)

        discount_factor = gamma_step ** self.accumulated_queue_steps

        self.accumulated_reward += discount_factor * step_reward
        self.accumulated_queue_sum += float(step_queue)
        self.accumulated_queue_steps += 1

    def step(
        self, step_duration: float = 1.0
    ) -> Tuple[str, bool, Optional[str], Optional[Dict[str, Any]]]:
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

        # 4. Đang trong pha xanh: ra quyết định định kỳ
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
                current_state = self.encode_state()
                valid_actions = self.get_valid_actions()

                last_reward = None
                last_duration = None

                # Cập nhật transition trước đó vào Replay Buffer (chỉ lưu khi learning_enabled=True)
                if self.learning_enabled and self.last_state is not None and self.last_action is not None and self.accumulated_queue_steps > 0:
                    dt = float(self.accumulated_queue_steps)
                    reward = round(self.accumulated_reward, 4)
                    last_reward = reward
                    last_duration = dt
                    eff_gamma = self.gamma ** (dt / self.decision_interval)

                    self.replay_buffer.push(
                        state=self.last_state,
                        action=self.last_action,
                        reward=reward,
                        next_state=current_state,
                        terminated=False,
                        valid_actions_next=valid_actions,
                        gamma=eff_gamma,
                    )
                    self.update_network()

                # Chọn hành động (Pha mục tiêu a in {0, ..., N-1})
                action = self.choose_action(current_state, valid_actions)

                # Phân tích nguyên nhân cưỡng chế Action Masking nếu chỉ có 1 hành động hợp lệ
                is_forced = (len(valid_actions) == 1)
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
                    "valid_actions": valid_actions,
                    "is_forced": is_forced,
                    "forced_reason": forced_reason,
                    "time_in_phase": self.time_in_phase,
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
          để không làm sai lệch Replay Buffer.
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
            eff_gamma = self.gamma ** (dt / self.decision_interval)
            final_state = self.encode_state()
            valid_next = self.get_valid_actions()
            if self.learning_enabled:
                # Time limit truncation: terminated=False để bootstrap giá trị tương lai
                self.replay_buffer.push(
                    state=self.last_state,
                    action=self.last_action,
                    reward=reward,
                    next_state=final_state,
                    terminated=False,
                    valid_actions_next=valid_next,
                    gamma=eff_gamma,
                )
                self.update_network()

        # Đặt lại các biến trạng thái kết thúc episode
        self.last_state = None
        self.last_action = None
        self.accumulated_reward = 0.0
        self.accumulated_queue_sum = 0.0
        self.accumulated_queue_steps = 0
        self.time_since_decision = 0.0

    def save(self, filepath: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Lưu toàn bộ trọng số mạng, cấu hình và metadata vào checkpoint."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "online_state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "update_count": self.update_count,
            "epsilon": self.epsilon,
            "config": {
                "state_dim": self.state_dim,
                "action_dim": self.num_phases,
                "q_max": self.q_max,
                "gamma": self.gamma,
                "lr": self.lr,
                "min_green": self.min_green,
                "max_green": self.max_green,
                "num_phases": self.num_phases,
                "decision_interval": self.decision_interval,
                "yellow_time": self.yellow_time,
                "all_red_time": self.all_red_time,
                "max_red": self.max_red,
                "per_phase_queues": self.per_phase_queues,
                "beta_wait": self.beta_wait,
                "switch_cost": self.switch_cost,
            },
            "metadata": metadata or {},
        }
        torch.save(checkpoint, filepath)

    def load(self, filepath: str) -> None:
        """Tải checkpoint và phục hồi đầy đủ siêu tham số cấu hình từ file."""
        self.model_path = filepath
        try:
            import hashlib
            with open(filepath, "rb") as f_bin:
                self.model_hash = hashlib.sha256(f_bin.read()).hexdigest()[:16]
        except Exception:
            self.model_hash = None

        checkpoint = torch.load(filepath, map_location=self.device)
        self.model_metadata = checkpoint.get("metadata", {})
        saved_cfg = checkpoint.get("config", {})
        if "beta_wait" in saved_cfg:
            self.beta_wait = float(saved_cfg["beta_wait"])
        if "switch_cost" in saved_cfg:
            self.switch_cost = float(saved_cfg["switch_cost"])
        self.online_net.load_state_dict(checkpoint["online_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_state_dict"])
        if "optimizer_state_dict" in checkpoint and self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = checkpoint.get("update_count", 0)
        self.epsilon = checkpoint.get("epsilon", self.epsilon_min)

        cfg = checkpoint.get("config", {})
        if "per_phase_queues" in cfg:
            self.per_phase_queues = bool(cfg["per_phase_queues"])
        elif "state_dim" in cfg:
            self.per_phase_queues = (int(cfg["state_dim"]) == 2 * self.num_phases + 1)
        if "min_green" in cfg:
            self.min_green = float(cfg["min_green"])
        if "max_green" in cfg:
            self.max_green = float(cfg["max_green"])
        if "decision_interval" in cfg:
            self.decision_interval = float(cfg["decision_interval"])
        if "yellow_time" in cfg:
            self.yellow_time = float(cfg["yellow_time"])
        if "all_red_time" in cfg:
            self.all_red_time = float(cfg["all_red_time"])
        if "max_red" in cfg:
            self.max_red = float(cfg["max_red"])
        if "gamma" in cfg:
            self.gamma = float(cfg["gamma"])
        if "q_max" in cfg:
            self.q_max = float(cfg["q_max"])

        self.target_net.eval()
        for param in self.target_net.parameters():
            param.requires_grad = False

