"""Kiểm thử thuật toán Tabular Q-Learning (QLearningController).

Bao gồm:
  1. Đối chiếu ví dụ tính tay cập nhật Q theo công thức Bellman.
  2. Mã hóa trạng thái rời rạc thu gọn (Discretization).
  3. Tính nhất quán khi lưu và tải Q-table từ JSON.
  4. Chế độ đánh giá (Evaluation mode): tắt khám phá, không cập nhật Q.
"""

import os
import tempfile
import pytest

from src.controllers.q_learning import QLearningController


def test_q_learning_hand_calculation():
    """Đối chiếu ví dụ tính tay từ đặc tả Giai đoạn 7:

    s = (0, 0, 0, 1), Q(s) = [-12.5, -8.0]
    Hành động: a = 1 (SWITCH)
    Phần thưởng: r = -4.0
    Trạng thái mới: s' = (1, 0, 1, 0), Q(s') = [-6.0, -10.0]
    => max Q(s') = -6.0
    => TD Target = -4.0 + 0.95 * (-6.0) = -9.70
    => TD Error delta = -9.70 - (-8.0) = -1.70
    => Q_new(s, 1) = -8.0 + 0.1 * (-1.70) = -8.17
    """
    controller = QLearningController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[{"link_index": 0, "from_lane": "l1", "to_lane": "l2"}],
        alpha=0.1,
        gamma=0.95,
        learning_enabled=True,
    )

    s = (0, 0, 0, 1)
    controller.q_table[s] = [-12.5, -8.0]

    s_prime = (1, 0, 1, 0)
    controller.q_table[s_prime] = [-6.0, -10.0]

    td_error = controller.update_q_value(
        state=s,
        action=1,
        reward=-4.0,
        next_state=s_prime,
        next_valid_actions=[0, 1],
    )

    # Đối chiếu delta và giá trị Q mới
    assert round(td_error, 2) == -1.70
    assert controller.q_table[s][1] == -8.17
    assert controller.update_count == 1


def test_state_discretization():
    """Kiểm tra logic rời rạc hóa trạng thái (Discretization)."""
    controller = QLearningController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[
            {"link_index": 0, "from_lane": "lane_0", "to_lane": "lane_out_0"},
            {"link_index": 3, "from_lane": "lane_1", "to_lane": "lane_out_1"},
        ],
        halting_count_fn=lambda lane: 2 if lane == "lane_0" else 8,
    )

    # 1. Tại t = 20s (< 35s), lane_0 có 2 xe (< 3), lane_1 có 8 xe (>= 6)
    controller.current_phase_idx = 0
    controller.time_in_phase = 20.0
    state = controller.discretize_state()
    # (p_curr=0, t_bin=0, q_serv=0, q_comp=1)
    assert state == (0, 0, 0, 1)

    # 2. Khi t = 36s (>= 35s)
    controller.time_in_phase = 36.0
    state = controller.discretize_state()
    assert state == (0, 1, 0, 1)


def test_save_load_consistency():
    """Kiểm tra tính nhất quán 100% khi lưu và tải Q-table từ JSON."""
    controller = QLearningController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[],
        alpha=0.15,
        gamma=0.90,
        epsilon=0.05,
        min_green=10.0,
        max_green=60.0,
        decision_interval=10.0,
        max_red=120.0,
    )

    s1 = (0, 0, 1, 1)
    s2 = (1, 1, 0, 0)
    controller.q_table[s1] = [-5.234, -2.111]
    controller.q_table[s2] = [-1.000, -8.765]
    controller.update_count = 42

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        temp_path = f.name

    try:
        controller.save(temp_path, metadata={"test": "ok"})

        # Khởi tạo một controller mới với tham số mặc định và nạp lại
        loaded_controller = QLearningController(
            green_phases=["GGGrrr", "rrrGGG"],
            movements=[],
        )
        loaded_controller.load(temp_path)

        assert loaded_controller.alpha == 0.15
        assert loaded_controller.gamma == 0.90
        assert loaded_controller.epsilon == 0.05
        assert loaded_controller.min_green == 10.0
        assert loaded_controller.max_green == 60.0
        assert loaded_controller.decision_interval == 10.0
        assert loaded_controller.max_red == 120.0
        assert loaded_controller.update_count == 42
        assert s1 in loaded_controller.q_table
        assert loaded_controller.q_table[s1] == [-5.234, -2.111]
        assert loaded_controller.q_table[s2] == [-1.000, -8.765]

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_evaluation_mode():
    """Kiểm tra chế độ đánh giá: tắt khám phá và không cập nhật Q-table."""
    controller = QLearningController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[],
        epsilon=0.0,
        learning_enabled=False,
    )

    unseen_state = (0, 0, 0, 0)
    controller.time_in_phase = 25.0

    # Lựa chọn hành động trên trạng thái chưa gặp
    action = controller.choose_action(unseen_state)
    # Cả 2 hành động có Q=0.0 -> mặc định chọn 0 (KEEP)
    assert action == 0

    # Kiểm tra số lần cập nhật Q vẫn bằng 0
    assert controller.update_count == 0


def test_reward_queue_counts_unique_incoming_lanes():
    """Kiểm tra reward không bị đếm lặp xe (double-counting) giữa các pha khi một làn thuộc nhiều pha."""
    green_phases = [
        "GGrr",  # Pha 0 bật link 0, 1 (lane_0, lane_1)
        "rGGr",  # Pha 1 bật link 1, 2 (lane_1, lane_2)
        "rrGG",  # Pha 2 bật link 2, 3 (lane_2, lane_3)
    ]
    movements = [
        {"link_index": 0, "from_lane": "lane_0"},
        {"link_index": 1, "from_lane": "lane_1"},
        {"link_index": 2, "from_lane": "lane_2"},
        {"link_index": 3, "from_lane": "lane_3"},
    ]
    # Mỗi làn đúng 1 xe dừng chờ -> tổng thực tế toàn nút = 4 xe
    controller = QLearningController(
        green_phases=green_phases,
        movements=movements,
        halting_count_fn=lambda l: 1,
    )
    # Phải bằng 4, KHÔNG được bằng 6 (do lane_1 và lane_2 bị đếm 2 lần trong phase queues)
    assert controller.calculate_total_intersection_queue() == 4


def test_finalize_episode_q_learning():
    """Kiểm tra finalize_episode(): chỉ cập nhật khi ở ranh giới quyết định hợp lệ, loại bỏ bước dở dang nếu bị cắt ngang giữa chừng."""
    controller = QLearningController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[],
        decision_interval=5.0,
        min_green=20.0,
        learning_enabled=True,
    )
    s = (0, 0, 0, 0)
    controller.last_state = s
    controller.last_action = 0
    controller.accumulated_reward = -4.5
    controller.accumulated_queue_sum = 25.0
    controller.accumulated_queue_steps = 5

    # Trường hợp 1: Episode bị cắt ngang khi đang đèn vàng -> bỏ qua bước dở dang
    controller.is_yellow = True
    controller.finalize_episode()
    assert controller.update_count == 0  # Không cập nhật bước dở dang
    assert controller.last_state is None

    # Trường hợp 2: Episode kết thúc ngay tại ranh giới quyết định (pha xanh, đủ min_green, đủ decision_interval)
    controller.last_state = s
    controller.last_action = 0
    controller.accumulated_reward = -4.5
    controller.accumulated_queue_steps = 5
    controller.is_yellow = False
    controller.is_all_red = False
    controller.time_in_phase = 20.0
    controller.time_since_decision = 5.0

    controller.finalize_episode()
    assert controller.update_count == 1  # Cập nhật thành công
    assert controller.last_state is None
    assert controller.accumulated_queue_steps == 0


def test_intra_option_discounting_calculation():
    """Đối chiếu công thức chiết khấu nội bộ SMDP (Sutton, Precup, Singh 1999):
    Hàng đợi không đổi Q = 10 xe trong 24 giây, r_k = -10/5 = -2, gamma = 0.95.
    gamma_step = 0.95^(1/5) ~ 0.9897935.
    Tổng chiết khấu: sum_{k=0}^{23} gamma_step^k * (-2) ~ -42.75.
    """
    controller = QLearningController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[],
        gamma=0.95,
        decision_interval=5.0,
    )
    for _ in range(24):
        controller.observe_step(step_queue=10)

    assert round(controller.accumulated_reward, 2) == -42.77
    assert controller.accumulated_queue_steps == 24


def test_sensor_exception_not_swallowed_q_learning():
    """Kiểm tra cảm biến gặp lỗi không bị nuốt âm thầm thành 0."""
    def broken_sensor(lane: str) -> int:
        raise RuntimeError("Cảm biến hỏng hoặc mất kết nối")

    controller = QLearningController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[{"link_index": 0, "from_lane": "lane_fail"}],
        halting_count_fn=broken_sensor,
    )
    with pytest.raises(RuntimeError, match="Cảm biến hỏng hoặc mất kết nối"):
        controller.calculate_total_intersection_queue()


def test_direct_phase_selection_q_learning():
    """Kiểm tra lựa chọn pha trực tiếp (Phase Selection):
    Cho phép nhảy thẳng từ Pha 0 sang Pha 2 (bỏ qua Pha 1) khi Q(s, 2) cao nhất.
    """
    green_phases = [
        "GGrrrr",  # Pha 0
        "rrGGrr",  # Pha 1
        "rrrrGG",  # Pha 2
    ]
    movements = [
        {"link_index": 0, "from_lane": "l0", "to_lane": "out0"},
        {"link_index": 2, "from_lane": "l1", "to_lane": "out1"},
        {"link_index": 4, "from_lane": "l2", "to_lane": "out2"},
    ]
    controller = QLearningController(
        green_phases=green_phases,
        movements=movements,
        min_green=20.0,
        max_green=45.0,
        yellow_time=3.0,
        all_red_time=1.0,
        epsilon=0.0,
        learning_enabled=False,
    )
    controller.current_phase_idx = 0
    controller.time_in_phase = 20.0
    controller.time_since_decision = 5.0

    state = controller.discretize_state()
    # Gán Q-values: Pha 2 có giá trị lớn nhất (15.0)
    controller.q_table[state] = [2.0, 1.0, 15.0]

    valid_actions = controller.get_valid_actions()
    assert valid_actions == [0, 1, 2]

    # Q-Learning chọn trực tiếp Pha 2
    chosen_action = controller.choose_action(state)
    assert chosen_action == 2

    # Chạy 1 bước: kích hoạt đèn vàng chuyển tiếp sang Pha 2
    state_sig, changed, event, dec = controller.step(1.0)
    assert changed is True
    assert event == "enter_yellow"
    assert controller.next_phase_idx == 2
    assert dec["action"] == 2
    assert dec["action_name"] == "SWITCH"
    assert dec["target_phase"] == 2

    # Chạy qua 3s vàng + 1s đỏ toàn phần (tổng 4s dọn nút)
    for _ in range(4):
        controller.step(1.0)

    # Đã vào Pha 2 thành công
    assert controller.current_phase_idx == 2
    assert controller.is_yellow is False
    assert controller.is_all_red is False


def test_negative_pressure_reward_hand_calculation():
    """Đối chiếu tính tay công thức áp suất âm PressLight / MPLight:
    m1 (u1 -> d1): u1 = 8 xe, d1 = 2 xe => w1 = 8 - 2 = 6
    m2 (u2 -> d2): u2 = 5 xe, d2 = 1 xe => w2 = 5 - 1 = 4
    => Total Pressure P = 6 + 4 = 10
    => Step Reward (decision_interval=5.0s) = - 10 / 5.0 = -2.0
    """
    green_phases = ["GGrr", "rrGG"]
    movements = [
        {"link_index": 0, "from_lane": "u1", "to_lane": "d1"},
        {"link_index": 2, "from_lane": "u2", "to_lane": "d2"},
    ]
    counts = {"u1": 8, "d1": 2, "u2": 5, "d2": 1}

    controller = QLearningController(
        green_phases=green_phases,
        movements=movements,
        decision_interval=5.0,
        vehicle_count_fn=lambda lane: counts.get(lane, 0),
    )

    pressure = controller.calculate_intersection_pressure()
    assert pressure == 10.0

    controller.observe_step()
    # Bước đầu tiên (accumulated_queue_steps=0, discount=1.0)
    assert round(controller.accumulated_reward, 2) == -2.0


def test_flexible_starvation_prevention_q_learning():
    """Kiểm tra cơ chế chống đói pha linh hoạt:
    - Pha 1 không có xe chờ: thời gian đỏ không tích lũy (luôn = 0.0), không bị coi là đói.
    - Pha 2 có xe chờ liên tục chạm ngưỡng 90s: kích hoạt cưỡng chế can thiệp chống đói (max_red_starvation).
    """
    green_phases = ["GGrrrr", "rrGGrr", "rrrrGG"]
    movements = [
        {"link_index": 0, "from_lane": "l0", "to_lane": "out0"},
        {"link_index": 2, "from_lane": "l1", "to_lane": "out1"},
        {"link_index": 4, "from_lane": "l2", "to_lane": "out2"},
    ]
    # l1 không có xe (0), l2 có 4 xe dừng chờ
    halting_counts = {"l0": 1, "l1": 0, "l2": 4}

    controller = QLearningController(
        green_phases=green_phases,
        movements=movements,
        min_green=20.0,
        max_green=120.0,
        max_red=90.0,
        learning_enabled=False,
        epsilon=0.0,
        halting_count_fn=lambda l: halting_counts.get(l, 0),
    )
    controller.current_phase_idx = 0

    decisions = []
    # Chạy 90 giây khi đang ở Pha 0
    for _ in range(90):
        _, _, _, dec = controller.step(1.0)
        if dec:
            decisions.append(dec)

    # Pha 1 rỗng xe -> timer phải bằng 0.0, không bị đói
    assert controller.phase_red_timers[1] == 0.0
    # Quyết định tại giây thứ 90 phát hiện Pha 2 bị đói -> bắt buộc chuyển sang Pha 2
    assert len(decisions) > 0
    last_dec = decisions[-1]
    assert last_dec["is_forced"] is True
    assert last_dec["forced_reason"] == "max_red_starvation"
    assert last_dec["valid_actions"] == [2]
    assert last_dec["action"] == 2


def test_presslight_absolute_pressure_reward_q_learning():
    """Kiểm tra công thức PressLight chuẩn:
    1. Hai movement trái dấu (+10 và -10) không được triệt tiêu thành 0 mà phải có áp suất = |10| + |-10| = 20.
    2. Chỉ làn ra có 10 xe (chênh lệch -10) phải chịu phạt áp suất = |-10| = 10, reward = -2 (không được dương +2).
    """
    green_phases = ["GGGrrr", "rrrGGG"]
    movements = [
        {"link_index": 0, "from_lane": "u1", "to_lane": "d1"},
        {"link_index": 3, "from_lane": "u2", "to_lane": "d2"},
    ]
    # Tình huống 1: u1 có 10 xe, d1 có 0 xe (+10); u2 có 0 xe, d2 có 10 xe (-10)
    counts1 = {"u1": 10, "d1": 0, "u2": 0, "d2": 10}
    controller1 = QLearningController(
        green_phases=green_phases,
        movements=movements,
        decision_interval=5.0,
        vehicle_count_fn=lambda lane: counts1.get(lane, 0),
    )
    p1 = controller1.calculate_intersection_pressure()
    assert p1 == 20.0, f"Kỳ vọng áp suất 20.0 nhưng nhận được {p1}"
    controller1.observe_step()
    assert round(controller1.accumulated_reward, 2) == -4.0

    # Tình huống 2: Chỉ làn ra d1 có 10 xe (u1=0, u2=0, d2=0) -> chênh lệch -10
    counts2 = {"u1": 0, "d1": 10, "u2": 0, "d2": 0}
    controller2 = QLearningController(
        green_phases=green_phases,
        movements=movements,
        decision_interval=5.0,
        vehicle_count_fn=lambda lane: counts2.get(lane, 0),
    )
    p2 = controller2.calculate_intersection_pressure()
    assert p2 == 10.0, f"Kỳ vọng áp suất 10.0 nhưng nhận được {p2}"
    controller2.observe_step()
    # Reward phải âm (-2.0), tuyệt đối không được nhận reward dương (+2.0)
    assert round(controller2.accumulated_reward, 2) == -2.0


def test_finalize_episode_keep_full_interval_q_learning():
    """Kiểm tra finalize_episode() khi kết thúc ngay sau khi hành động KEEP chạy đủ 5 giây:
    - Đã thu 5 mẫu reward (accumulated_queue_steps == 5).
    - time_since_decision phải nhất quán là 5.0 (không được là 4.0).
    - finalize_episode() phải ghi nhận chuyển tiếp, không được bỏ qua.
    """
    controller = QLearningController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[{"link_index": 0, "from_lane": "l1"}],
        decision_interval=5.0,
        min_green=20.0,
        learning_enabled=True,
        epsilon=0.0,
    )
    # Chạy 20s min_green ban đầu
    for _ in range(20):
        controller.step(1.0)
        controller.observe_step(step_queue=1)

    # Tại t=21: ra quyết định KEEP đầu tiên
    sig, ch, ev, dec = controller.step(1.0)
    controller.observe_step(step_queue=1)
    assert dec is not None
    assert dec["action_name"] == "KEEP"

    # Chạy tiếp 4 bước nữa để hoàn thành đủ 5s KEEP (t=22, 23, 24, 25)
    for _ in range(4):
        controller.step(1.0)
        controller.observe_step(step_queue=1)

    assert controller.accumulated_queue_steps == 5
    assert controller.time_since_decision == 5.0, f"Kỳ vọng time_since_decision=5.0 nhưng nhận được {controller.time_since_decision}"

    # Gọi finalize_episode() khi hết giờ mô phỏng
    assert controller.update_count == 0
    controller.finalize_episode()
    assert controller.update_count == 1, "finalize_episode() đã bỏ qua chuyển tiếp KEEP cuối dù đã thực thi đủ 5 giây"


def test_competing_phase_identification_distinguishes_situations():
    """Tái hiện tình huống từ phản hồi người dùng:
    Pha 0 đang xanh (0 xe).
    Tình huống A: Pha 1 có 10 xe, Pha 2 có 0 xe.
    Tình huống B: Pha 1 có 0 xe, Pha 2 có 10 xe.

    Kiểm tra:
      1. Với controller cũ (include_comp_phase_id=False): Cả 2 tình huống đều trả về (0, 0, 0, 1)
      2. Với controller mới (include_comp_phase_id=True):
         - Tình huống A trả về (0, 0, 0, 1, 1) -> nhận diện pha 1
         - Tình huống B trả về (0, 0, 0, 1, 2) -> nhận diện pha 2
         => Hai state khác biệt hoàn toàn, giúp model học chọn đúng pha cần phục vụ.
    """
    phases = ["Grr", "rGr", "rrG"]
    movements = [
        {"link_index": 0, "from_lane": "lane_0"},
        {"link_index": 1, "from_lane": "lane_1"},
        {"link_index": 2, "from_lane": "lane_2"},
    ]

    # 1. Controller mặc định (kế thừa)
    ctrl_legacy = QLearningController(
        green_phases=phases,
        movements=movements,
        include_comp_phase_id=False,
    )
    ctrl_legacy.current_phase_idx = 0
    ctrl_legacy.time_in_phase = 20.0

    # Tình huống A
    ctrl_legacy._halting_count_fn = lambda l: 0 if l == "lane_0" else (10 if l == "lane_1" else 0)
    state_legacy_A = ctrl_legacy.discretize_state()

    # Tình huống B
    ctrl_legacy._halting_count_fn = lambda l: 0 if l == "lane_0" else (0 if l == "lane_1" else 10)
    state_legacy_B = ctrl_legacy.discretize_state()

    # Xác nhận hạn chế thiết kế: Cả 2 tình huống bị trùng mã hóa
    assert state_legacy_A == (0, 0, 0, 1)
    assert state_legacy_B == (0, 0, 0, 1)
    assert state_legacy_A == state_legacy_B

    # 2. Controller nâng cấp với include_comp_phase_id=True
    ctrl_enhanced = QLearningController(
        green_phases=phases,
        movements=movements,
        include_comp_phase_id=True,
    )
    ctrl_enhanced.current_phase_idx = 0
    ctrl_enhanced.time_in_phase = 20.0

    # Tình huống A
    ctrl_enhanced._halting_count_fn = lambda l: 0 if l == "lane_0" else (10 if l == "lane_1" else 0)
    state_enhanced_A = ctrl_enhanced.discretize_state()

    # Tình huống B
    ctrl_enhanced._halting_count_fn = lambda l: 0 if l == "lane_0" else (0 if l == "lane_1" else 10)
    state_enhanced_B = ctrl_enhanced.discretize_state()

    # Phân biệt chính xác pha đối thủ có nhu cầu cao nhất
    assert state_enhanced_A == (0, 0, 0, 1, 1)
    assert state_enhanced_B == (0, 0, 0, 1, 2)
    assert state_enhanced_A != state_enhanced_B


def test_ql_dap_reward_and_anti_starvation_timer():
    """Kiểm tra hàm thưởng DAP (Delay-Aware Pressure) và bộ đếm red timer trong QLearningController."""
    phases = ["Grr", "rGr", "rrG"]
    movements = [
        {"link_index": 0, "from_lane": "lane_0", "to_lane": "out_0"},
        {"link_index": 1, "from_lane": "lane_1", "to_lane": "out_1"},
        {"link_index": 2, "from_lane": "lane_2", "to_lane": "out_2"},
    ]
    ctrl = QLearningController(
        green_phases=phases,
        movements=movements,
        decision_interval=5.0,
        beta_wait=0.05,
        switch_cost=1.0,
        halting_count_fn=lambda l: 0,
        vehicle_count_fn=lambda l: 1 if l == "lane_1" else 0,
    )
    ctrl.current_phase_idx = 0
    ctrl.time_in_phase = 20.0

    # 1. Kiểm tra step_reward tính đúng công thức DAP
    ctrl.accumulated_reward = 0.0
    ctrl.accumulated_queue_steps = 0
    ctrl.observe_step(step_pressure=10.0, step_waiting_time=80.0)
    expected_step_reward = -(10.0 / 5.0 + 0.05 * (80.0 / (5.0 * 3)))
    assert round(ctrl.accumulated_reward, 4) == round(expected_step_reward, 4)

    # 2. Kiểm tra red timer của pha 1 không bị reset về 0 khi có xe tồn tại (vehicle_count=1, halting=0)
    ctrl.phase_red_timers[1] = 10.0
    ctrl.step(step_duration=1.0)
    assert ctrl.phase_red_timers[1] == 11.0






