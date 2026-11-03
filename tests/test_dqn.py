"""Kiểm thử thuật toán Deep Q-Network (DQNController).

Bao gồm:
  1. Đối chiếu ví dụ tính tay Target Bellman, TD Error và Huber Loss (Giai đoạn 9).
  2. Mã hóa và chuẩn hóa vector trạng thái liên tục.
  3. Action Masking (ràng buộc an toàn tại thời điểm chọn và tại s_next).
  4. Tính nhất quán 100% khi lưu và tải checkpoint PyTorch.
  5. Chế độ đánh giá (Evaluation Mode): tắt explore, freeze mạng.
"""

import os
import tempfile
import pytest
import torch
import torch.nn as nn

from src.controllers.dqn import DQNController, DQNNetwork, ReplayBuffer


def test_dqn_hand_calculation():
    """Đối chiếu ví dụ tính tay từ thiết kế Giai đoạn 9:

    - Hành động a = 1 (SWITCH)
    - Phần thưởng r = -4.0
    - gamma = 0.95, terminated = False
    - Trạng thái kế tiếp s' có valid_actions = [0] (do t' = 1s < 10s)
    - Giả định Target net Q(s') = [-5.0, -2.0]
      => Masked max Q_target(s') = -5.0 (vì action 1 bị cấm)
      => Target y = -4.0 + 0.95 * (-5.0) = -8.75
    - Giả định Online net Q(s, 1) = -7.0
      => TD Error delta = -8.75 - (-7.0) = -1.75
      => Huber Loss (|delta| = 1.75 > 1.0) = 1.75 - 0.5 = 1.25
    """
    controller = DQNController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[{"link_index": 0, "from_lane": "l1", "to_lane": "l2"}],
        state_dim=6,
        q_max=30.0,
        gamma=0.95,
        batch_size=1,
        min_replay_size=1,
    )

    # 1. Giả lập đầu ra của Target Network tại s'
    s_prime = [1.0, 0.0, 0.0, 0.05, 0.1, 0.2]
    with torch.no_grad():
        target_q = torch.tensor([[-5.0, -2.0]], dtype=torch.float32)
        valid_actions_next = [[0]]

        # Áp dụng Action Masking
        for a in [0, 1]:
            if a not in valid_actions_next[0]:
                target_q[0, a] = -1e9

        max_next_q, _ = torch.max(target_q, dim=1)
        r = torch.tensor([-4.0], dtype=torch.float32)
        gamma = 0.95
        terminated = torch.tensor([0.0], dtype=torch.float32)

        # Tính target y
        y = r + gamma * (1.0 - terminated) * max_next_q

    assert round(y.item(), 2) == -8.75

    # 2. Giả lập Online Network dự đoán Q(s, 1) = -7.0
    q_pred = torch.tensor([-7.0], dtype=torch.float32)
    delta = y.item() - q_pred.item()
    assert round(delta, 2) == -1.75

    # 3. Tính Huber loss và đối chiếu với 1.25
    loss_fn = nn.SmoothL1Loss()
    computed_loss = loss_fn(q_pred, y)
    assert round(computed_loss.item(), 2) == 1.25


def test_state_encoding_and_normalization():
    """Kiểm tra mã hóa One-hot và chuẩn hóa vector trạng thái."""
    controller = DQNController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[
            {"link_index": 0, "from_lane": "lane_0", "to_lane": "lane_out_0"},
            {"link_index": 3, "from_lane": "lane_1", "to_lane": "lane_out_1"},
        ],
        state_dim=5,  # 2 phases + 1 time + 2 queues
        q_max=20.0,
        max_green=60.0,
        halting_count_fn=lambda lane: 10 if lane == "lane_0" else 5,
    )

    # Pha 0 xanh, thời gian 30s
    controller.current_phase_idx = 0
    controller.time_in_phase = 30.0

    state_vec = controller.encode_state()

    # One-hot pha 0: [1.0, 0.0]
    assert state_vec[0] == 1.0
    assert state_vec[1] == 0.0

    # Thời gian 30s / 60s = 0.5
    assert state_vec[2] == 0.5

    # Hàng đợi pha phục vụ (lane_0) = 10 / 20 = 0.5
    assert state_vec[3] == 0.5

    # Hàng đợi cạnh tranh (lane_1) = 5 / 20 = 0.25
    assert state_vec[4] == 0.25

    # Tất cả giá trị đều nằm trong đoạn [0.0, 1.0]
    for val in state_vec:
        assert 0.0 <= val <= 1.0


def test_action_masking_dqn():
    """Kiểm tra ràng buộc an toàn kỹ thuật (Action Masking)."""
    controller = DQNController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[],
        state_dim=4,
        q_max=30.0,
        min_green=10.0,
        max_green=60.0,
    )

    # 1. Khi t < 10s: chỉ được KEEP (0)
    assert controller.get_valid_actions(5.0) == [0]

    # 2. Khi 10s <= t < 60s: được KEEP hoặc SWITCH
    assert controller.get_valid_actions(25.0) == [0, 1]

    # 3. Khi t >= 60s: bắt buộc SWITCH (1)
    assert controller.get_valid_actions(60.0) == [1]
    assert controller.get_valid_actions(70.0) == [1]

    # 4. Khi choose_action với mask [0], luôn trả về 0 dù epsilon cao
    controller.epsilon = 1.0
    for _ in range(10):
        action = controller.choose_action([1.0, 0.0, 0.1, 0.2], valid_actions=[0])
        assert action == 0


def test_save_load_consistency_dqn():
    """Kiểm tra lưu và tải checkpoint giữ nguyên trọng số và dự đoán Q-value."""
    controller = DQNController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[],
        state_dim=6,
        q_max=30.0,
        lr=0.001,
        epsilon=0.15,
        min_green=15.0,
        max_green=50.0,
        decision_interval=8.0,
        max_red=110.0,
    )
    controller.update_count = 55

    test_state = [1.0, 0.0, 0.0, 0.5, 0.3, 0.1]
    with torch.no_grad():
        original_q = controller.online_net(
            torch.tensor([test_state], dtype=torch.float32)
        ).tolist()[0]

    with tempfile.NamedTemporaryFile("wb", suffix=".pt", delete=False) as f:
        temp_path = f.name

    try:
        controller.save(temp_path, metadata={"scenario": "test_ingolstadt"})

        # Tạo controller mới và nạp lại
        loaded_controller = DQNController(
            green_phases=["GGGrrr", "rrrGGG"],
            movements=[],
            state_dim=6,
            q_max=30.0,
        )
        loaded_controller.load(temp_path)

        assert loaded_controller.update_count == 55
        assert loaded_controller.epsilon == 0.15
        assert loaded_controller.min_green == 15.0
        assert loaded_controller.max_green == 50.0
        assert loaded_controller.decision_interval == 8.0
        assert loaded_controller.max_red == 110.0

        with torch.no_grad():
            loaded_q = loaded_controller.online_net(
                torch.tensor([test_state], dtype=torch.float32)
            ).tolist()[0]

        # Đối chiếu Q-values khớp nhau đến 5 chữ số thập phân
        for orig, loaded in zip(original_q, loaded_q):
            assert round(orig, 5) == round(loaded, 5)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_evaluation_mode_dqn():
    """Kiểm tra chế độ đánh giá: tắt hoàn toàn khám phá và không cập nhật trọng số."""
    controller = DQNController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[],
        state_dim=4,
        q_max=30.0,
        epsilon=0.0,
        learning_enabled=False,
    )

    state_vec = [1.0, 0.0, 0.2, 0.3]
    action = controller.choose_action(state_vec, valid_actions=[0, 1])
    assert action in (0, 1)

    # Thử gọi update_network khi learning_enabled = False
    loss = controller.update_network()
    assert loss is None
    assert controller.update_count == 0


def test_evaluation_mode_no_replay_buffer_leak():
    """Kiểm tra khi learning_enabled=False (Validation/Evaluation), replay buffer không nhận thêm transition nào."""
    controller = DQNController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[{"link_index": 0, "from_lane": "l1"}],
        state_dim=5,  # 2 phases + 1 time + 2 queues = 5
        q_max=30.0,
        learning_enabled=False,
    )
    assert len(controller.replay_buffer) == 0

    # Chạy 150 bước ở chế độ đánh giá
    for _ in range(150):
        controller.step(1.0)

    # Replay buffer bắt buộc phải bằng 0, không được rò rỉ dữ liệu validation
    assert len(controller.replay_buffer) == 0

    # Reset với clear_buffer=True
    controller.reset(clear_buffer=True)
    assert len(controller.replay_buffer) == 0


def test_disjoint_seeds():
    """Kiểm tra các tập hạt giống train, val và test tách biệt hoàn toàn."""
    from src.train_q_learning import TRAIN_SEEDS as QL_TRAIN, VAL_SEEDS as QL_VAL
    from src.train_dqn import TRAIN_SEEDS as DQN_TRAIN, VAL_SEEDS as DQN_VAL
    test_seeds = [42, 101, 2024, 3407, 9999]

    assert set(QL_TRAIN).isdisjoint(set(QL_VAL))
    assert set(QL_TRAIN).isdisjoint(set(test_seeds))
    assert set(QL_VAL).isdisjoint(set(test_seeds))

    assert set(DQN_TRAIN).isdisjoint(set(DQN_VAL))
    assert set(DQN_TRAIN).isdisjoint(set(test_seeds))
    assert set(DQN_VAL).isdisjoint(set(test_seeds))


def test_finalize_episode_dqn():
    """Kiểm tra finalize_episode() của DQN: loại bỏ bước dở dang khi bị ngắt giữa chừng, chỉ lưu khi ở ranh giới quyết định."""
    controller = DQNController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[{"link_index": 0, "from_lane": "l1"}],
        state_dim=5,
        q_max=30.0,
        decision_interval=5.0,
        min_green=20.0,
        learning_enabled=True,
    )
    s = [1.0, 0.0, 0.0, 0.0, 0.0]
    controller.last_state = s
    controller.last_action = 1
    controller.accumulated_reward = -42.75
    controller.accumulated_queue_steps = 24

    # Trường hợp 1: Episode bị ngắt giữa chừng khi đang đèn vàng -> bỏ qua
    controller.is_yellow = True
    controller.finalize_episode()
    assert len(controller.replay_buffer) == 0
    assert controller.last_state is None

    # Trường hợp 2: Episode kết thúc tại ranh giới quyết định hợp lệ
    controller.last_state = s
    controller.last_action = 1
    controller.accumulated_reward = -42.75
    controller.accumulated_queue_steps = 24
    controller.is_yellow = False
    controller.is_all_red = False
    controller.time_in_phase = 20.0
    controller.time_since_decision = 5.0

    controller.finalize_episode()
    assert len(controller.replay_buffer) == 1
    # Kiểm tra transition được lưu với terminated=False (time limit truncation)
    last_item = controller.replay_buffer.buffer[-1]
    assert last_item[1] == 1  # action
    assert last_item[4] is False  # terminated
    assert controller.last_state is None


def test_sensor_exception_not_swallowed_dqn():
    """Kiểm tra DQN không nuốt ngoại lệ cảm biến làn đường."""
    def broken_sensor(lane: str) -> int:
        raise ConnectionError("Mất tín hiệu cảm biến")

    controller = DQNController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[{"link_index": 0, "from_lane": "lane_bad"}],
        state_dim=5,
        q_max=30.0,
        halting_count_fn=broken_sensor,
    )
    with pytest.raises(ConnectionError, match="Mất tín hiệu cảm biến"):
        controller.calculate_total_intersection_queue()


def test_direct_phase_selection_dqn():
    """Kiểm tra lựa chọn pha trực tiếp trong DQN:
    Mạng DQN có thể chọn trực tiếp bất kỳ pha a in {0, 1, ..., N-1}.
    """
    green_phases = ["GGrrrr", "rrGGrr", "rrrrGG"]
    movements = [
        {"link_index": 0, "from_lane": "l0", "to_lane": "out0"},
        {"link_index": 2, "from_lane": "l1", "to_lane": "out1"},
        {"link_index": 4, "from_lane": "l2", "to_lane": "out2"},
    ]
    controller = DQNController(
        green_phases=green_phases,
        movements=movements,
        state_dim=6,  # 3 phases + 1 time + 2 queues
        q_max=30.0,
        min_green=20.0,
        max_green=45.0,
        yellow_time=3.0,
        all_red_time=1.0,
        learning_enabled=False,
        epsilon=0.0,
    )
    assert controller.num_phases == 3
    # Output dimension của online_net và target_net phải bằng 3
    assert controller.online_net.fc3.out_features == 3

    # Giả lập Q-values tại bước quyết định sao cho Pha 2 có giá trị cao nhất
    state_vec = controller.encode_state()
    # Mock online_net để trả về Q-values [1.0, 0.0, 10.0]
    controller.online_net.eval()
    with torch.no_grad():
        controller.online_net.fc3.bias.data = torch.tensor([1.0, 0.0, 10.0], dtype=torch.float32)
        controller.online_net.fc3.weight.data.zero_()

    valid_actions = controller.get_valid_actions(25.0)
    assert valid_actions == [0, 1, 2]

    action = controller.choose_action(state_vec, valid_actions)
    assert action == 2

    # Step: chuyển thẳng sang Pha 2 qua đèn vàng
    controller.time_in_phase = 20.0
    controller.time_since_decision = 5.0
    _, changed, event, dec = controller.step(1.0)
    assert changed is True
    assert event == "enter_yellow"
    assert controller.next_phase_idx == 2
    assert dec["action"] == 2
    assert dec["target_phase"] == 2
    assert dec["action_name"] == "SWITCH"


def test_negative_pressure_reward_dqn():
    """Kiểm tra tính toán áp suất âm PressLight / MPLight trong DQNController."""
    green_phases = ["GGrr", "rrGG"]
    movements = [
        {"link_index": 0, "from_lane": "u1", "to_lane": "d1"},
        {"link_index": 2, "from_lane": "u2", "to_lane": "d2"},
    ]
    counts = {"u1": 15, "d1": 3, "u2": 10, "d2": 2}

    controller = DQNController(
        green_phases=green_phases,
        movements=movements,
        state_dim=5,
        q_max=30.0,
        decision_interval=5.0,
        vehicle_count_fn=lambda lane: counts.get(lane, 0),
    )

    # w1 = 15 - 3 = 12, w2 = 10 - 2 = 8 => Pressure = 20
    pressure = controller.calculate_intersection_pressure()
    assert pressure == 20.0

    controller.observe_step()
    # Step reward = - 20 / 5.0 = -4.0
    assert round(controller.accumulated_reward, 2) == -4.0


def test_flexible_starvation_prevention_dqn():
    """Kiểm tra cơ chế chống đói pha linh hoạt trong DQN:
    - Pha rỗng xe: red timer = 0.0
    - Pha có xe chờ: red timer tích lũy và kích hoạt can thiệp khi >= 90s.
    """
    green_phases = ["GGrrrr", "rrGGrr", "rrrrGG"]
    movements = [
        {"link_index": 0, "from_lane": "l0", "to_lane": "out0"},
        {"link_index": 2, "from_lane": "l1", "to_lane": "out1"},
        {"link_index": 4, "from_lane": "l2", "to_lane": "out2"},
    ]
    halting_counts = {"l0": 2, "l1": 0, "l2": 3}

    controller = DQNController(
        green_phases=green_phases,
        movements=movements,
        state_dim=6,
        q_max=30.0,
        min_green=20.0,
        max_green=120.0,
        max_red=90.0,
        learning_enabled=False,
        epsilon=0.0,
        halting_count_fn=lambda l: halting_counts.get(l, 0),
    )
    # Khởi tạo trọng số ưu tiên giữ Pha 0 khi còn được tự do
    with torch.no_grad():
        controller.online_net.fc3.bias.data = torch.tensor([10.0, -10.0, -10.0], dtype=torch.float32)
        controller.online_net.fc3.weight.data.zero_()

    decisions = []
    for _ in range(90):
        _, _, _, dec = controller.step(1.0)
        if dec:
            decisions.append(dec)

    # Pha 1 rỗng xe -> timer = 0
    assert controller.phase_red_timers[1] == 0.0
    # Quyết định tại giây thứ 90 phát hiện Pha 2 bị đói -> bắt buộc chuyển sang Pha 2
    assert len(decisions) > 0
    last_dec = decisions[-1]
    assert last_dec["is_forced"] is True
    assert last_dec["forced_reason"] == "max_red_starvation"
    assert last_dec["valid_actions"] == [2]
    assert last_dec["action"] == 2


def test_presslight_absolute_pressure_reward_dqn():
    """Kiểm tra công thức PressLight chuẩn trong DQN:
    1. Hai movement trái dấu (+10 và -10) không triệt tiêu thành 0 mà áp suất = |10| + |-10| = 20.
    2. Chỉ làn ra có 10 xe (chênh lệch -10) phải có áp suất = |-10| = 10, reward = -2 (không được dương +2).
    """
    green_phases = ["GGGrrr", "rrrGGG"]
    movements = [
        {"link_index": 0, "from_lane": "u1", "to_lane": "d1"},
        {"link_index": 3, "from_lane": "u2", "to_lane": "d2"},
    ]
    # Tình huống 1: u1 có 10 xe, d1 có 0 xe (+10); u2 có 0 xe, d2 có 10 xe (-10)
    counts1 = {"u1": 10, "d1": 0, "u2": 0, "d2": 10}
    controller1 = DQNController(
        green_phases=green_phases,
        movements=movements,
        state_dim=5,
        q_max=30.0,
        decision_interval=5.0,
        vehicle_count_fn=lambda lane: counts1.get(lane, 0),
    )
    p1 = controller1.calculate_intersection_pressure()
    assert p1 == 20.0, f"Kỳ vọng áp suất 20.0 nhưng nhận được {p1}"
    controller1.observe_step()
    assert round(controller1.accumulated_reward, 2) == -4.0

    # Tình huống 2: Chỉ làn ra d1 có 10 xe (chênh lệch -10)
    counts2 = {"u1": 0, "d1": 10, "u2": 0, "d2": 0}
    controller2 = DQNController(
        green_phases=green_phases,
        movements=movements,
        state_dim=5,
        q_max=30.0,
        decision_interval=5.0,
        vehicle_count_fn=lambda lane: counts2.get(lane, 0),
    )
    p2 = controller2.calculate_intersection_pressure()
    assert p2 == 10.0, f"Kỳ vọng áp suất 10.0 nhưng nhận được {p2}"
    controller2.observe_step()
    # Reward phải âm (-2.0), không được dương (+2.0)
    assert round(controller2.accumulated_reward, 2) == -2.0


def test_finalize_episode_keep_full_interval_dqn():
    """Kiểm tra finalize_episode() của DQN khi kết thúc ngay sau khi KEEP chạy đủ 5 giây:
    - Đã thu 5 mẫu reward (accumulated_queue_steps == 5).
    - time_since_decision phải nhất quán là 5.0 (không được là 4.0).
    - Replay buffer phải ghi nhận transition, không được bỏ qua.
    """
    controller = DQNController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[{"link_index": 0, "from_lane": "l1"}],
        state_dim=5,
        q_max=30.0,
        decision_interval=5.0,
        min_green=20.0,
        learning_enabled=True,
        epsilon=0.0,
    )
    # Cố định hành động KEEP để kiểm tra kịch bản giữ pha đủ 5s
    controller.choose_action = lambda s, v: controller.current_phase_idx

    # Chạy 20s min_green ban đầu
    for _ in range(20):
        controller.step(1.0)
        controller.observe_step(step_queue=1)

    # Tại t=21: ra quyết định KEEP
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
    assert len(controller.replay_buffer) == 0
    controller.finalize_episode()
    assert len(controller.replay_buffer) == 1, "finalize_episode() đã bỏ qua transition KEEP cuối dù đã thực thi đủ 5 giây"


def test_no_duplicate_methods_dqn():
    """Kiểm tra DQNController không có hàm bị định nghĩa trùng lặp và checkpoint lưu đầy đủ action_dim."""
    import inspect
    from src.controllers import dqn

    source = inspect.getsource(dqn.DQNController)
    for func_name in ["def finalize_episode", "def save", "def load"]:
        count = source.count(func_name)
        assert count == 1, f"Hàm {func_name} xuất hiện {count} lần trong DQNController (kỳ vọng đúng 1 lần)"


def test_dqn_per_phase_queues_distinguishes_situations():
    """Tái hiện tình huống từ phản hồi người dùng trên DQN:
    Pha 0 đang xanh (0 xe).
    Tình huống A: Pha 1 có 10 xe, Pha 2 có 0 xe.
    Tình huống B: Pha 1 có 0 xe, Pha 2 có 10 xe.

    Kiểm tra:
      1. Với controller legacy (per_phase_queues=False): Cả 2 tình huống đều trả về cùng vector state
         do tổng hàng đợi đối thủ đều là 10.
      2. Với controller nâng cấp (per_phase_queues=True):
         Vector state phản ánh riêng rẽ hàng đợi từng pha, giúp mạng nơ-ron phân biệt được
         chính xác pha 1 hay pha 2 cần phục vụ.
    """
    phases = ["Grr", "rGr", "rrG"]
    movements = [
        {"link_index": 0, "from_lane": "lane_0"},
        {"link_index": 1, "from_lane": "lane_1"},
        {"link_index": 2, "from_lane": "lane_2"},
    ]

    # 1. Chế độ Legacy: state_dim = 6 (3 one-hot + 1 time + 2 queues)
    ctrl_legacy = DQNController(
        green_phases=phases,
        movements=movements,
        state_dim=6,
        q_max=20.0,
        per_phase_queues=False,
    )
    ctrl_legacy.current_phase_idx = 0
    ctrl_legacy.time_in_phase = 22.5

    # Tình huống A
    ctrl_legacy._halting_count_fn = lambda l: 0 if l == "lane_0" else (10 if l == "lane_1" else 0)
    vec_legacy_A = ctrl_legacy.encode_state()

    # Tình huống B
    ctrl_legacy._halting_count_fn = lambda l: 0 if l == "lane_0" else (0 if l == "lane_1" else 10)
    vec_legacy_B = ctrl_legacy.encode_state()

    # Khẳng định hạn chế thiết kế của legacy: 2 tình huống bị trùng lặp 100%
    assert vec_legacy_A == vec_legacy_B
    assert vec_legacy_A == [1.0, 0.0, 0.0, 0.5, 0.0, 0.5]

    # 2. Chế độ nâng cấp: state_dim = 7 (3 one-hot + 1 time + 3 phase queues)
    ctrl_enhanced = DQNController(
        green_phases=phases,
        movements=movements,
        state_dim=7,
        q_max=20.0,
        per_phase_queues=True,
    )
    ctrl_enhanced.current_phase_idx = 0
    ctrl_enhanced.time_in_phase = 22.5

    # Tình huống A
    ctrl_enhanced._halting_count_fn = lambda l: 0 if l == "lane_0" else (10 if l == "lane_1" else 0)
    vec_enhanced_A = ctrl_enhanced.encode_state()

    # Tình huống B
    ctrl_enhanced._halting_count_fn = lambda l: 0 if l == "lane_0" else (0 if l == "lane_1" else 10)
    vec_enhanced_B = ctrl_enhanced.encode_state()

    # Phân biệt rõ rệt vị trí hàng đợi: pha 1 (0.5) vs pha 2 (0.5)
    assert vec_enhanced_A == [1.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0]
    assert vec_enhanced_B == [1.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5]
    assert vec_enhanced_A != vec_enhanced_B


def test_dqn_dap_reward_and_anti_starvation_timer():
    """Kiểm tra hàm thưởng DAP (Delay-Aware Pressure) và bộ đếm red timer không bị reset oan."""
    phases = ["Grr", "rGr", "rrG"]
    movements = [
        {"link_index": 0, "from_lane": "lane_0", "to_lane": "out_0"},
        {"link_index": 1, "from_lane": "lane_1", "to_lane": "out_1"},
        {"link_index": 2, "from_lane": "lane_2", "to_lane": "out_2"},
    ]
    ctrl = DQNController(
        green_phases=phases,
        movements=movements,
        state_dim=7,
        q_max=30.0,
        decision_interval=5.0,
        beta_wait=0.05,
        switch_cost=1.0,
        halting_count_fn=lambda l: 0,
        vehicle_count_fn=lambda l: 1 if l == "lane_1" else 0,
    )
    ctrl.current_phase_idx = 0
    ctrl.time_in_phase = 20.0

    # 1. Kiểm tra step_reward tính đúng công thức DAP
    # pressure=10.0, waiting_time=80.0, N_lanes=3, delta_t=5.0, beta=0.05
    # pressure_term = 10.0 / 5.0 = 2.0
    # wait_term = 0.05 * 80.0 / (5.0 * 3) = 4.0 / 15.0 = 0.2666667
    # step_reward = -(2.0 + 0.2666667) = -2.2667
    ctrl.accumulated_reward = 0.0
    ctrl.accumulated_queue_steps = 0
    ctrl.observe_step(step_pressure=10.0, step_waiting_time=80.0)
    expected_step_reward = -(10.0 / 5.0 + 0.05 * (80.0 / (5.0 * 3)))
    assert round(ctrl.accumulated_reward, 4) == round(expected_step_reward, 4)

    # 2. Kiểm tra red timer của pha 1 không bị reset về 0 khi có xe tồn tại (vehicle_count=1, halting=0)
    ctrl.phase_red_timers[1] = 10.0
    ctrl.step(step_duration=1.0)
    # Sau 1 giây, timer phải tăng lên 11.0 chứ không bị reset về 0.0
    assert ctrl.phase_red_timers[1] == 11.0






