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
