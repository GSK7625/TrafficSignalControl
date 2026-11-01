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

        # Khởi tạo một controller mới và nạp lại
        loaded_controller = QLearningController(
            green_phases=["GGGrrr", "rrrGGG"],
            movements=[],
        )
        loaded_controller.load(temp_path)

        assert loaded_controller.alpha == 0.15
        assert loaded_controller.gamma == 0.90
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
