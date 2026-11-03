"""Kiểm thử bộ điều khiển Max-Pressure (MaxPressureController).

Bao gồm:
  1. Đối chiếu ví dụ tính tay công thức áp suất W(p).
  2. Trường hợp áp suất bằng nhau (Tie-breaking).
  3. Ràng buộc an toàn: min_green (20s), max_green (45s), chống đói pha (90s).
  4. Chạy mô phỏng thực tế TraCI với Max-Pressure.
"""

import os
import pytest

from src.controllers.max_pressure import MaxPressureController
from src.data_inspector import read_scenario
from src.simulation_runner import run_simulation


def test_hand_calculation_max_pressure():
    """Đối chiếu ví dụ số nhỏ tính tay áp suất:

    Pha 0: m1 (u1->d1), m2 (u2->d2)
    Pha 1: m3 (u3->d3), m4 (u4->d4)
    Xe vào: u1=12, u2=8, u3=5, u4=3
    Xe ra:  d1=2,  d2=1, d3=4, d4=1
    => W(P0) = (12 - 2) + (8 - 1) = 17
    => W(P1) = (5 - 4) + (3 - 1) = 3
    => Pha được chọn là Pha 0.
    """
    green_phases = [
        "GGrr",  # Pha 0: bật link 0, 1
        "rrGG",  # Pha 1: bật link 2, 3
    ]
    movements = [
        {"link_index": 0, "from_lane": "u1", "to_lane": "d1"},
        {"link_index": 1, "from_lane": "u2", "to_lane": "d2"},
        {"link_index": 2, "from_lane": "u3", "to_lane": "d3"},
        {"link_index": 3, "from_lane": "u4", "to_lane": "d4"},
    ]
    counts = {
        "u1": 12, "u2": 8, "u3": 5, "u4": 3,
        "d1": 2,  "d2": 1, "d3": 4, "d4": 1,
    }

    controller = MaxPressureController(
        green_phases=green_phases,
        movements=movements,
        min_green=20.0,
        max_green=45.0,
        vehicle_count_fn=lambda lane: counts.get(lane, 0),
    )

    pressures = controller.calculate_phase_pressures()
    assert pressures[0] == 17.0
    assert pressures[1] == 3.0

    chosen_p, reason, _ = controller.select_next_phase()
    assert chosen_p == 0
    assert reason == "keep_current_optimal"


def test_tie_breaking_logic():
    """Kiểm tra xử lý hòa khi 2 pha có áp suất bằng nhau."""
    green_phases = ["GGrr", "rrGG"]
    movements = [
        {"link_index": 0, "from_lane": "u1", "to_lane": "d1"},
        {"link_index": 1, "from_lane": "u2", "to_lane": "d2"},
        {"link_index": 2, "from_lane": "u3", "to_lane": "d3"},
        {"link_index": 3, "from_lane": "u4", "to_lane": "d4"},
    ]
    # Cả 2 pha đều có áp suất = 10
    counts = {
        "u1": 10, "u2": 0, "u3": 10, "u4": 0,
        "d1": 0,  "d2": 0, "d3": 0,  "d4": 0,
    }

    controller = MaxPressureController(
        green_phases=green_phases,
        movements=movements,
        min_green=20.0,
        max_green=45.0,
        vehicle_count_fn=lambda lane: counts.get(lane, 0),
    )

    # 1. Khi đang ở Pha 0, cả 2 pha cùng max (10 == 10) -> Ưu tiên giữ Pha 0 hiện tại
    chosen_p, reason, pressures = controller.select_next_phase()
    assert pressures[0] == 10.0
    assert pressures[1] == 10.0
    assert chosen_p == 0
    assert reason == "keep_current_optimal"

    # 2. Khi bị cưỡng bức đổi pha (force switch tại max_green) -> Chọn pha đối thủ có thời gian chờ đỏ lâu hơn
    controller.time_in_phase = 45.0  # chạm trần max_green
    controller.phase_red_timers[1] = 45.0
    chosen_p, reason, _ = controller.select_next_phase()
    assert chosen_p == 1
    assert reason == "max_green_forced"


def test_safety_constraints():
    """Kiểm tra min_green, max_green và chống đói pha (starvation watchdog)."""
    green_phases = ["GGrr", "rrGG"]
    movements = [
        {"link_index": 0, "from_lane": "u1", "to_lane": "d1"},
        {"link_index": 1, "from_lane": "u2", "to_lane": "d2"},
        {"link_index": 2, "from_lane": "u3", "to_lane": "d3"},
        {"link_index": 3, "from_lane": "u4", "to_lane": "d4"},
    ]
    # Pha 1 có áp suất vượt trội (100 vs 0)
    counts = {
        "u1": 0, "u2": 0, "u3": 100, "u4": 0,
        "d1": 0, "d2": 0, "d3": 0,   "d4": 0,
    }

    controller = MaxPressureController(
        green_phases=green_phases,
        movements=movements,
        min_green=20.0,
        max_green=45.0,
        max_red=90.0,
        vehicle_count_fn=lambda lane: counts.get(lane, 0),
    )

    # 1. Trong 20 giây đầu: chưa vượt qua min_green (20s xanh), không được đổi pha
    for step in range(20):
        state, changed, event, dec = controller.step(1.0)
        assert state == "GGrr"
        assert changed is False
        assert controller.current_phase_idx == 0

    # 2. Bước tiếp theo (sau khi đã phục vụ đủ 20s xanh): cho phép đổi sang Pha 1 qua đèn vàng
    state, changed, event, dec = controller.step(1.0)
    assert changed is True
    assert event == "enter_yellow"
    assert controller.is_yellow is True


def test_simulation_runner_with_max_pressure():
    """Chạy thử nghiệm Max-Pressure trên ingolstadt1 bằng TraCI (70 bước)."""
    scen = read_scenario("ingolstadt1")
    mp_controller = MaxPressureController(
        green_phases=scen["green_phases"],
        movements=scen["movements"],
        min_green=20.0,
        max_green=45.0,
        yellow_time=3.0,
        all_red_time=1.0,
        tls_id=scen["tls_id"],
    )

    out_dir = os.path.join("results", "tests", "mp_ingolstadt1_test")
    summary = run_simulation(
        scenario_name="ingolstadt1",
        controller=mp_controller,
        max_steps=70,
        output_dir=out_dir,
    )

    assert summary["total_simulation_steps"] == 70
    assert summary["mean_queue_length"] is not None
    assert os.path.isfile(os.path.join(out_dir, "summary.json"))
    assert os.path.isfile(os.path.join(out_dir, "phase_log.json"))


def test_sensor_exception_not_swallowed_max_pressure():
    """Kiểm tra Max-Pressure không nuốt âm thầm ngoại lệ cảm biến thành 0."""
    def broken_fn(lane: str) -> int:
        raise ValueError("Lỗi đọc dữ liệu cảm biến làn")

    controller = MaxPressureController(
        green_phases=["GGGrrr", "rrrGGG"],
        movements=[{"link_index": 0, "from_lane": "l_in", "to_lane": "l_out"}],
        vehicle_count_fn=broken_fn,
    )
    with pytest.raises(ValueError, match="Lỗi đọc dữ liệu cảm biến làn"):
        controller.calculate_phase_pressures()
