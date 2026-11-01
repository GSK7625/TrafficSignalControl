"""Kiểm thử chu kỳ và thứ tự chuyển pha của FixedTimeController."""

from src.controllers.fixed_time import FixedTimeController, build_all_red_state, build_yellow_state
from src.simulation_runner import run_simulation


def test_fixed_time_controller_step_logic():
    """Kiểm tra logic đếm bước từng giây của FixedTimeController trên kịch bản 2 pha mẫu."""
    p0 = "GGGrrr"
    p1 = "rrrGGG"
    controller = FixedTimeController(
        green_phases=[p0, p1],
        green_time=30.0,
        yellow_time=3.0,
        all_red_time=1.0,
    )

    assert controller.cycle_time == (30.0 + 3.0 + 1.0) * 2  # 68.0s

    states = []
    events = []

    # Chạy đúng 1 chu kỳ hoàn chỉnh: 34s pha 0 + 34s pha 1 = 68 bước
    for _ in range(68):
        state, changed, event = controller.step(1.0)
        states.append(state)
        if changed and event:
            events.append(event)

    # 1. 30 giây đầu: Xanh P0
    assert all(s == p0 for s in states[:30])

    # 2. 3 giây tiếp theo (giây 31, 32, 33): Vàng
    assert all(s == "yyyrrr" for s in states[30:33])

    # 3. 1 giây tiếp theo (giây 34): Đỏ toàn phần
    assert states[33] == "rrrrrr"

    # 4. 30 giây tiếp theo (giây 35 đến 64): Xanh P1
    assert all(s == p1 for s in states[34:64])

    # 5. 3 giây vàng P1 (giây 65, 66, 67)
    assert all(s == "rrryyy" for s in states[64:67])

    # 6. 1 giây đỏ toàn phần (giây 68)
    assert states[67] == "rrrrrr"

    # Bước thứ 69 phải quay trở lại Pha xanh 0
    next_state, changed, event = controller.step(1.0)
    assert next_state == p0
    assert controller.current_phase_idx == 0


def test_fixed_time_one_full_cycle_ingolstadt1():
    """Kiểm tra một chu kỳ đầy đủ trên ingolstadt1 (3 pha x 34s = 102s)."""
    # 3 pha xanh x 34s = 102s
    summary = run_simulation(
        scenario_name="ingolstadt1",
        green_time=30.0,
        yellow_time=3.0,
        all_red_time=1.0,
        seed=42,
        max_steps=102,
        output_dir="results/tests/ft_cycle_ingolstadt1",
    )
    assert summary["total_simulation_steps"] == 102
    assert summary["config"]["cycle_time"] == 102.0


def test_fixed_time_one_full_cycle_cologne1():
    """Kiểm tra một chu kỳ đầy đủ trên cologne1 (8 pha x 34s = 272s)."""
    # 8 pha xanh x 34s = 272s
    summary = run_simulation(
        scenario_name="cologne1",
        green_time=30.0,
        yellow_time=3.0,
        all_red_time=1.0,
        seed=42,
        max_steps=272,
        output_dir="results/tests/ft_cycle_cologne1",
    )
    assert summary["total_simulation_steps"] == 272
    assert summary["config"]["cycle_time"] == 272.0
