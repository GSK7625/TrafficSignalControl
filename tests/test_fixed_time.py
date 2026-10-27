"""Unit tests for FixedTimeController and yellow state generation."""

import pytest
from src.data_inspector import inspect_scenario


def test_build_yellow_state_standalone():
    """Test safe yellow state generation between opposing phases."""
    from src.controllers.base import build_yellow_state

    # When next state is opposing, green turning red becomes yellow, red stays red
    curr_state = "GGGrrr"
    next_state = "rrrGGG"
    assert build_yellow_state(curr_state, next_state) == "yyyrrr"

    # Opposite direction
    assert build_yellow_state("rrrGGG", "GGGrrr") == "rrryyy"

    # Single argument: all greens become yellow
    assert build_yellow_state("GGGGrrrr") == "yyyyrrrr"
    assert build_yellow_state("ggrr") == "yyrr"

    # Overlapping greens: signals that remain green do not turn yellow when only_changing=True
    assert build_yellow_state("GGrr", "GrrG", only_changing=True) == "Gyrr"

    # Real-world phases from Ingolstadt1
    # Phase 2 ('GGGrrrrr') -> Phase 4 ('rrrGGGrr')
    assert build_yellow_state("GGGrrrrr", "rrrGGGrr") == "yyyrrrrr"
    # Phase 4 ('rrrGGGrr') -> Phase 0 ('GGgGrGGG')
    assert build_yellow_state("rrrGGGrr", "GGgGrGGG") == "rrryyyrr"


def test_fixed_time_initialization():
    """Test FixedTimeController initialization and default parameters."""
    from src.controllers.fixed_time import FixedTimeController

    phases = ["GGGrrr", "rrrGGG"]
    controller = FixedTimeController(
        green_phases=phases,
        green_time=30.0,
        yellow_time=3.0,
        tls_id="tls_1",
    )

    assert controller.green_time == 30.0
    assert controller.yellow_time == 3.0
    assert controller.all_red_time == 1.0
    assert controller.max_red == 90.0
    assert controller.tls_id == "tls_1"
    assert controller.current_phase == 0
    assert controller.is_yellow is False
    assert controller.is_all_red is False
    assert controller.get_current_state() == "GGGrrr"

    # Validation errors
    with pytest.raises(ValueError):
        FixedTimeController(green_phases=[])

    with pytest.raises(ValueError):
        FixedTimeController(green_phases=phases, green_time=0.0)

    with pytest.raises(ValueError):
        FixedTimeController(green_phases=phases, yellow_time=-1.0)


def test_fixed_time_step_cycle_with_all_red_1s():
    """Test full cycle: 30s green -> 3s yellow -> 1s all-red -> 30s next green."""
    from src.controllers.fixed_time import FixedTimeController

    p0 = "GGGrrr"
    p1 = "rrrGGG"
    controller = FixedTimeController(
        green_phases=[p0, p1],
        green_time=30.0,
        yellow_time=3.0,
        all_red_time=1.0,
    )

    states = []
    yellow_flags = []
    all_red_flags = []

    # 30 green + 3 yellow + 1 all-red + 30 green = 64 steps
    for _ in range(64):
        state, _ = controller.step(step_duration=1.0)
        states.append(state)
        yellow_flags.append(controller.is_yellow)
        all_red_flags.append(controller.is_all_red)

    # 1. First 30 steps: Green phase 0
    assert all(s == p0 for s in states[:30])
    assert all(y is False for y in yellow_flags[:30])
    assert all(r is False for r in all_red_flags[:30])

    # 2. Next 3 steps (steps 30, 31, 32): Yellow phase
    assert all(s == "yyyrrr" for s in states[30:33])
    assert all(y is True for y in yellow_flags[30:33])
    assert all(r is False for r in all_red_flags[30:33])

    # 3. Next 1 step (step 33): All-Red clearance
    assert states[33] == "rrrrrr"
    assert yellow_flags[33] is False
    assert all_red_flags[33] is True

    # 4. Next 30 steps (steps 34 to 63): Green phase 1
    assert all(s == p1 for s in states[34:64])
    assert all(y is False for y in yellow_flags[34:64])
    assert all(r is False for r in all_red_flags[34:64])



def test_fixed_time_step_cycle_30_3_30():
    """Test full cycle step counting: 30s green -> 3s yellow -> 30s next green."""
    from src.controllers.fixed_time import FixedTimeController

    p0 = "GGGrrr"
    p1 = "rrrGGG"
    controller = FixedTimeController(
        green_phases=[p0, p1],
        green_time=30.0,
        yellow_time=3.0,
        all_red_time=0.0,
    )

    states = []
    yellow_flags = []
    phase_indices = []
    change_flags = []

    # Total 63 steps (30 green P0 + 3 yellow + 30 green P1)
    for _ in range(63):
        state, changed = controller.step(step_duration=1.0)
        states.append(state)
        change_flags.append(changed)
        yellow_flags.append(controller.is_yellow)
        phase_indices.append(controller.current_phase)

    # 1. First 30 steps: Green phase 0
    assert len(states[:30]) == 30
    assert all(s == p0 for s in states[:30])
    assert all(y is False for y in yellow_flags[:30])
    assert all(idx == 0 for idx in phase_indices[:30])
    assert change_flags[0] is True  # Initial activation
    assert all(c is False for c in change_flags[1:30])

    # 2. Next 3 steps (steps 31, 32, 33 / indices 30, 31, 32): Yellow phase
    assert len(states[30:33]) == 3
    assert all(s == "yyyrrr" for s in states[30:33])
    assert all(y is True for y in yellow_flags[30:33])
    assert change_flags[30] is True  # Transition to yellow
    assert change_flags[31] is False
    assert change_flags[32] is False

    # 3. Next 30 steps (steps 34 to 63 / indices 33 to 62): Green phase 1
    assert len(states[33:63]) == 30
    assert all(s == p1 for s in states[33:63])
    assert all(y is False for y in yellow_flags[33:63])
    assert all(idx == 1 for idx in phase_indices[33:63])
    assert change_flags[33] is True  # Transition to green phase 1
    assert all(c is False for c in change_flags[34:63])


def test_infinite_cycle_loop():
    """Test infinite periodic loop over multiple cycles."""
    from src.controllers.fixed_time import FixedTimeController

    p0 = "GGGrrr"
    p1 = "rrrGGG"
    controller = FixedTimeController(
        green_phases=[p0, p1],
        green_time=30.0,
        yellow_time=3.0,
        all_red_time=0.0,
    )

    num_cycles = 5

    for _ in range(num_cycles):
        # Phase 0 green (30 steps)
        for _ in range(30):
            state, _ = controller.step(1.0)
            assert state == p0
            assert controller.is_yellow is False
            assert controller.current_phase == 0

        # Phase 0 yellow (3 steps)
        for _ in range(3):
            state, _ = controller.step(1.0)
            assert state == "yyyrrr"
            assert controller.is_yellow is True

        # Phase 1 green (30 steps)
        for _ in range(30):
            state, _ = controller.step(1.0)
            assert state == p1
            assert controller.is_yellow is False
            assert controller.current_phase == 1

        # Phase 1 yellow (3 steps)
        for _ in range(3):
            state, _ = controller.step(1.0)
            assert state == "rrryyy"
            assert controller.is_yellow is True


def test_controller_reset():
    """Test reset restores initial state and timing."""
    from src.controllers.fixed_time import FixedTimeController

    p0 = "GGGrrr"
    p1 = "rrrGGG"
    controller = FixedTimeController(
        green_phases=[p0, p1],
        green_time=30.0,
        yellow_time=3.0,
        all_red_time=0.0,
    )

    # Advance 45 steps (into green phase 1)
    for _ in range(45):
        controller.step(1.0)

    assert controller.current_phase == 1
    assert controller.is_yellow is False

    # Reset
    controller.reset()
    assert controller.current_phase == 0
    assert controller.is_yellow is False
    assert controller.get_current_state() == p0

    # Step again from beginning
    state, changed = controller.step(1.0)
    assert state == p0
    assert changed is True


def test_custom_step_duration():
    """Test controller with sub-second step durations (dt = 0.5s)."""
    from src.controllers.fixed_time import FixedTimeController

    controller = FixedTimeController(
        green_phases=["GG", "rr"],
        green_time=10.0,
        yellow_time=2.0,
    )

    # 10s green with dt=0.5s -> 20 steps
    for _ in range(20):
        state, _ = controller.step(step_duration=0.5)
        assert state == "GG"
        assert controller.is_yellow is False

    # 2s yellow with dt=0.5s -> 4 steps
    for _ in range(4):
        state, _ = controller.step(step_duration=0.5)
        assert state == "yy"
        assert controller.is_yellow is True


def test_integration_with_scenario_info_cologne1():
    """Test FixedTimeController using phases loaded from cologne1 network."""
    from src.controllers.fixed_time import FixedTimeController

    info = inspect_scenario("cologne1", data_dir="data")
    green_phases = info.get_green_phases()
    assert len(green_phases) > 0

    controller = FixedTimeController(
        green_phases=green_phases,
        green_time=30.0,
        yellow_time=3.0,
        tls_id=info.tls_ids[0],
    )

    state, changed = controller.step(1.0)
    assert len(state) == 20
    assert changed is True
    assert controller.current_phase == 0
    assert controller.is_yellow is False
