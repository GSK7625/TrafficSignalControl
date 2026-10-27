"""Unit tests for MaxPressureController."""

from typing import Dict, List
import pytest
from src.data_inspector import MovementInfo, PhaseInfo


def _make_dummy_movements() -> List[MovementInfo]:
    """Helper to create dummy movements for a 2-phase intersection.

    Phase 0: Links 0, 1, 2 ('GGGrrr')
    Phase 1: Links 3, 4, 5 ('rrrGGG')
    """
    return [
        MovementInfo(link_index=0, from_lane="N_in_0", to_lane="S_out_0", direction="s"),
        MovementInfo(link_index=1, from_lane="N_in_1", to_lane="S_out_1", direction="s"),
        MovementInfo(link_index=2, from_lane="S_in_0", to_lane="N_out_0", direction="s"),
        MovementInfo(link_index=3, from_lane="E_in_0", to_lane="W_out_0", direction="s"),
        MovementInfo(link_index=4, from_lane="E_in_1", to_lane="W_out_1", direction="s"),
        MovementInfo(link_index=5, from_lane="W_in_0", to_lane="E_out_0", direction="s"),
    ]


def test_max_pressure_initialization():
    """Test initialization, defaults, properties, and input validations."""
    from src.controllers.max_pressure import MaxPressureController

    phases = ["GGGrrr", "rrrGGG"]
    movements = _make_dummy_movements()

    controller = MaxPressureController(
        green_phases=phases,
        movements=movements,
        min_green=10.0,
        max_green=60.0,
        yellow_time=3.0,
        tls_id="tls_test",
    )

    assert controller.tls_id == "tls_test"
    assert controller.min_green == 10.0
    assert controller.max_green == 60.0
    assert controller.yellow_time == 3.0
    assert controller.num_phases == 2
    assert controller.current_phase == 0
    assert controller.get_current_phase() == 0
    assert controller.is_yellow is False
    assert controller.is_in_yellow() is False
    assert controller.time_in_phase == 0.0
    assert controller.total_time == 0.0
    assert controller.step_count == 0
    assert controller.get_current_state() == "GGGrrr"

    # Support for PhaseInfo objects
    phase_objs = [
        PhaseInfo(index=0, duration=30.0, state="GGGrrr"),
        PhaseInfo(index=1, duration=30.0, state="rrrGGG"),
    ]
    controller_objs = MaxPressureController(green_phases=phase_objs, movements=movements)
    assert controller_objs.green_phases == ["GGGrrr", "rrrGGG"]

    # Input validations
    with pytest.raises(ValueError, match="at least one phase"):
        MaxPressureController(green_phases=[], movements=movements)

    with pytest.raises(ValueError, match="min_green must be positive"):
        MaxPressureController(green_phases=phases, movements=movements, min_green=0.0)

    with pytest.raises(ValueError, match="max_green .* must be greater"):
        MaxPressureController(green_phases=phases, movements=movements, min_green=10.0, max_green=5.0)

    with pytest.raises(ValueError, match="yellow_time must be non-negative"):
        MaxPressureController(green_phases=phases, movements=movements, yellow_time=-1.0)


def test_calculate_phase_pressures():
    """Test calculation of phase pressures using mock vehicle counts."""
    from src.controllers.max_pressure import MaxPressureController

    phases = ["GGGrrr", "rrrGGG"]
    movements = _make_dummy_movements()

    # Movement 0: N_0 -> S_0 (10 - 2 = 8)
    # Movement 1: N_1 -> S_1 (5 - 1 = 4)
    # Movement 2: S_0 -> N_0 (6 - 3 = 3)
    # Phase 0 pressure = 8 + 4 + 3 = 15.0
    #
    # Movement 3: E_0 -> W_0 (12 - 1 = 11)
    # Movement 4: E_1 -> W_1 (8 - 2 = 6)
    # Movement 5: W_0 -> E_0 (4 - 4 = 0)
    # Phase 1 pressure = 11 + 6 + 0 = 17.0
    counts = {
        "N_0": 10, "S_0": 2,
        "N_1": 5, "S_1": 1,
        # note S_0 is also used as from_lane for movement 2, and N_0 as to_lane
        # for movement 2: from S_0 (2) to N_0 (10) -> diff is 2 - 10 = -8
        # Let's use separate lane names to make test math unambiguous:
    }

    clean_movements = [
        MovementInfo(link_index=0, from_lane="n0_in", to_lane="s0_out", direction="s"),
        MovementInfo(link_index=1, from_lane="n1_in", to_lane="s1_out", direction="s"),
        MovementInfo(link_index=2, from_lane="s0_in", to_lane="n0_out", direction="s"),
        MovementInfo(link_index=3, from_lane="e0_in", to_lane="w0_out", direction="s"),
        MovementInfo(link_index=4, from_lane="e1_in", to_lane="w1_out", direction="s"),
        MovementInfo(link_index=5, from_lane="w0_in", to_lane="e0_out", direction="s"),
    ]
    lane_vehicle_map = {
        "n0_in": 10, "s0_out": 2,   # diff: 8
        "n1_in": 5, "s1_out": 1,    # diff: 4
        "s0_in": 6, "n0_out": 3,    # diff: 3   -> Phase 0 total = 15.0
        "e0_in": 12, "w0_out": 1,   # diff: 11
        "e1_in": 8, "w1_out": 2,    # diff: 6
        "w0_in": 4, "e0_out": 4,    # diff: 0   -> Phase 1 total = 17.0
    }

    controller = MaxPressureController(
        green_phases=phases,
        movements=clean_movements,
        lane_vehicle_fn=lambda lane_id: lane_vehicle_map.get(lane_id, 0),
    )

    pressures = controller.calculate_phase_pressures()
    assert pressures[0] == 15.0
    assert pressures[1] == 17.0

    # Also verify lowercase 'g' (permissive green) counts as active movement
    g_phases = ["ggGrrr", "rrrGGG"]
    controller_g = MaxPressureController(
        green_phases=g_phases,
        movements=clean_movements,
        lane_vehicle_fn=lambda lane_id: lane_vehicle_map.get(lane_id, 0),
    )
    assert controller_g.calculate_phase_pressures()[0] == 15.0


def test_select_next_phase_decision_rules():
    """Test next phase selection logic: max pressure, ties, non-positive pressure."""
    from src.controllers.max_pressure import MaxPressureController

    phases = ["GGGrrr", "rrrGGG", "rrrrGG"]
    movements = [
        MovementInfo(link_index=0, from_lane="in0", to_lane="out0", direction="s"),
        MovementInfo(link_index=3, from_lane="in1", to_lane="out1", direction="s"),
        MovementInfo(link_index=4, from_lane="in2", to_lane="out2", direction="s"),
    ]

    traffic: Dict[str, int] = {}
    controller = MaxPressureController(
        green_phases=phases,
        movements=movements,
        lane_vehicle_fn=lambda l: traffic.get(l, 0),
    )

    # 1. Phase 1 has strictly higher positive pressure than Phase 0
    traffic.update({"in0": 5, "out0": 0, "in1": 20, "out1": 0, "in2": 0, "out2": 0})
    # Phase 0 pressure = 5, Phase 1 pressure = 20, Phase 2 pressure = 0
    assert controller.select_next_phase() == 1

    # 2. Phase 0 has strictly higher positive pressure than all other phases
    traffic.update({"in0": 25, "out0": 0, "in1": 10, "out1": 0})
    assert controller.select_next_phase() == 0

    # 3. All pressures are <= 0: stay on current phase
    traffic.update({"in0": 0, "out0": 5, "in1": 0, "out1": 10, "in2": 0, "out2": 0})
    # Pressures: {0: -5, 1: -10, 2: 0}
    assert controller.select_next_phase() == 0

    # 4. Max pressure is equal to current phase pressure: stay on current phase
    traffic.update({"in0": 10, "out0": 0, "in1": 10, "out1": 0, "in2": 0, "out2": 0})
    # Phase 0 = 10, Phase 1 = 10 -> stay on current phase (0)
    assert controller.select_next_phase() == 0

    # 5. Forced switch (e.g. at max_green): selects highest alternative even if negative
    traffic.update({"in0": 50, "out0": 0, "in1": 0, "out1": 20, "in2": 0, "out2": 5})
    # Phase 0 = 50, Phase 1 = -20, Phase 2 = -5
    # Force switch must ignore current phase (0) and pick best alternative (Phase 2 with -5)
    assert controller.select_next_phase(force_switch=True) == 2


def test_timing_and_safety_constraints():
    """Test min_green (10s), yellow clearance (3s), and transition to next phase."""
    from src.controllers.max_pressure import MaxPressureController

    phases = ["GGGrrr", "rrrGGG"]
    movements = [
        MovementInfo(link_index=0, from_lane="in0", to_lane="out0", direction="s"),
        MovementInfo(link_index=3, from_lane="in1", to_lane="out1", direction="s"),
    ]

    # Traffic heavily favors Phase 1 right from t = 0
    traffic = {"in0": 0, "out0": 0, "in1": 50, "out1": 0}

    controller = MaxPressureController(
        green_phases=phases,
        movements=movements,
        min_green=10.0,
        max_green=60.0,
        yellow_time=3.0,
        lane_vehicle_fn=lambda l: traffic.get(l, 0),
    )

    states: List[str] = []
    yellow_flags: List[bool] = []
    phase_indices: List[int] = []
    change_flags: List[bool] = []

    # Run 15 steps (1.0s each)
    for _ in range(15):
        s, changed = controller.step(step_duration=1.0)
        states.append(s)
        change_flags.append(changed)
        yellow_flags.append(controller.is_yellow)
        phase_indices.append(controller.current_phase)

    # 1. First 10 steps (indices 0..9): MUST remain in Green Phase 0 ("GGGrrr")
    # Even though Phase 1 has overwhelming pressure, min_green=10s prevents switching.
    assert len(states[:10]) == 10
    assert all(s == "GGGrrr" for s in states[:10])
    assert all(y is False for y in yellow_flags[:10])
    assert all(p == 0 for p in phase_indices[:10])
    assert change_flags[0] is True  # initial activation
    assert all(c is False for c in change_flags[1:10])

    # 2. Next 3 steps (indices 10..12): Yellow clearance phase ("yyyrrr")
    # At t = 10s, min_green is satisfied, controller initiates switch to Phase 1 via yellow.
    assert len(states[10:13]) == 3
    assert all(s == "yyyrrr" for s in states[10:13])
    assert all(y is True for y in yellow_flags[10:13])
    assert change_flags[10] is True  # transitioned to yellow
    assert change_flags[11] is False
    assert change_flags[12] is False

    # 3. Next steps (indices 13..14): Green Phase 1 ("rrrGGG")
    # At t = 13s, yellow clearance is complete, controller enters Phase 1.
    assert states[13] == "rrrGGG"
    assert yellow_flags[13] is False
    assert phase_indices[13] == 1
    assert change_flags[13] is True  # transitioned to green Phase 1
    assert states[14] == "rrrGGG"
    assert yellow_flags[14] is False
    assert phase_indices[14] == 1
    assert change_flags[14] is False


def test_max_green_forced_switch():
    """Test that max_green forces a switch even if current phase has highest pressure."""
    from src.controllers.max_pressure import MaxPressureController

    phases = ["GGGrrr", "rrrGGG"]
    movements = [
        MovementInfo(link_index=0, from_lane="in0", to_lane="out0", direction="s"),
        MovementInfo(link_index=3, from_lane="in1", to_lane="out1", direction="s"),
    ]

    # Traffic heavily favors Phase 0 continuously
    traffic = {"in0": 100, "out0": 0, "in1": 10, "out1": 0}

    controller = MaxPressureController(
        green_phases=phases,
        movements=movements,
        min_green=10.0,
        max_green=60.0,
        yellow_time=3.0,
        lane_vehicle_fn=lambda l: traffic.get(l, 0),
    )

    # Run for 60 steps: should stay in Phase 0 for full max_green duration
    for _ in range(60):
        s, _ = controller.step(step_duration=1.0)
        assert s == "GGGrrr"
        assert controller.is_yellow is False
        assert controller.current_phase == 0

    assert controller.time_in_phase == 60.0

    # Step 61 (at t = 60s elapsed): max_green is reached, MUST force transition to yellow!
    state_61, changed_61 = controller.step(step_duration=1.0)
    assert state_61 == "yyyrrr"
    assert changed_61 is True
    assert controller.is_yellow is True

    # 2 more yellow steps
    controller.step(step_duration=1.0)
    controller.step(step_duration=1.0)

    # Step 64: must enter Phase 1
    state_64, changed_64 = controller.step(step_duration=1.0)
    assert state_64 == "rrrGGG"
    assert changed_64 is True
    assert controller.is_yellow is False
    assert controller.current_phase == 1


def test_reset():
    """Test resetting controller state to initial configuration."""
    from src.controllers.max_pressure import MaxPressureController

    phases = ["GGGrrr", "rrrGGG"]
    movements = _make_dummy_movements()
    traffic = {"N_in_0": 0, "E_in_0": 50}

    controller = MaxPressureController(
        green_phases=phases,
        movements=movements,
        min_green=10.0,
        max_green=60.0,
        yellow_time=3.0,
        lane_vehicle_fn=lambda l: traffic.get(l, 0),
    )

    # Run into Phase 1
    for _ in range(15):
        controller.step(step_duration=1.0)

    assert controller.current_phase == 1
    assert controller.total_time == 15.0
    assert controller.step_count == 15

    # Reset
    controller.reset()
    assert controller.current_phase == 0
    assert controller.get_current_phase() == 0
    assert controller.is_yellow is False
    assert controller.is_in_yellow() is False
    assert controller.time_in_phase == 0.0
    assert controller.total_time == 0.0
    assert controller.step_count == 0
    assert controller.get_current_state() == "GGGrrr"


def test_traci_fallback_graceful_when_not_loaded():
    """Test that controller defaults to 0 vehicle count if TraCI is not running."""
    from src.controllers.max_pressure import MaxPressureController

    phases = ["GGGrrr", "rrrGGG"]
    movements = _make_dummy_movements()

    # lane_vehicle_fn is None, traci is not connected
    controller = MaxPressureController(green_phases=phases, movements=movements)
    pressures = controller.calculate_phase_pressures()

    assert pressures == {0: 0.0, 1: 0.0}
    # With 0 pressure, stays on current phase
    assert controller.select_next_phase() == 0


def test_edge_cases_and_step_durations():
    """Test edge cases: zero yellow time, single phase, custom step durations, invalid step."""
    from src.controllers.max_pressure import MaxPressureController

    # 1. Invalid step duration
    controller = MaxPressureController(green_phases=["GGG"], movements=[])
    with pytest.raises(ValueError, match="step_duration must be positive"):
        controller.step(step_duration=0.0)
    with pytest.raises(ValueError, match="step_duration must be positive"):
        controller.step(step_duration=-1.0)

    # 2. Single phase: never switches, stays green
    single_p_controller = MaxPressureController(
        green_phases=["GGGG"],
        movements=[MovementInfo(link_index=0, from_lane="in", to_lane="out", direction="s")],
        min_green=5.0,
        max_green=10.0,
        yellow_time=2.0,
        lane_vehicle_fn=lambda l: 10,
    )
    for _ in range(20):
        state, changed = single_p_controller.step(step_duration=1.0)
        assert state == "GGGG"
        assert single_p_controller.current_phase == 0
        assert single_p_controller.is_yellow is False

    # 3. Zero yellow time: switches instantly from green 0 to green 1
    no_yellow_controller = MaxPressureController(
        green_phases=["GGG", "rrrGGG"],
        movements=[
            MovementInfo(link_index=0, from_lane="in0", to_lane="out0", direction="s"),
            MovementInfo(link_index=3, from_lane="in1", to_lane="out1", direction="s"),
        ],
        min_green=5.0,
        max_green=15.0,
        yellow_time=0.0,
        lane_vehicle_fn=lambda l: 20 if l == "in0" else (0 if l != "in1" else 5),
    )
    # Force switch: run 5 steps
    for _ in range(5):
        no_yellow_controller.step(step_duration=1.0)
    # At t=5, min_green is met; phase 1 now has higher positive pressure:
    no_yellow_controller._lane_vehicle_fn = lambda l: 30 if l == "in1" else 0
    state, changed = no_yellow_controller.step(step_duration=1.0)
    # Directly switched to phase 1 without yellow
    assert state == "rrrGGG"
    assert changed is True
    assert no_yellow_controller.current_phase == 1
    assert no_yellow_controller.is_yellow is False

    # 4. Custom step duration (e.g. 0.5s)
    half_step_ctrl = MaxPressureController(
        green_phases=["GG", "rr"],
        movements=[],
        min_green=4.0,
        max_green=10.0,
        yellow_time=2.0,
    )
    for _ in range(4):
        half_step_ctrl.step(step_duration=0.5)
    assert half_step_ctrl.time_in_phase == 2.0
    assert half_step_ctrl.total_time == 2.0
    assert half_step_ctrl.step_count == 4
