"""Unit tests for FixedTime, MaxPressure, and QLearning controllers."""

import os
import tempfile
import pytest
import numpy as np

from src.controllers.base import ControlledLink, IntersectionObservation
from src.controllers.fixed_time import FixedTimeController
from src.controllers.max_pressure import MaxPressureController, compute_phase_pressure
from src.controllers.q_learning import QLearningController


def create_dummy_observation(
    tls_id: str = "tls_1",
    current_phase: int = 0,
    time_in_phase: float = 10.0,
    is_in_transition: bool = False,
    green_phases: list = None,
    controlled_links: list = None,
    in_lane_queues: dict = None,
    out_lane_queues: dict = None,
    sim_time: float = 100.0,
    step: int = 100,
) -> IntersectionObservation:
    """Helper to create a standard IntersectionObservation for testing."""
    if green_phases is None:
        green_phases = ["GGGrrr", "rrrGGG"]
    if controlled_links is None:
        controlled_links = [
            ControlledLink(signal_index=0, from_lane="in_0", to_lane="out_0"),
            ControlledLink(signal_index=1, from_lane="in_0", to_lane="out_1"),
            ControlledLink(signal_index=2, from_lane="in_1", to_lane="out_2"),
            ControlledLink(signal_index=3, from_lane="in_2", to_lane="out_3"),
            ControlledLink(signal_index=4, from_lane="in_2", to_lane="out_4"),
            ControlledLink(signal_index=5, from_lane="in_3", to_lane="out_5"),
        ]
    if in_lane_queues is None:
        in_lane_queues = {"in_0": 0.0, "in_1": 0.0, "in_2": 0.0, "in_3": 0.0}
    if out_lane_queues is None:
        out_lane_queues = {"out_0": 0.0, "out_1": 0.0, "out_2": 0.0, "out_3": 0.0, "out_4": 0.0, "out_5": 0.0}

    return IntersectionObservation(
        tls_id=tls_id,
        current_phase=current_phase,
        time_in_phase=time_in_phase,
        is_in_transition=is_in_transition,
        green_phases=green_phases,
        controlled_links=controlled_links,
        in_lane_queues=in_lane_queues,
        out_lane_queues=out_lane_queues,
        in_lane_vehicles={k: v for k, v in in_lane_queues.items()},
        out_lane_vehicles={k: v for k, v in out_lane_queues.items()},
        sim_time=sim_time,
        step=step,
    )


class TestFixedTimeController:
    """Test suite for FixedTimeController."""

    def test_invalid_default_green_time(self):
        with pytest.raises(ValueError, match="default_green_time must be positive"):
            FixedTimeController(default_green_time=0.0)

    def test_normal_cycle_progression(self):
        controller = FixedTimeController(default_green_time=30.0)
        obs = create_dummy_observation(current_phase=0, time_in_phase=10.0)

        # Before 30s elapsed: holds current phase 0
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 0

        # At 30s: cycles to next phase (phase 1)
        obs.time_in_phase = 30.0
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 1

        # While transitioning to phase 1: keeps requesting target phase 1
        obs.is_in_transition = True
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 1

    def test_custom_sequence_and_offsets(self):
        controller = FixedTimeController(
            default_green_time=20.0,
            phase_sequences={"tls_1": [1, 0]},
            offsets={"tls_1": 10.0},
        )
        # Sequence starts at phase 1
        obs = create_dummy_observation(current_phase=1, time_in_phase=0.0)
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 1

        # Effective duration for first cycle is 20 + 10 = 30s
        obs.time_in_phase = 25.0
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 1

        # At 30s: cycles to phase 0, offset consumed
        obs.time_in_phase = 30.0
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 0

    def test_input_validation_nan_and_inf(self):
        with pytest.raises(ValueError, match="positive and finite"):
            FixedTimeController(default_green_time=float("nan"))

        with pytest.raises(ValueError, match="positive and finite"):
            FixedTimeController(default_green_time=float("inf"))

        with pytest.raises(ValueError, match="positive and finite"):
            FixedTimeController(default_green_time=-5.0)

    def test_input_validation_phase_durations(self):
        with pytest.raises(ValueError, match="positive and finite"):
            FixedTimeController(phase_durations={"tls_1": {0: float("nan")}})

        with pytest.raises(ValueError, match="positive and finite"):
            FixedTimeController(phase_durations={"tls_1": {0: float("inf")}})

        with pytest.raises(ValueError, match="positive and finite"):
            FixedTimeController(phase_durations={"tls_1": {0: -10.0}})

        with pytest.raises(ValueError, match="non-negative integer"):
            FixedTimeController(phase_durations={"tls_1": {-1: 30.0}})

        with pytest.raises(ValueError, match="non-negative integer"):
            FixedTimeController(phase_durations={"tls_1": {"not_int": 30.0}})

    def test_input_validation_phase_sequences(self):
        with pytest.raises(ValueError, match="non-empty"):
            FixedTimeController(phase_sequences={"tls_1": []})

        with pytest.raises(ValueError, match="non-negative integer"):
            FixedTimeController(phase_sequences={"tls_1": [-1, 0]})

        with pytest.raises(ValueError, match="non-negative integer"):
            FixedTimeController(phase_sequences={"tls_1": ["0", 1]})

    def test_input_validation_offsets(self):
        with pytest.raises(ValueError, match="non-negative finite"):
            FixedTimeController(offsets={"tls_1": float("nan")})

        with pytest.raises(ValueError, match="non-negative finite"):
            FixedTimeController(offsets={"tls_1": float("inf")})

        with pytest.raises(ValueError, match="non-negative finite"):
            FixedTimeController(offsets={"tls_1": -5.0})

    def test_schedule_resynchronization_on_forced_phase_switch(self):
        controller = FixedTimeController(default_green_time=90.0)
        # Normal initial step: Phase 0 active
        obs0 = create_dummy_observation(current_phase=0, time_in_phase=59.0)
        actions0 = controller.select_action({"tls_1": obs0})
        assert actions0["tls_1"] == 0

        # Forced switch to Phase 1 occurs (e.g. runner forced by max_green=60)
        # When Phase 1 becomes active green, controller must sync to Phase 1
        # and NOT revert immediately to Phase 0
        obs1 = create_dummy_observation(current_phase=1, time_in_phase=0.0, is_in_transition=False)
        actions1 = controller.select_action({"tls_1": obs1})
        assert actions1["tls_1"] == 1, "Controller should hold forced Phase 1, not revert to Phase 0"

        # At time_in_phase=5.0 (min_green reached), controller should STILL hold Phase 1
        obs1_5 = create_dummy_observation(current_phase=1, time_in_phase=5.0, is_in_transition=False)
        actions1_5 = controller.select_action({"tls_1": obs1_5})
        assert actions1_5["tls_1"] == 1, "Controller should continue holding Phase 1, avoiding the 5s desync glitch"

    def test_compatibility_validation_with_max_and_min_green(self):
        controller = FixedTimeController(default_green_time=90.0)
        # With max_green=60, 90s is incompatible
        errors = controller.validate_compatibility(min_green=5.0, max_green=60.0)
        assert len(errors) > 0
        assert any("exceeds max_green" in e for e in errors)

        # Offsets causing duration + offset > max_green
        controller_offset = FixedTimeController(default_green_time=40.0, offsets={"tls_1": 30.0})
        errors_offset = controller_offset.validate_compatibility(min_green=5.0, max_green=60.0)
        assert any("exceeds max_green" in e for e in errors_offset)

        # Invalid phase sequence index
        controller_seq = FixedTimeController(phase_sequences={"tls_1": [0, 3]})
        errors_seq = controller_seq.validate_compatibility(num_phases_per_tls={"tls_1": 2})
        assert any("out of bounds" in e or "invalid" in e for e in errors_seq)



class TestMaxPressureController:
    """Test suite for MaxPressureController and compute_phase_pressure."""

    def test_compute_phase_pressure_empty_inputs(self):
        assert compute_phase_pressure("", [], {}, {}) == 0.0

    def test_compute_phase_pressure_math(self):
        links = [
            ControlledLink(signal_index=0, from_lane="in_A", to_lane="out_A"),
            ControlledLink(signal_index=1, from_lane="in_A", to_lane="out_B"),
            ControlledLink(signal_index=2, from_lane="in_C", to_lane="out_C"),
        ]
        in_q = {"in_A": 10.0, "in_C": 4.0}
        out_q = {"out_A": 2.0, "out_B": 0.0, "out_C": 1.0}

        # Phase 0: 'GGr' -> Serves in_A to out_A and out_B (both links green)
        # in_A has 2 total out links, 2 served -> ratio = 1.0 -> inflow = 10.0
        # out lanes served: out_A (q=2) and out_B (q=0) -> backpressure = 2.0
        # Expected pressure = 10.0 - 2.0 = 8.0
        p0 = compute_phase_pressure("GGr", links, in_q, out_q)
        assert pytest.approx(p0, 1e-6) == 8.0

        # Phase 1: 'rrG' -> Serves in_C to out_C
        # inflow = 4.0, outflow = 1.0 -> Expected pressure = 3.0
        p1 = compute_phase_pressure("rrG", links, in_q, out_q)
        assert pytest.approx(p1, 1e-6) == 3.0

    def test_min_green_constraint_held(self):
        controller = MaxPressureController(min_green=5.0)
        # Even if Phase 1 has huge queue, min_green must hold current phase 0
        in_q = {"in_0": 0.0, "in_1": 0.0, "in_2": 20.0, "in_3": 20.0}
        obs = create_dummy_observation(
            current_phase=0, time_in_phase=3.0, in_lane_queues=in_q
        )
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 0

    def test_switches_to_max_pressure_phase(self):
        controller = MaxPressureController(min_green=5.0)
        # Phase 0: 'GGGrrr' (in_0, in_1) queue = 2
        # Phase 1: 'rrrGGG' (in_2, in_3) queue = 15
        in_q = {"in_0": 1.0, "in_1": 1.0, "in_2": 8.0, "in_3": 7.0}
        obs = create_dummy_observation(
            current_phase=0, time_in_phase=6.0, in_lane_queues=in_q
        )
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 1

    def test_tie_breaking_keeps_current_phase(self):
        controller = MaxPressureController(min_green=5.0)
        # All queues 0 -> pressures are tied at 0.0
        # Should retain current_phase=0 to prevent signal chatter
        obs = create_dummy_observation(current_phase=0, time_in_phase=6.0)
        actions = controller.select_action({"tls_1": obs})
        assert actions["tls_1"] == 0


class TestQLearningController:
    """Test suite for QLearningController."""

    def test_initialization_validation(self):
        with pytest.raises(ValueError, match="num_actions must be positive"):
            QLearningController(num_actions=0)

        with pytest.raises(ValueError, match="learning_rate must be in"):
            QLearningController(num_actions=2, learning_rate=0.0)

        with pytest.raises(ValueError, match="gamma must be in"):
            QLearningController(num_actions=2, gamma=-0.1)

    def test_action_selection_epsilon_greedy(self):
        ctrl = QLearningController(num_actions=3, epsilon=0.0, eval_mode=True, tie_break_strategy="first")
        state = (0, 1, 2, 1)
        # Populate Q-table
        ctrl.q_table[state] = np.array([0.5, 2.5, 1.0], dtype=np.float64)

        # Greedy choice: action 1 has highest Q (2.5)
        action = ctrl.select_action_for_state(state)
        assert action == 1

        # Valid action masking: only actions [0, 2] valid
        action_masked = ctrl.select_action_for_state(state, valid_actions=[0, 2])
        assert action_masked == 2  # Among 0 (0.5) and 2 (1.0), 2 is higher

    def test_bellman_update(self):
        ctrl = QLearningController(num_actions=2, learning_rate=0.5, gamma=0.9, epsilon=0.0)
        s0 = (0, 0, 0, 1)
        s1 = (1, 0, 0, 1)

        ctrl.q_table[s0] = np.array([0.0, 0.0], dtype=np.float64)
        ctrl.q_table[s1] = np.array([10.0, 5.0], dtype=np.float64)

        # Action 0 taken in s0, reward = 2.0, next_state = s1
        # target = 2.0 + 0.9 * max(10.0, 5.0) = 2.0 + 9.0 = 11.0
        # Q(s0, 0) <- 0.0 + 0.5 * (11.0 - 0.0) = 5.5
        td_error = ctrl.update(state=s0, action=0, reward=2.0, next_state=s1, done=False)
        assert pytest.approx(ctrl.q_table[s0][0], 1e-6) == 5.5
        assert pytest.approx(td_error, 1e-6) == 11.0

        # Terminal state update (done=True): no bootstrapping
        td_term = ctrl.update(state=s0, action=1, reward=4.0, next_state=s1, done=True)
        # target = 4.0 -> Q(s0, 1) <- 0.0 + 0.5 * (4.0 - 0.0) = 2.0
        assert pytest.approx(ctrl.q_table[s0][1], 1e-6) == 2.0
        assert pytest.approx(td_term, 1e-6) == 4.0

    def test_epsilon_decay(self):
        ctrl = QLearningController(num_actions=2, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
        assert ctrl.decay_epsilon() == 0.5
        assert ctrl.decay_epsilon() == 0.25
        assert ctrl.decay_epsilon() == 0.125
        # Floor reached
        assert ctrl.decay_epsilon() == 0.1

    def test_save_and_load(self):
        ctrl = QLearningController(num_actions=2, learning_rate=0.2, gamma=0.95)
        s = (0, 1, 1, 1)
        ctrl.q_table[s] = np.array([1.23, 4.56], dtype=np.float64)
        ctrl.total_updates = 42

        with tempfile.TemporaryDirectory() as tmpdir:
            model_file = os.path.join(tmpdir, "model.json")
            ctrl.save(model_file)
            assert os.path.isfile(model_file)

            new_ctrl = QLearningController(num_actions=2)
            new_ctrl.load(model_file)

            assert new_ctrl.num_actions == 2
            assert new_ctrl.learning_rate == 0.2
            assert new_ctrl.gamma == 0.95
            assert new_ctrl.total_updates == 42
            assert s in new_ctrl.q_table
            assert pytest.approx(new_ctrl.q_table[s][0], 1e-6) == 1.23
            assert pytest.approx(new_ctrl.q_table[s][1], 1e-6) == 4.56
