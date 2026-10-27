"""Unit tests for QLearningController and CompactStateDiscretizer."""

import os
import tempfile
import pytest

from src.controllers.q_learning import CompactStateDiscretizer, QLearningController


class MockMovement:
    def __init__(self, link_index: int, from_lane: str, to_lane: str = ""):
        self.link_index = link_index
        self.from_lane = from_lane
        self.to_lane = to_lane


@pytest.fixture
def sample_setup():
    # 2 green phases: Phase 0 (East-West), Phase 1 (North-South)
    phases = ["GGGrrrGGGrrr", "rrrGGGrrrGGG"]
    movements = [
        MockMovement(0, "E2_0", "W1_0"),
        MockMovement(1, "E2_1", "W1_1"),
        MockMovement(2, "E2_2", "W1_2"),
        MockMovement(6, "W2_0", "E1_0"),
        MockMovement(7, "W2_1", "E1_1"),
        MockMovement(8, "W2_2", "E1_2"),
        MockMovement(3, "N2_0", "S1_0"),
        MockMovement(4, "N2_1", "S1_1"),
        MockMovement(5, "N2_2", "S1_2"),
        MockMovement(9, "S2_0", "N1_0"),
        MockMovement(10, "S2_1", "N1_1"),
        MockMovement(11, "S2_2", "N1_2"),
    ]
    return phases, movements


def test_compact_state_discretizer():
    """Verify CompactStateDiscretizer computes expected 4-element compact tuple."""
    discretizer = CompactStateDiscretizer(
        serving_threshold=2.0,
        competing_threshold=6.0,
    )
    # Phase 0, serving queue = 10 (>= 2.0 -> 1),
    # phase 1 queue = 25 (strongest competing=1, >= 6.0 -> 1)
    phase_queues = {0: 10.0, 1: 25.0}
    state = discretizer.discretize(
        curr_phase=0,
        time_in_phase=15.0,
        phase_queues=phase_queues,
    )
    # Expected: (curr_phase=0, serving_bin=1, competing_phase=1, competing_bin=1)
    assert state == (0, 1, 1, 1)

    # Empty queue condition
    empty_queues = {0: 0.0, 1: 1.0}
    state_empty = discretizer.discretize(
        curr_phase=0,
        time_in_phase=20.0,
        phase_queues=empty_queues,
    )
    # serving_q=0 (< 2.0 -> 0), competing_q=1 (< 6.0 -> 0)
    assert state_empty == (0, 0, 1, 0)


def test_compact_state_discretizer_backward_compatibility():
    """Verify CompactStateDiscretizer handles legacy threshold parameters cleanly."""
    discretizer = CompactStateDiscretizer(
        time_thresholds=(10.0, 25.0, 45.0),
        queue_thresholds=(2.0, 8.0, 20.0),
    )
    assert discretizer.serving_threshold == 2.0
    assert discretizer.competing_threshold == 8.0
    state = discretizer.discretize(
        curr_phase=0,
        time_in_phase=15.0,
        phase_queues={0: 5.0, 1: 15.0},
    )
    assert state == (0, 1, 1, 1)


def test_q_learning_controller_initialization(sample_setup):
    """Verify QLearningController initial values and parameters."""
    phases, movements = sample_setup
    ctrl = QLearningController(
        green_phases=phases,
        movements=movements,
        min_green=10.0,
        max_green=45.0,
        yellow_time=3.0,
        all_red_time=1.0,
        max_red=90.0,
        lr=0.1,
        gamma=0.95,
        epsilon=0.1,
    )
    assert ctrl.num_phases == 2
    assert ctrl.current_phase == 0
    assert not ctrl.is_yellow
    assert not ctrl.is_all_red
    assert ctrl.min_green == 10.0
    assert ctrl.max_green == 45.0
    assert ctrl.all_red_time == 1.0
    assert ctrl.max_red == 90.0
    assert ctrl.time_in_phase == 0.0
    assert ctrl.step_count == 0


def test_q_table_update_bellman_equation(sample_setup):
    """Verify Bellman equation update: Q(s, a) <- Q + lr * [r + gamma * max Q(s', a') - Q]."""
    phases, movements = sample_setup
    ctrl = QLearningController(
        green_phases=phases,
        movements=movements,
        lr=0.2,
        gamma=0.9,
        epsilon=0.0,
    )
    s = (0, 1, 1, 1, 2)
    a = 1
    r = 10.0
    s_next = (1, 0, 2, 0, 1)

    # Pre-set Q(s_next, 0) = 4.0, Q(s_next, 1) = 6.0 -> max is 6.0
    ctrl.q_table[s_next] = [4.0, 6.0]
    # Initial Q(s, a) = 0.0
    # Expected: 0.0 + 0.2 * (10.0 + 0.9 * 6.0 - 0.0) = 0.2 * 15.4 = 3.08
    ctrl.update_q(s, a, r, s_next)
    assert ctrl.q_table[s][a] == pytest.approx(3.08)


def test_choose_action_greedy_vs_explore(sample_setup):
    """Verify greedy selection selects argmax while explore=True obeys epsilon."""
    phases, movements = sample_setup
    ctrl = QLearningController(
        green_phases=phases,
        movements=movements,
        epsilon=0.0,
    )
    s = (0, 1, 1, 1, 2)
    ctrl.q_table[s] = [2.5, 7.8]
    # Greedy must select action 1
    assert ctrl.choose_action(s, explore=False) == 1

    ctrl.q_table[s] = [10.2, 1.1]
    # Greedy must select action 0
    assert ctrl.choose_action(s, explore=False) == 0


def test_safety_min_green_enforcement(sample_setup):
    """Verify controller does not allow switching phase before min_green seconds."""
    phases, movements = sample_setup
    ctrl = QLearningController(
        green_phases=phases,
        movements=movements,
        min_green=10.0,
        decision_interval=5.0,
        yellow_time=3.0,
    )
    # Even if Q-table strongly prefers action 1
    s = ctrl.get_state()
    ctrl.q_table[s] = [-100.0, 100.0]

    # For steps 1 to 9 (time_in_phase < 10.0), controller must stay on phase 0
    for _ in range(1, 10):
        state, changed = ctrl.step(1.0)
        assert ctrl.current_phase == 0
        assert not ctrl.is_yellow
        assert state == phases[0]


def test_safety_yellow_clearance_transition(sample_setup):
    """Verify yellow clearance of yellow_time seconds occurs when switching."""
    from collections import defaultdict

    phases, movements = sample_setup
    ctrl = QLearningController(
        green_phases=phases,
        movements=movements,
        min_green=5.0,
        decision_interval=5.0,
        yellow_time=3.0,
        all_red_time=0.0,
    )
    # Step 4s in phase 0
    for _ in range(4):
        ctrl.step(1.0)

    # Force action = 1 on decision point
    ctrl.q_table = defaultdict(lambda: [-100.0, 100.0])

    # Step at t=5s (min_green reached) -> triggers switch to phase 1
    state, changed = ctrl.step(1.0)
    assert ctrl.is_yellow
    assert "y" in state.lower()
    assert changed

    # Step at t=6s (yellow second 1)
    state, _ = ctrl.step(1.0)
    assert ctrl.is_yellow

    # Step at t=7s (yellow second 2)
    state, _ = ctrl.step(1.0)
    assert ctrl.is_yellow

    # Step at t=8s (yellow 3s completes, green phase 1 starts)
    state, changed = ctrl.step(1.0)
    assert not ctrl.is_yellow
    assert ctrl.current_phase == 1
    assert state == phases[1]
    assert changed


def test_safety_all_red_clearance_transition(sample_setup):
    """Verify Green -> Yellow (3s) -> All-Red (1s) -> Next Green sequence."""
    from collections import defaultdict

    phases, movements = sample_setup
    ctrl = QLearningController(
        green_phases=phases,
        movements=movements,
        min_green=5.0,
        decision_interval=5.0,
        yellow_time=3.0,
        all_red_time=1.0,
    )
    for _ in range(4):
        ctrl.step(1.0)

    # Force switch to phase 1
    ctrl.q_table = defaultdict(lambda: [-100.0, 100.0])

    # t=5: yellow starts
    _, c5 = ctrl.step(1.0)
    assert ctrl.is_yellow and not ctrl.is_all_red and c5

    # t=6, 7: yellow continues
    ctrl.step(1.0)
    ctrl.step(1.0)

    # t=8: yellow complete -> all-red starts!
    s8, c8 = ctrl.step(1.0)
    assert not ctrl.is_yellow
    assert ctrl.is_all_red
    assert s8 == "r" * len(phases[0])
    assert c8

    # t=9: all-red complete -> next green starts!
    s9, c9 = ctrl.step(1.0)
    assert not ctrl.is_yellow
    assert not ctrl.is_all_red
    assert ctrl.current_phase == 1
    assert s9 == phases[1]
    assert c9


def test_q_learning_max_red_starvation_override(sample_setup):
    """Verify max_red (90s) overrides greedy action to serve starved phase."""
    from collections import defaultdict

    phases, movements = sample_setup
    ctrl = QLearningController(
        green_phases=phases,
        movements=movements,
        min_green=10.0,
        max_green=45.0,
        yellow_time=3.0,
        all_red_time=1.0,
        max_red=90.0,
    )
    # Q-table strongly prefers staying in phase 0
    ctrl.q_table = defaultdict(lambda: [100.0, -100.0])

    # Advance 10s of min_green in phase 0
    for _ in range(10):
        ctrl.step(1.0)

    # Simulate Phase 1 starving at 90s red wait
    ctrl._phase_red_timers[1] = 90.0

    # Step at t=11: starvation watchdog triggers switch to phase 1!
    s11, c11 = ctrl.step(1.0)
    assert ctrl.is_yellow
    assert ctrl._next_phase_idx == 1
    assert c11



def test_safety_max_green_forced_switch(sample_setup):
    """Verify max_green forces a switch even if Q-table prefers current phase."""
    from collections import defaultdict

    phases, movements = sample_setup
    ctrl = QLearningController(
        green_phases=phases,
        movements=movements,
        min_green=10.0,
        max_green=20.0,
        yellow_time=3.0,
    )
    # Q-table always prefers keeping phase 0
    ctrl.q_table = defaultdict(lambda: [100.0, -100.0])

    # Step for 19 seconds (time_in_phase reaches 19s, still green)
    for _ in range(19):
        state, changed = ctrl.step(1.0)
        assert not ctrl.is_yellow
        assert ctrl.current_phase == 0

    # Step at t=20s (max_green reached): must trigger yellow transition
    state, changed = ctrl.step(1.0)
    assert ctrl.is_yellow
    assert changed


def test_save_and_load_q_table(sample_setup):
    """Verify saving Q-table and reloading reproduces exact Q-values."""
    phases, movements = sample_setup
    ctrl1 = QLearningController(
        green_phases=phases,
        movements=movements,
        lr=0.15,
        gamma=0.92,
        epsilon=0.05,
    )
    s = (0, 2, 1, 1, 3)
    ctrl1.q_table[s] = [3.14, 2.71]

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "q_model.json")
        ctrl1.save(model_path)

        assert os.path.isfile(model_path)

        ctrl2 = QLearningController(
            green_phases=phases,
            movements=movements,
        )
        ctrl2.load(model_path)

        assert ctrl2.lr == 0.15
        assert ctrl2.gamma == 0.92
        assert ctrl2.epsilon == 0.05
        assert ctrl2.q_table[s] == [3.14, 2.71]


def test_reset_preserves_q_table(sample_setup):
    """Verify reset() clears phase timing and counters but retains learned Q-table."""
    phases, movements = sample_setup
    ctrl = QLearningController(
        green_phases=phases,
        movements=movements,
    )
    s = (0, 1, 2, 1, 1)
    ctrl.q_table[s] = [5.5, 9.9]

    for _ in range(12):
        ctrl.step(1.0)

    assert ctrl.step_count == 12
    assert ctrl.total_time == 12.0

    ctrl.reset()
    assert ctrl.step_count == 0
    assert ctrl.total_time == 0.0
    assert ctrl.current_phase == 0
    assert not ctrl.is_yellow
    assert ctrl.q_table[s] == [5.5, 9.9]


@pytest.fixture
def four_phase_controller():
    phases = ["GGrr", "rrGG", "GrGr", "rGrG"]
    movements = [
        MockMovement(0, "lane_0"),
        MockMovement(1, "lane_1"),
        MockMovement(2, "lane_2"),
        MockMovement(3, "lane_3"),
    ]
    # Vehicle count mocking: lane_0 has 10 vehicles, others have 0
    vehicle_map = {"lane_0": 10, "lane_1": 0, "lane_2": 0, "lane_3": 0}
    ctrl = QLearningController(
        green_phases=phases,
        movements=movements,
        min_green=10.0,
        max_green=60.0,
        decision_interval=5.0,
        yellow_time=3.0,
        lane_vehicle_fn=lambda l: vehicle_map.get(l, 0),
    )
    return ctrl


def test_greedy_policy_does_not_blindly_prefer_unvisited_empty_phases(four_phase_controller):
    """In greedy mode, if curr_phase=0 has 10 vehicles and Q=-0.5, but unvisited phases 1,2,3
    have 0 vehicles and default Q=0.0, the controller must NOT switch to an empty phase.
    """
    ctrl = four_phase_controller
    s = ctrl.get_state()
    ctrl.q_table[s] = [-0.5, 0.0, 0.0, 0.0]

    # In greedy mode, choosing action should KEEP phase 0 because competing phases are empty
    chosen = ctrl.choose_action(s, explore=False)
    assert chosen == 0, f"Expected action 0 (keep busy phase), but got {chosen}!"


def test_controller_decision_point_metadata(four_phase_controller):
    """QLearningController should track whether the current step was an RL decision step
    and what (state, action) was decided, enabling clean SMDP updates.
    """
    ctrl = four_phase_controller
    assert ctrl.last_decision is None

    # Step for 9 seconds (less than min_green 10s): no decision should be made
    for _ in range(9):
        ctrl.step(1.0)
        assert ctrl.last_decision is None

    # At step 10 (min_green reached): RL decision must occur
    state, changed = ctrl.step(1.0)
    assert ctrl.last_decision is not None
    dec_state, dec_action = ctrl.last_decision
    assert isinstance(dec_state, tuple)
    assert isinstance(dec_action, int)

