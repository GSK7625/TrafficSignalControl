"""Unit tests reproducing Q-learning training and action selection defects (TDD).

Tests:
1. test_greedy_policy_does_not_blindly_prefer_unvisited_empty_phases:
   Verifies that in greedy mode, the controller does not abandon an active green
   phase with queued vehicles to switch to an empty competing phase just because
   the unvisited competing phase has default Q = 0.0 > -0.5.
2. test_action_attribution_when_switching:
   Verifies that when a switch action a != curr_phase is selected, the Q-table
   updates Q(s, a) rather than incorrectly attributing the transition to curr_phase.
3. test_controller_decision_point_metadata:
   Verifies that QLearningController explicitly signals decision points so the
   training loop can perform exact SMDP Bellman updates without guesswork.
"""

import pytest
from src.controllers.q_learning import CompactStateDiscretizer, QLearningController


class MockMovement:
    def __init__(self, link_index: int, from_lane: str, to_lane: str = ""):
        self.link_index = link_index
        self.from_lane = from_lane
        self.to_lane = to_lane


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
    # Simulate that phase 0 was tried and received negative reward (-0.5),
    # while phases 1, 2, 3 have unvisited default value 0.0
    ctrl.q_table[s] = [-0.5, 0.0, 0.0, 0.0]

    # In greedy mode, choosing action should KEEP phase 0 because competing phases are empty
    chosen = ctrl.choose_action(s, explore=False)
    assert chosen == 0, f"Expected action 0 (keep busy phase), but got {chosen} (switched to empty phase)!"


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


def test_training_loop_updates_switch_actions(tmp_path):
    """Training with exploration must update switch actions (a != curr_phase) in Q-table."""
    from src.training.train_q_learning import train_q_learning_agent

    save_file = str(tmp_path / "test_model.json")
    ctrl, stats = train_q_learning_agent(
        scenario_name="ingolstadt1",
        data_dir="data",
        episodes=1,
        max_steps_per_episode=100,
        save_path=save_file,
        lr=0.2,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_min=1.0,  # Force pure exploration
        min_green=5.0,
        decision_interval=5.0,
        verbose=False,
    )

    # Check non-zero updates where action != state[0]
    switch_updates = 0
    for state, q_vals in ctrl.q_table.items():
        curr_p = state[0]
        for a, val in enumerate(q_vals):
            if a != curr_p and val != 0.0:
                switch_updates += 1

    assert switch_updates > 0, "No switch action (a != curr_phase) was ever updated during training!"

