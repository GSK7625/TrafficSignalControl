"""Unit tests for RL components: rewards, discretization, and observations."""

import numpy as np
import pytest

from src.controllers.base import ControlledLink, IntersectionObservation
from src.rl.discretization import StateDiscretizer
from src.rl.observations import ObservationBuilder
from src.rl.rewards import NegativeAverageQueueReward


def create_dummy_observation(
    current_phase: int = 0,
    time_in_phase: float = 6.0,
    is_in_transition: bool = False,
    in_queues: dict = None,
) -> IntersectionObservation:
    """Creates a mock IntersectionObservation for unit testing."""
    if in_queues is None:
        in_queues = {"lane_1": 2.0, "lane_2": 8.0, "lane_3": 0.0}

    links = [
        ControlledLink(signal_index=0, from_lane="lane_1", to_lane="out_1"),
        ControlledLink(signal_index=1, from_lane="lane_2", to_lane="out_2"),
        ControlledLink(signal_index=2, from_lane="lane_3", to_lane="out_3"),
    ]

    green_phases = ["GGg", "rrr", "rGr"]

    return IntersectionObservation(
        tls_id="tls_test",
        current_phase=current_phase,
        time_in_phase=time_in_phase,
        is_in_transition=is_in_transition,
        green_phases=green_phases,
        controlled_links=links,
        in_lane_queues=in_queues,
        out_lane_queues={"out_1": 0.0, "out_2": 1.0, "out_3": 0.0},
        in_lane_vehicles={"lane_1": 3.0, "lane_2": 10.0, "lane_3": 0.0},
        out_lane_vehicles={"out_1": 1.0, "out_2": 2.0, "out_3": 0.0},
        sim_time=100.0,
        step=10,
    )


class TestRLRewards:
    """Tests for NegativeAverageQueueReward."""

    def test_empty_reward_returns_zero(self):
        reward_fn = NegativeAverageQueueReward()
        assert reward_fn.calculate_reward() == 0.0

    def test_average_queue_calculation(self):
        reward_fn = NegativeAverageQueueReward(scale=1.0)
        # Step 1: queue sum = 10.0
        reward_fn.record_step({"l1": 6.0, "l2": 4.0})
        # Step 2: queue sum = 20.0
        reward_fn.record_step({"l1": 12.0, "l2": 8.0})

        # Average queue = 15.0 -> Reward = -15.0
        reward = reward_fn.calculate_reward()
        assert reward == -15.0

    def test_reset_clears_records(self):
        reward_fn = NegativeAverageQueueReward()
        reward_fn.record_step({"l1": 5.0})
        reward_fn.reset()
        assert reward_fn.calculate_reward() == 0.0

    def test_scaling_factor(self):
        reward_fn = NegativeAverageQueueReward(scale=0.1)
        reward_fn.record_step({"l1": 10.0})
        assert reward_fn.calculate_reward() == -1.0


class TestRLDiscretization:
    """Tests for StateDiscretizer."""

    def test_discretization_output_format(self):
        discretizer = StateDiscretizer(min_green=5.0, max_effective_green=25.0)
        obs = create_dummy_observation(current_phase=0, time_in_phase=10.0)

        # In phase 0 ("GGg"):
        # lane_1 (signal 0 'G') -> queue 2.0 (Low bin 1)
        # lane_2 (signal 1 'G') -> queue 8.0 (High bin 2)
        # lane_3 (signal 2 'g') -> queue 0.0
        # Total serving queue = 10.0 -> High (bin 2)
        # Waiting queue = 0.0 -> None (bin 0)
        # Time = 10.0 -> bin 1 (between 5 and 25)
        state = discretizer.discretize(obs)
        assert isinstance(state, tuple)
        assert len(state) == 4
        assert state[0] == 0  # phase
        assert state[1] == 2  # serving queue >= 5.0
        assert state[2] == 0  # waiting queue < 1.0
        assert state[3] == 1  # time in [5.0, 25.0)

    def test_discretization_min_green_not_met(self):
        discretizer = StateDiscretizer(min_green=5.0)
        obs = create_dummy_observation(time_in_phase=3.0)
        state = discretizer.discretize(obs)
        assert state[3] == 0  # time bin 0 when < min_green

    def test_discretization_in_transition(self):
        discretizer = StateDiscretizer(min_green=5.0)
        obs = create_dummy_observation(time_in_phase=15.0, is_in_transition=True)
        state = discretizer.discretize(obs)
        assert state[3] == 0  # time bin 0 when transitioning


class TestRLObservations:
    """Tests for ObservationBuilder."""

    def test_observation_vector_shape_and_bounds(self):
        lanes = ["lane_1", "lane_2", "lane_3"]
        builder = ObservationBuilder(
            incoming_lanes=lanes,
            num_green_phases=3,
            min_green=5.0,
            max_queue_scale=20.0,
            max_time_scale=60.0,
        )

        expected_dim = len(lanes) + 3 + 3  # 3 lanes + 3 phases + 3 scalars = 9
        assert builder.observation_dim == expected_dim

        obs = create_dummy_observation(current_phase=1, time_in_phase=30.0)
        vec = builder.build_vector(obs)

        assert isinstance(vec, np.ndarray)
        assert vec.shape == (expected_dim,)
        assert vec.dtype == np.float32

        # Check normalization bounds [0.0, 1.0]
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)
        assert not np.isnan(vec).any()
        assert not np.isinf(vec).any()

        # Check one-hot encoding for phase 1: indices [3, 4, 5] correspond to phases
        np.testing.assert_array_almost_equal(vec[3:6], [0.0, 1.0, 0.0])

        # Check normalized time: 30 / 60 = 0.5
        assert vec[6] == pytest.approx(0.5)

        # Transition flag: False -> 0.0
        assert vec[7] == 0.0

        # Min green satisfied: 30.0 >= 5.0 -> 1.0
        assert vec[8] == 1.0

    def test_builder_rejects_empty_lanes(self):
        with pytest.raises(ValueError):
            ObservationBuilder(incoming_lanes=[], num_green_phases=2)
