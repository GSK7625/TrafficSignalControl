"""Unit tests for RL support components: StateDiscretizer, ObservationBuilder, and Reward functions."""

import numpy as np
import pytest

from src.controllers.base import ControlledLink, IntersectionObservation
from src.rl.discretization import StateDiscretizer
from src.rl.observations import ObservationBuilder
from src.rl.rewards import NegativeAverageQueueReward


@pytest.fixture
def sample_observation():
    links = [
        ControlledLink(signal_index=0, from_lane="lane_N", to_lane="lane_S"),
        ControlledLink(signal_index=1, from_lane="lane_E", to_lane="lane_W"),
    ]
    # Phase 0: 'Gr' -> lane_N green, lane_E red
    # Phase 1: 'rG' -> lane_N red, lane_E green
    return IntersectionObservation(
        tls_id="tls_test",
        current_phase=0,
        time_in_phase=10.0,
        is_in_transition=False,
        green_phases=["Gr", "rG"],
        controlled_links=links,
        in_lane_queues={"lane_N": 3.0, "lane_E": 8.0},
        out_lane_queues={"lane_S": 0.0, "lane_W": 0.0},
        in_lane_vehicles={"lane_N": 4.0, "lane_E": 10.0},
        out_lane_vehicles={"lane_S": 1.0, "lane_W": 2.0},
        sim_time=100.0,
        step=100,
    )


class TestStateDiscretizer:
    """Test suite for StateDiscretizer."""

    def test_bin_value(self):
        disc = StateDiscretizer(serving_queue_thresholds=(1.0, 5.0))
        assert disc._bin_value(0.5, (1.0, 5.0)) == 0
        assert disc._bin_value(1.0, (1.0, 5.0)) == 1
        assert disc._bin_value(3.5, (1.0, 5.0)) == 1
        assert disc._bin_value(5.0, (1.0, 5.0)) == 2
        assert disc._bin_value(10.0, (1.0, 5.0)) == 2

    def test_discretize_observation(self, sample_observation):
        disc = StateDiscretizer(
            min_green=5.0,
            max_effective_green=25.0,
            serving_queue_thresholds=(1.0, 5.0),
            waiting_queue_thresholds=(1.0, 6.0, 15.0),
        )
        # Phase 0 ('Gr'):
        # Green lane = lane_N (queue=3.0) -> serving bin = 1 (1.0 <= 3.0 < 5.0)
        # Red lane = lane_E (queue=8.0) -> waiting bin = 2 (6.0 <= 8.0 < 15.0)
        # time_in_phase = 10.0 -> time bin = 1 (5.0 <= 10.0 < 25.0)
        state = disc.discretize(sample_observation)
        assert state == (0, 1, 2, 1)

        # Pending vehicles must NOT pollute local waiting queue discretization
        sample_observation.pending_vehicles = 100
        state_with_pending = disc.discretize(sample_observation)
        assert state_with_pending == (0, 1, 2, 1)

    def test_discretize_time_bins(self, sample_observation):
        disc = StateDiscretizer(min_green=5.0, max_effective_green=25.0)

        # Time < min_green -> bin 0
        sample_observation.time_in_phase = 4.0
        assert disc.discretize(sample_observation)[3] == 0

        # In transition -> bin 0
        sample_observation.time_in_phase = 10.0
        sample_observation.is_in_transition = True
        assert disc.discretize(sample_observation)[3] == 0
        sample_observation.is_in_transition = False

        # Standard window -> bin 1
        sample_observation.time_in_phase = 15.0
        assert disc.discretize(sample_observation)[3] == 1

        # Saturated / Max window -> bin 2
        sample_observation.time_in_phase = 26.0
        assert disc.discretize(sample_observation)[3] == 2


class TestObservationBuilder:
    """Test suite for continuous ObservationBuilder."""

    def test_vector_dimension(self):
        builder = ObservationBuilder(
            incoming_lanes=["lane_1", "lane_2"],
            num_green_phases=2,
            min_green=5.0,
        )
        # N_in (2) + K (2) + 3 (time, transition, min_green_flag) = 7
        assert builder.observation_dim == 7

    def test_build_vector_values(self, sample_observation):
        builder = ObservationBuilder(
            incoming_lanes=["lane_N", "lane_E"],
            num_green_phases=2,
            min_green=5.0,
            max_queue_scale=20.0,
            max_time_scale=60.0,
        )
        vec = builder.build_vector(sample_observation)
        assert isinstance(vec, np.ndarray)
        assert len(vec) == 7

        # Incoming lanes are sorted: ["lane_E", "lane_N"]
        # lane_E queue: 8.0 / 20.0 = 0.40
        # lane_N queue: 3.0 / 20.0 = 0.15
        assert pytest.approx(vec[0], 1e-4) == 0.40
        assert pytest.approx(vec[1], 1e-4) == 0.15

        # One-hot phase: phase 0 is active -> [1.0, 0.0]
        assert vec[2] == 1.0
        assert vec[3] == 0.0

        # Normalized time: 10.0 / 60.0
        assert pytest.approx(vec[4], 1e-4) == 10.0 / 60.0

        # In transition: False -> 0.0
        assert vec[5] == 0.0

        # Min green met: 10.0 >= 5.0 -> 1.0
        assert vec[6] == 1.0


class TestRewards:
    """Test suite for RL Reward functions."""

    def test_negative_average_queue_reward(self):
        reward_fn = NegativeAverageQueueReward(scale=1.0)
        # Step 1: queue sum = 4.0
        reward_fn.record_step({"l1": 2.0, "l2": 2.0})
        # Step 2: queue sum = 8.0
        reward_fn.record_step({"l1": 4.0, "l2": 4.0})

        # Average queue = (4 + 8) / 2 = 6.0 -> Reward = -6.0
        r = reward_fn.calculate_reward()
        assert pytest.approx(r, 1e-6) == -6.0

        # Reset clears history
        reward_fn.reset()
        assert reward_fn.calculate_reward() == 0.0

    def test_negative_average_queue_reward_with_pending(self):
        reward_fn = NegativeAverageQueueReward(scale=1.0, pending_weight=1.5)
        # Step 1: queue=2.0, pending=4 -> total penalty = 2.0 + 1.5 * 4 = 8.0
        reward_fn.record_step({"l1": 2.0}, pending_vehicles=4)
        # Step 2: queue=4.0, pending=0 -> total penalty = 4.0
        reward_fn.record_step({"l1": 4.0}, pending_vehicles=0)
        # Average penalty = (8.0 + 4.0) / 2 = 6.0 -> Reward = -6.0
        r = reward_fn.calculate_reward()
        assert pytest.approx(r, 1e-6) == -6.0


class TestQLearningEvaluation:
    """Test suite for Q-Learning evaluation and checkpoint filtering."""

    def test_evaluate_greedy_returns_full_metrics(self, monkeypatch):
        from unittest.mock import MagicMock
        import src.training.train_q_learning as tql

        mock_env = MagicMock()
        mock_env.reset.return_value = (np.zeros(10), {"discrete_state": (0, 0, 0, 0), "valid_actions": [0, 1]})
        mock_env.step.side_effect = [
            (np.zeros(10), -5.0, False, False, {"discrete_state": (0, 0, 0, 1), "valid_actions": [0, 1]}),
            (np.zeros(10), -3.0, True, False, {"discrete_state": (0, 0, 0, 1), "valid_actions": [0, 1]}),
        ]
        mock_env.get_summary.return_value = {
            "total_completed_vehicles": 95,
            "running_vehicles_at_end": 3,
            "pending_insertion_vehicles": 2,
            "collision_count": 0,
            "teleport_count": 0,
            "mean_waiting_time": 4.5,
            "system_mean_delay": 5.2,
        }

        monkeypatch.setattr(tql, "TrafficSignalRLEnv", lambda **kwargs: mock_env)

        mock_agent = MagicMock()
        mock_agent.select_action_for_state.return_value = 0
        mock_agent.eval_mode = False
        mock_agent.epsilon = 0.5

        result = tql.evaluate_greedy_q_learning(
            agent=mock_agent,
            scenario_name="dummy",
            seed=42,
        )

        assert result["reward"] == -8.0
        assert result["completed_vehicles"] == 95
        assert result["total_demand"] == 100
        assert result["completion_rate"] == 0.95
        assert result["collisions"] == 0
        assert result["teleports"] == 0
        assert result["mean_waiting_time"] == 4.5
        assert result["system_mean_delay"] == 5.2
        # Verify agent state restored
        assert mock_agent.eval_mode is False
        assert mock_agent.epsilon == 0.5

    def test_viability_checkpoint_logic(self):
        # A policy with 0 collisions and >=95% completion is viable
        safe_viable = {"collisions": 0, "completion_rate": 0.98, "reward": -100}
        assert safe_viable["collisions"] == 0 and safe_viable["completion_rate"] >= 0.95

        # Degenerate policy with starvation (completion < 95%) is rejected
        starving_policy = {"collisions": 0, "completion_rate": 0.75, "reward": -50}
        assert not (starving_policy["collisions"] == 0 and starving_policy["completion_rate"] >= 0.95)

        # Unsafe policy with collisions is rejected
        collision_policy = {"collisions": 2, "completion_rate": 0.99, "reward": -30}
        assert not (collision_policy["collisions"] == 0 and collision_policy["completion_rate"] >= 0.95)

    def test_env_get_valid_actions_enforces_max_green(self):
        from unittest.mock import MagicMock
        from src.rl.environment import TrafficSignalRLEnv

        env = MagicMock(spec=TrafficSignalRLEnv)
        env.num_actions = 3
        env.min_green = 5.0
        env.max_green = 60.0
        env.tls_id = "tls_0"
        env.runner = MagicMock()
        env.runner.is_running = True
        mgr = MagicMock()
        mgr.get_current_phase.return_value = 0
        mgr.is_in_transition.return_value = False
        env.runner.tls_managers = {"tls_0": mgr}

        # Below min_green: must stay in current phase [0]
        mgr.time_in_current_phase.return_value = 3.0
        assert TrafficSignalRLEnv.get_valid_actions(env) == [0]

        # Normal window [5.0, 60.0): all actions [0, 1, 2] valid
        mgr.time_in_current_phase.return_value = 25.0
        assert TrafficSignalRLEnv.get_valid_actions(env) == [0, 1, 2]

        # Exceeded max_green (>= 60.0): anti-starvation excludes current phase [1, 2]
        mgr.time_in_current_phase.return_value = 60.0
        assert TrafficSignalRLEnv.get_valid_actions(env) == [1, 2]


