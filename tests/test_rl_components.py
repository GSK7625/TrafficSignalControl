"""Unit tests for RL components: StateDiscretizer, Reward functions, and Environment."""

import inspect
import subprocess
import sys
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.config import DEFAULT_SAFETY_CONFIG
from src.controllers.base import ControlledLink
from src.controllers.q_learning import QLearningController
from src.evaluation.evaluate_rl import create_evaluation_session, run_greedy_eval_loop
from src.phase_manager import SafePhaseManager
from src.rl.discretization import StateDiscretizer
from src.rl.environment import TrafficSignalRLEnv
from src.rl.rewards import MetricsAlignedReward, NegativePressureReward
import src.training.train_q_learning as tql


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

    def test_discretize_recognizes_signal_s_as_green_serving(self, sample_observation):
        disc = StateDiscretizer(serving_queue_thresholds=(1.0, 5.0), waiting_queue_thresholds=(1.0, 5.0))
        sample_observation.green_phases = ["srrr"]
        sample_observation.current_phase = 0
        sample_observation.controlled_links = [ControlledLink(signal_index=0, from_lane="in_0", to_lane="out_0")]
        sample_observation.in_lane_queues = {"in_0": 3.0}

        green_lanes = disc._get_green_lanes(sample_observation)
        assert "in_0" in green_lanes

        state = disc.discretize(sample_observation)
        assert state[1] == 1  # serving bin
        assert state[2] == 0  # waiting bin

    def test_discretizer_aggregate_mode(self):
        """Tests that StateDiscretizer in aggregate mode discretizes queues correctly."""
        links = [
            ControlledLink(signal_index=0, from_lane="lane_N_0", to_lane="out_S"),
            ControlledLink(signal_index=1, from_lane="lane_E_0", to_lane="out_W"),
        ]
        from src.controllers.base import IntersectionObservation
        obs = IntersectionObservation(
            tls_id="tls_test",
            current_phase=0,
            time_in_phase=10.0,
            is_in_transition=False,
            green_phases=["Gr", "rG"],
            controlled_links=links,
            in_lane_queues={"lane_N_0": 3.0, "lane_E_0": 8.0},
            out_lane_queues={}, in_lane_vehicles={}, out_lane_vehicles={}, sim_time=100.0, step=100,
        )
        disc = StateDiscretizer(group_by="aggregate")
        state = disc.discretize(obs)
        assert state == (0, 1, 2, 1)

    def test_discretizer_config_serialization_and_validation(self):
        disc = StateDiscretizer(min_green=6.0, group_by="aggregate")
        cfg = disc.get_config()
        assert cfg["group_by"] == "aggregate"
        assert cfg["min_green"] == 6.0

        restored = StateDiscretizer.from_config(cfg)
        assert restored.group_by == "aggregate"
        assert restored.min_green == 6.0

        with pytest.raises(ValueError, match="group_by only supports 'aggregate'"):
            StateDiscretizer(group_by="invalid_mode")

    def test_phase_state_distinguishes_competing_directions(self, make_observation):
        links = [ControlledLink(i, f"in_{i}", f"out_{i}") for i in range(3)]
        obs = make_observation(green_phases=["Grr", "rGr", "rrG"], controlled_links=links,
                               in_lane_queues={"in_0": 2, "in_1": 8, "in_2": 0})
        disc = StateDiscretizer(group_by="phase")
        first = disc.discretize(obs)
        aggregate = StateDiscretizer().discretize(obs)
        obs.in_lane_queues = {"in_0": 2, "in_1": 0, "in_2": 8}
        second = disc.discretize(obs)
        assert StateDiscretizer().discretize(obs) == aggregate
        assert first != second
        assert first[2] == 1 and second[2] == 2

    def test_phase_state_accounts_for_downstream_blockage(self, make_observation):
        links = [ControlledLink(i, f"in_{i}", f"out_{i}") for i in range(3)]
        obs = make_observation(green_phases=["Grr", "rGr", "rrG"], controlled_links=links,
                               in_lane_queues={"in_0": 0, "in_1": 8, "in_2": 4},
                               out_lane_queues={"out_1": 10})
        disc = StateDiscretizer(group_by="phase")
        assert disc.discretize(obs)[2] == 2
        restored = StateDiscretizer.from_config(disc.get_config())
        assert restored.discretize(obs) == disc.discretize(obs)

    def test_phase_state_checkpoint_roundtrip(self, sample_observation, tmp_path):
        disc = StateDiscretizer(group_by="phase")
        agent = QLearningController(2, discretizer=disc)
        state = disc.discretize(sample_observation)
        agent.update(state, 1, 3, state, done=True)
        path = str(tmp_path / "phase.json")
        agent.save(path)
        restored = QLearningController.from_checkpoint(path)
        assert restored.discretizer.discretize(sample_observation) == state
        assert restored.select_action_for_state(state) == 1


class TestRewards:
    """Test suite for RL Reward functions."""

    def test_negative_pressure_reward_unmapped(self):
        reward_fn = NegativePressureReward(scale=1.0)
        reward_fn.record_step({"l1": 2.0, "l2": 2.0})
        reward_fn.record_step({"l1": 4.0, "l2": 4.0})

        r = reward_fn.calculate_reward()
        assert pytest.approx(r, 1e-6) == -6.0

        reward_fn.reset()
        assert reward_fn.calculate_reward() == 0.0

    def test_negative_pressure_reward_with_pending(self):
        reward_fn = NegativePressureReward(scale=1.0, pending_weight=1.5)
        reward_fn.record_step({"l1": 2.0}, pending_vehicles=4)
        reward_fn.record_step({"l1": 4.0}, pending_vehicles=0)
        r = reward_fn.calculate_reward()
        assert pytest.approx(r, 1e-6) == -6.0

    def test_negative_pressure_reward_with_controlled_links(self):
        reward_fn = NegativePressureReward(scale=1.0)
        links = [
            ControlledLink(signal_index=0, from_lane="in_1", to_lane="out_1"),
            ControlledLink(signal_index=1, from_lane="in_2", to_lane="out_2"),
        ]
        in_queues = {"in_1": 5.0, "in_2": 3.0}
        out_queues = {"out_1": 2.0, "out_2": 1.0}

        reward_fn.record_step(
            in_lane_queues=in_queues,
            out_lane_queues=out_queues,
            controlled_links=links,
        )
        assert pytest.approx(reward_fn.calculate_reward(), 1e-6) == -5.0

        reward_fn.reset()
        blocked_out_queues = {"out_1": 10.0, "out_2": 1.0}
        reward_fn.record_step(
            in_lane_queues=in_queues,
            out_lane_queues=blocked_out_queues,
            controlled_links=links,
        )
        assert pytest.approx(reward_fn.calculate_reward(), 1e-6) == -7.0

    def test_negative_pressure_reward_static_topology(self):
        links = [
            ControlledLink(signal_index=0, from_lane="in_1", to_lane="out_1"),
            ControlledLink(signal_index=1, from_lane="in_2", to_lane="out_2"),
        ]
        reward_fn = NegativePressureReward(scale=2.0)
        reward_fn.set_topology_from_links(links)
        assert reward_fn.topology == {"in_1": {"out_1"}, "in_2": {"out_2"}}

        reward_fn.record_step(
            in_lane_queues={"in_1": 4.0, "in_2": 2.0},
            out_lane_queues={"out_1": 1.0, "out_2": 1.0},
        )
        # in_1: |4 - 1| = 3, in_2: |2 - 1| = 1 -> sum = 4.0. With scale 2.0 -> -8.0
        assert pytest.approx(reward_fn.calculate_reward(), 1e-6) == -8.0

    def test_negative_pressure_reward_validation(self):
        with pytest.raises(ValueError, match="scale must be non-negative"):
            NegativePressureReward(scale=-1.0)
        with pytest.raises(ValueError, match="pending_weight must be non-negative"):
            NegativePressureReward(pending_weight=-0.5)

    def test_negative_pressure_reward_accepts_extra_kwargs(self):
        reward_fn = NegativePressureReward(scale=1.0)
        # Should not raise TypeError when passed extra metrics
        reward_fn.record_step(
            in_lane_queues={"lane_0": 2.0},
            out_lane_queues={},
            pending_vehicles=1,
            waiting_time=10.0,
            completed_vehicles=2,
            is_switch=True,
        )
        assert reward_fn.calculate_reward() == -2.0


class TestMetricsAlignedReward:
    """Test suite for MetricsAlignedReward."""

    def test_initialization_defaults_and_get_config(self):
        reward_fn = MetricsAlignedReward()
        cfg = reward_fn.get_config()
        assert cfg["name"] == "MetricsAlignedReward"
        assert cfg["scale"] == 1.0
        assert cfg["w_completed"] == 1.5
        assert cfg["w_queue"] == 0.05
        assert cfg["w_waiting"] == 0.01
        assert cfg["w_switch"] == 0.5
        assert cfg["w_pending"] == 0.02

    @pytest.mark.parametrize("param,invalid_val", [
        ("scale", -1.0),
        ("scale", float("nan")),
        ("w_completed", -0.1),
        ("w_queue", float("inf")),
        ("w_waiting", -1.0),
        ("w_switch", -0.5),
        ("w_pending", -2.0),
    ])
    def test_parameter_validation(self, param, invalid_val):
        with pytest.raises(ValueError, match=f"{param} must be non-negative and finite"):
            MetricsAlignedReward(**{param: invalid_val})

    def test_throughput_bonus(self):
        reward_fn = MetricsAlignedReward(
            scale=1.0, w_completed=2.0, w_queue=0.0, w_waiting=0.0, w_switch=0.0, w_pending=0.0
        )
        # 3 vehicles completed across 2 simulation steps
        reward_fn.record_step(completed_vehicles=1)
        reward_fn.record_step(completed_vehicles=2)
        assert reward_fn.calculate_reward() == 6.0  # 3 * 2.0

    def test_quadratic_queue_penalty_penalizes_concentrated_queues(self):
        # Balanced: 4 lanes with 2 veh each -> sum(q^2) = 4 * 4 = 16
        reward_balanced = MetricsAlignedReward(w_completed=0.0, w_queue=1.0, w_waiting=0.0, w_switch=0.0, w_pending=0.0)
        reward_balanced.record_step(in_lane_queues={"l1": 2.0, "l2": 2.0, "l3": 2.0, "l4": 2.0})

        # Concentrated: 1 lane with 8 veh, 3 empty -> sum(q^2) = 64
        reward_concentrated = MetricsAlignedReward(w_completed=0.0, w_queue=1.0, w_waiting=0.0, w_switch=0.0, w_pending=0.0)
        reward_concentrated.record_step(in_lane_queues={"l1": 8.0, "l2": 0.0, "l3": 0.0, "l4": 0.0})

        # Concentrated queue must receive 4x heavier penalty
        assert reward_balanced.calculate_reward() == -16.0
        assert reward_concentrated.calculate_reward() == -64.0

    def test_waiting_time_and_pending_penalties(self):
        reward_fn = MetricsAlignedReward(
            w_completed=0.0, w_queue=0.0, w_waiting=0.1, w_switch=0.0, w_pending=0.5
        )
        reward_fn.record_step(waiting_time=50.0, pending_vehicles=4)
        reward_fn.record_step(waiting_time=30.0, pending_vehicles=2)
        # avg waiting = 40.0 -> penalty 4.0; avg pending = 3.0 -> penalty 1.5
        assert pytest.approx(reward_fn.calculate_reward(), 1e-6) == -5.5

    def test_switch_penalty_applied_once_per_interval(self):
        reward_fn = MetricsAlignedReward(
            w_completed=0.0, w_queue=0.0, w_waiting=0.0, w_switch=3.0, w_pending=0.0
        )
        # First step triggers a phase switch
        reward_fn.record_step(is_switch=True)
        # Subsequent steps in same interval do not switch
        reward_fn.record_step(is_switch=False)
        reward_fn.record_step(is_switch=False)
        assert reward_fn.calculate_reward() == -3.0

    def test_reset_interval_clears_metrics(self):
        reward_fn = MetricsAlignedReward()
        reward_fn.record_step(completed_vehicles=5, is_switch=True)
        assert reward_fn.calculate_reward() != 0.0
        reward_fn.reset_interval()
        assert reward_fn.calculate_reward() == 0.0


class TestTrafficSignalRLEnv:
    """Test suite for TrafficSignalRLEnv lifecycle and configuration."""

    def test_switch_penalty_counts_actual_switch_once_during_clearance(self):
        env = TrafficSignalRLEnv("ingolstadt1", decision_interval=1, max_steps=15)
        try:
            env.reset()
            record = MagicMock(wraps=env.reward_fn.record_step)
            env.reward_fn.record_step = record
            for _ in range(5):
                env.step(0)
            # Request the same target throughout yellow and all-red.
            for _ in range(4):
                env.step(2)
            assert sum(call.kwargs["is_switch"] for call in record.call_args_list) == 1
        finally:
            env.close()

    def test_env_save_run_details_default(self):
        env = TrafficSignalRLEnv(scenario_name="cologne1")
        assert env.config.save_run_details is False

    def test_env_reset_clears_summary_and_zero_limit_never_steps(self):
        env = TrafficSignalRLEnv("cologne1", max_steps=0)
        try:
            env._last_summary = {"previous": True}
            env.reset()
            assert env.get_summary() is None
            _, _, _, truncated, _ = env.step(0)
            assert truncated is True
            assert env.runner.current_step == 0
            with pytest.raises(RuntimeError):
                env.step(0)
        finally:
            env.close()

    def test_env_rejects_out_of_range_actions(self):
        env = TrafficSignalRLEnv("cologne1", max_steps=5)
        try:
            env.reset()
            with pytest.raises(ValueError):
                env.step(env.num_actions)
        finally:
            env.close()

    def test_environment_rejects_misspelled_options(self):
        with pytest.raises(TypeError):
            TrafficSignalRLEnv("cologne1", decison_interval=99)

    def test_checkpoint_restores_reward_configuration(self, tmp_path):
        probe = TrafficSignalRLEnv("cologne1")
        agent = QLearningController(probe.num_actions)
        agent.reward_config = {"name": "NegativePressureReward", "scale": 2.0, "pending_weight": 0.5}
        path = str(tmp_path / "agent.json")
        agent.save(path)
        _, env = create_evaluation_session("cologne1", path)
        assert env.reward_fn.get_config() == {"name": "NegativePressureReward", "scale": 2.0, "pending_weight": 0.5}
        env.reward_fn.record_step({"in": 3}, pending_vehicles=4)
        assert env.reward_fn.calculate_reward() == -10.0

    def test_checkpoint_restores_metrics_aligned_reward(self, tmp_path):
        probe = TrafficSignalRLEnv("cologne1")
        agent = QLearningController(probe.num_actions)
        agent.reward_config = {
            "name": "MetricsAlignedReward",
            "scale": 1.0,
            "w_completed": 2.0,
            "w_queue": 0.1,
            "w_waiting": 0.05,
            "w_switch": 1.0,
            "w_pending": 0.03,
        }
        path = str(tmp_path / "agent_metrics.json")
        agent.save(path)
        _, env = create_evaluation_session("cologne1", path)
        assert env.reward_fn.get_config()["name"] == "MetricsAlignedReward"
        assert env.reward_fn.w_completed == 2.0
        env.reward_fn.record_step(completed_vehicles=3)
        assert env.reward_fn.calculate_reward() == 6.0

    def test_checkpoint_rejects_unknown_reward_name(self, tmp_path):
        probe = TrafficSignalRLEnv("cologne1")
        agent = QLearningController(probe.num_actions)
        agent.reward_config = {"name": "AlienReward"}
        path = str(tmp_path / "agent_alien.json")
        agent.save(path)
        with pytest.raises(ValueError, match="Unsupported checkpoint reward function: AlienReward"):
            create_evaluation_session("cologne1", path)

    def test_rl_environment_uses_shared_all_red_default(self):
        assert (
            inspect.signature(TrafficSignalRLEnv).parameters["all_red_duration"].default
            == DEFAULT_SAFETY_CONFIG["all_red_duration"]
        )

    def test_rl_environment_import_is_independent_of_import_order(self):
        result = subprocess.run(
            [sys.executable, "-c", "from src.rl.environment import TrafficSignalRLEnv"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    def test_env_get_valid_actions_enforces_max_green(self):
        env = MagicMock(spec=TrafficSignalRLEnv)
        env.num_actions = 3
        env.min_green = 5.0
        env.max_green = 60.0
        env.tls_id = "tls_0"
        env.runner = MagicMock()
        env.runner.is_running = True
        mgr = SafePhaseManager("tls_0", ["Grr", "rGr", "rrG"], min_green=5, max_green=60)
        env.runner.tls_managers = {"tls_0": mgr}

        mgr.update(3.0)
        assert TrafficSignalRLEnv.get_valid_actions(env) == [0]

        mgr.update(22.0)
        assert TrafficSignalRLEnv.get_valid_actions(env) == [0, 1, 2]

        mgr.update(35.0)
        assert TrafficSignalRLEnv.get_valid_actions(env) == [1, 2]


class TestQLearningEvaluation:
    """Test suite for Q-Learning evaluation and checkpoint filtering."""

    def test_evaluate_greedy_returns_full_metrics(self, monkeypatch):
        mock_env = MagicMock()
        mock_env.num_actions = 2
        mock_env.tls_id = "junction-1"
        mock_env.reset.return_value = ((0, 0, 0, 0), {"discrete_state": (0, 0, 0, 0), "valid_actions": [0, 1]})
        mock_env.step.side_effect = [
            ((0, 0, 0, 1), -5.0, False, False, {"discrete_state": (0, 0, 0, 1), "valid_actions": [0, 1]}),
            ((0, 0, 0, 1), -3.0, True, False, {"discrete_state": (0, 0, 0, 1), "valid_actions": [0, 1]}),
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
        mock_agent.num_actions = 2
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
        assert mock_agent.eval_mode is False
        assert mock_agent.epsilon == 0.5

    def test_viability_checkpoint_logic(self):
        selector = tql.ValidationCheckpointSelector(min_completion=0.95)

        safe_viable = {
            "reward": -100,
            "completion_rate": 0.98,
            "summary": {
                "is_valid": True,
                "scheduled_demand_in_window": 100,
                "total_completed_vehicles": 98,
                "tripinfo_required": True,
                "tripinfo_reconciliation_status": "MATCH",
            },
        }
        assert selector.is_viable([safe_viable]) is True
        assert selector.consider(1, [safe_viable]) is True

        starving_policy = {
            "reward": -50,
            "completion_rate": 0.75,
            "summary": {
                "is_valid": True,
                "scheduled_demand_in_window": 100,
                "total_completed_vehicles": 75,
                "tripinfo_required": True,
                "tripinfo_reconciliation_status": "MATCH",
            },
        }
        assert selector.is_viable([starving_policy]) is False
        assert selector.consider(2, [starving_policy]) is False

        collision_policy = {
            "reward": -30,
            "completion_rate": 0.99,
            "summary": {
                "is_valid": True,
                "scheduled_demand_in_window": 100,
                "total_completed_vehicles": 99,
                "collision_count": 2,
                "tripinfo_required": True,
                "tripinfo_reconciliation_status": "MATCH",
            },
        }
        assert selector.is_viable([collision_policy]) is False
        assert selector.consider(3, [collision_policy]) is False

    def test_run_greedy_eval_loop(self):
        agent = MagicMock()
        agent.eval_mode = False
        agent.epsilon = 0.5
        agent.select_action_for_state.return_value = 1

        env = MagicMock()
        env.reset.return_value = ((0, 1, 2, 0), {"discrete_state": (0, 1, 2, 0), "valid_actions": [0, 1]})
        env.step.return_value = (
            (0, 1, 2, 0),
            12.5,
            True,
            False,
            {"discrete_state": (0, 1, 2, 0), "valid_actions": [0, 1]},
        )
        env.get_summary.return_value = {"total_completed_vehicles": 10, "mean_waiting_time": 4.2}

        total_reward, summary = run_greedy_eval_loop(agent, env, seed=123)

        assert total_reward == 12.5
        assert summary["total_completed_vehicles"] == 10
        assert agent.eval_mode is False
        assert agent.epsilon == 0.5
        env.reset.assert_called_once_with(seed=123)
        agent.select_action_for_state.assert_called_once_with(
            (0, 1, 2, 0), valid_actions=[0, 1], current_phase=0
        )

    def test_no_observation_builder_dependency(self):
        import src.rl
        assert not hasattr(src.rl, "ObservationBuilder")

    @pytest.mark.parametrize("factory", [
        lambda: StateDiscretizer(serving_queue_thresholds=(5, 1)),
        lambda: NegativePressureReward(scale=float("nan")),
        lambda: SafePhaseManager("tls", ["Gr", "rG"], yellow_duration=float("nan")),
    ])
    def test_invalid_state_reward_and_clearance_parameters_fail_early(self, factory):
        with pytest.raises(ValueError):
            factory()



