"""Unit tests for experiment runner, benchmark execution, and evaluate_rl."""

from dataclasses import replace
import json
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.controllers import make_controller
from src.controllers.q_learning import QLearningController
import src.evaluation.evaluate_rl as eval_rl
from src.evaluation.evaluate_rl import create_evaluation_session, run_greedy_eval_loop
import src.evaluation.experiment_runner as evaluation
from src.evaluation.experiment_runner import (
    ExperimentConfig,
    aggregate_benchmark_statistics,
    compute_improvement,
    compute_run_fingerprint,
    get_sumo_version,
    load_experiments_yaml,
    run_full_benchmark_matrix,
    run_single_experiment,
)
from src.metrics import REPORT_COMPLETION_THRESHOLD, completion_statistics, service_status


class TestExperimentRunnerUtilities:
    """Test suite for helper utilities in experiment runner."""

    def test_get_sumo_version(self):
        ver = get_sumo_version()
        assert ver != "unknown"
        assert ver != "2", f"SUMO version should not be '2', got '{ver}'"
        assert "." in ver, f"SUMO version should contain a dot, got '{ver}'"

    def test_missing_sumo_version_is_unknown(self, monkeypatch):
        def unavailable(*args, **kwargs):
            raise FileNotFoundError("sumo")
        monkeypatch.setattr(evaluation.subprocess, "check_output", unavailable)
        assert evaluation.get_sumo_version() == "unknown"

    def test_compute_improvement(self):
        # Lower is better (waiting time, queue, collisions)
        assert compute_improvement(20.0, 10.0, higher_is_better=False) == 50.0
        assert compute_improvement(10.0, 20.0, higher_is_better=False) == -100.0
        assert compute_improvement(0.0, 0.0, higher_is_better=False) == 0.0
        # Baseline 0 collisions, algorithm 4 collisions -> must be negative (degradation)
        assert compute_improvement(0.0, 4.0, higher_is_better=False) is None
        assert compute_improvement(0, 4) is None
        assert compute_improvement(float("nan"), 4) is None

        # Higher is better (throughput)
        assert compute_improvement(100.0, 150.0, higher_is_better=True) == 50.0
        assert compute_improvement(150.0, 100.0, higher_is_better=True) == pytest.approx(-33.333, abs=0.01)
        assert compute_improvement(0.0, 10.0, higher_is_better=True) is None

    def test_zero_baseline_does_not_invent_a_relative_percentage(self):
        assert compute_improvement(0, 4) is None
        assert compute_improvement(float("nan"), 4) is None

    def test_fingerprint_changes_with_steps_and_model_contents(self, tmp_path):
        path = tmp_path / "model.json"
        path.write_text("first", encoding="utf-8")
        config = ExperimentConfig("cologne1", "Q-Learning", steps=20, model_path=str(path))
        first = compute_run_fingerprint(config, str(path))
        assert compute_run_fingerprint(replace(config, steps=99), str(path)) != first
        path.write_text("second", encoding="utf-8")
        assert compute_run_fingerprint(config, str(path)) != first

    def test_experiments_loader_does_not_hide_bad_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("simulation: [unclosed", encoding="utf-8")
        with pytest.raises(Exception):
            load_experiments_yaml(str(path))


class TestBenchmarkExecution:
    """Test suite for benchmark execution and aggregation."""

    def test_invalid_summary_is_not_promoted_to_success(self, tmp_path, monkeypatch):
        runner = MagicMock()
        runner.run.return_value = {"is_valid": False, "measurement_errors": 3, "total_steps": 10}
        monkeypatch.setattr(evaluation, "SimulationRunner", lambda *a, **k: runner)
        result = run_single_experiment(ExperimentConfig("cologne1", "Fixed-Time", output_dir=str(tmp_path)))
        assert result["is_valid"] is False
        assert result["status"] == "FAILED"

    def test_aggregation_preserves_failed_seeds_and_deduplicates_retries(self):
        good = {
            "scenario_name": "test", "algorithm": "Fixed-Time", "seed": 1,
            "status": "SUCCESS", "is_valid": True, "mean_queue_length": 10.0,
        }
        failed = {**good, "seed": 2, "status": "FAILED", "is_valid": False, "error_message": "lost connection"}
        result = aggregate_benchmark_statistics([good, good, failed])["test"]["Fixed-Time"]
        assert result["status"] == "PARTIAL"
        assert result["run_count"] == 1
        assert result["failed_count"] == 1
        assert result["error_messages"] == ["lost connection"]

    def test_matrix_honors_yaml_timing_and_does_not_reuse_old_steps(self, tmp_path, monkeypatch):
        path = tmp_path / "experiment.yaml"
        path.write_text(
            "simulation:\n  steps: 10\n  step_length: 0.5\n  use_tripinfo: false\n  max_green: null\n",
            encoding="utf-8",
        )
        seen = []

        def run(config):
            seen.append(config)
            return {
                "scenario_name": "cologne1", "algorithm": "Fixed-Time", "seed": 42,
                "status": "SUCCESS", "is_valid": True, "total_steps": config.steps,
            }

        monkeypatch.setattr(evaluation, "run_single_experiment", run)
        args = dict(
            scenarios=["cologne1"], algorithms=["Fixed-Time"], seeds=[42],
            config_file=str(path), output_dir=str(tmp_path),
        )
        run_full_benchmark_matrix(**args)
        assert seen[0].step_length == 0.5
        assert seen[0].use_tripinfo is False
        assert seen[0].max_green is None

        run_full_benchmark_matrix(**args)
        assert len(seen) == 1

        path.write_text(path.read_text(encoding="utf-8").replace("steps: 10", "steps: 99"), encoding="utf-8")
        run_full_benchmark_matrix(**args)
        assert [config.steps for config in seen] == [10, 99]

    def test_zero_demand_run_is_not_a_failed_completion(self, tmp_path):
        result = run_single_experiment(ExperimentConfig("cologne1", "Fixed-Time", steps=0, output_dir=str(tmp_path)))
        assert result["tripinfo_mean_waiting_time"] is None
        assert result["tripinfo_mean_travel_time"] is None

    def test_nonfinite_rewards_do_not_break_strict_json_export(self):
        result = aggregate_benchmark_statistics([{
            "scenario_name": "audit", "algorithm": "Q-Learning", "seed": 1,
            "status": "SUCCESS", "is_valid": True, "episode_reward": float("nan"),
        }])
        assert result["audit"]["Q-Learning"]["episode_reward"]["mean"] is None
        json.dumps(result, allow_nan=False)

    def test_completion_statistics_uses_realized_demand_and_distinct_policies(self):
        completed, demand, rate = completion_statistics({
            "total_completed_vehicles": 96, "running_vehicles_at_end": 3, "pending_insertion_vehicles": 1,
        })
        assert (completed, demand, rate) == (96, 100, 0.96)
        assert service_status(rate) == "COMPLETED"
        assert service_status(rate, REPORT_COMPLETION_THRESHOLD) == "LOW_COMPLETION"


class TestEvaluateRL:
    """Test suite for RL evaluation setup and greedy loop."""

    def test_evaluation_requires_an_existing_checkpoint(self, tmp_path):
        path = str(tmp_path / "missing.json")
        for create in (
            lambda: make_controller("q-learning", model_path=path),
            lambda: create_evaluation_session("cologne1", path),
        ):
            with pytest.raises(FileNotFoundError):
                create()

    @pytest.mark.parametrize("fails", [False, True])
    def test_greedy_evaluation_does_not_change_learning_state(self, fails):
        agent = QLearningController(num_actions=2, epsilon=0.4)
        state = (0, 0, 0, 0)
        agent.q_table[state] = np.array([1.0, 2.0])
        env = MagicMock()
        info = {"discrete_state": state, "valid_actions": [0, 1]}
        next_info = {"discrete_state": (1, 0, 0, 0), "valid_actions": [0, 1]}
        env.reset.return_value = (None, info)
        env.step.side_effect = [
            (None, 1.0, False, False, next_info),
            RuntimeError("simulation stopped") if fails else (None, 1.0, True, False, next_info),
        ]
        env.get_summary.return_value = {}
        if fails:
            with pytest.raises(RuntimeError, match="simulation stopped"):
                run_greedy_eval_loop(agent, env)
        else:
            run_greedy_eval_loop(agent, env)
        assert list(agent.q_table) == [state]
        np.testing.assert_array_equal(agent.q_table[state], [1.0, 2.0])
        np.testing.assert_array_equal(agent.action_counts, [0, 0])
        assert agent.total_updates == 0
        assert agent.epsilon == 0.4
        assert agent.eval_mode is False

    def test_evaluation_setup_shares_checkpoint_target_and_disabled_timing(self, tmp_path, monkeypatch):
        path = str(tmp_path / "agent.json")
        original = QLearningController(num_actions=3, target_tls_id="junction-2", max_green=None)
        original.discretizer.serving_thresholds = (2.0, 8.0)
        original.save(path)
        options = {}
        env = MagicMock()
        env.num_actions = 3
        env.tls_id = "junction-2"

        def make_env(**kwargs):
            options.update(kwargs)
            env.state_discretizer = kwargs["state_discretizer"]
            return env

        monkeypatch.setattr(eval_rl, "TrafficSignalRLEnv", make_env)
        agent, actual_env = create_evaluation_session(
            "cologne3", path, seed=17, timing_overrides={"decision_interval": 10.0},
        )
        assert actual_env is env
        assert options["target_tls_id"] == "junction-2"
        assert options["max_green"] is None
        assert options["decision_interval"] == agent.decision_interval == 10.0
        assert agent.discretizer is env.state_discretizer
        assert agent.discretizer.serving_thresholds == (2.0, 8.0)

    def test_evaluation_setup_closes_env_on_action_count_mismatch(self, tmp_path, monkeypatch):
        path = str(tmp_path / "agent.json")
        QLearningController(num_actions=3).save(path)
        env = MagicMock()
        env.num_actions = 2
        monkeypatch.setattr(eval_rl, "TrafficSignalRLEnv", lambda **_: env)
        with pytest.raises(ValueError, match="actions"):
            create_evaluation_session("cologne1", path)
        env.close.assert_called_once()


