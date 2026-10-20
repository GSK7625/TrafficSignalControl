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
    load_experiments_yaml,
    run_full_benchmark_matrix,
    run_single_experiment,
)
from src.metrics import REPORT_COMPLETION_THRESHOLD, completion_statistics, service_status


class TestExperimentRunnerUtilities:
    """Test suite for helper utilities in experiment runner."""


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

    def test_aggregate_benchmark_statistics_normalizes_aliases(self):
        r1 = {"scenario_name": "cologne1", "algorithm": "ft", "seed": 42, "status": "SUCCESS", "is_valid": True, "mean_queue_length": 5.0}
        r2 = {"scenario_name": "cologne1", "algorithm": "mp", "seed": 42, "status": "SUCCESS", "is_valid": True, "mean_queue_length": 3.0}
        stats = aggregate_benchmark_statistics([r1, r2])
        assert "Fixed-Time" in stats["cologne1"]
        assert "Max-Pressure" in stats["cologne1"]
        assert "ft" not in stats["cologne1"]
        assert "mp" not in stats["cologne1"]

    def test_experiment_config_normalizes_algorithm(self):
        cfg = ExperimentConfig(scenario_name="cologne1", algorithm="ft")
        assert cfg.algorithm == "Fixed-Time"
        cfg2 = ExperimentConfig(scenario_name="cologne1", algorithm="ql")
        assert cfg2.algorithm == "Q-Learning"

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
        assert seen[0].timing_overrides["max_green"] is None

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

    def test_cli_single_experiment_failure_exits_nonzero(self, monkeypatch, capsys, tmp_path):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            evaluation,
            "run_single_experiment",
            lambda cfg: {"status": "FAILED", "is_valid": False, "error_message": "Checkpoint not found"},
        )
        monkeypatch.setattr(
            evaluation,
            "load_experiments_yaml",
            lambda path: {"simulation": {"steps": 10}},
        )
        ret = evaluation.main(["--algorithm", "Q-Learning", "--scenario", "cologne1"])
        assert ret != 0
        captured = capsys.readouterr()
        assert "Checkpoint not found" in captured.err

    def test_cli_unified_singular_and_plural_flags(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        matrix_calls = []
        monkeypatch.setattr(
            evaluation,
            "run_full_benchmark_matrix",
            lambda **kwargs: (matrix_calls.append(kwargs), ([{"status": "SUCCESS", "is_valid": True}], {}))[1],
        )
        monkeypatch.setattr(
            evaluation,
            "load_experiments_yaml",
            lambda path: {"simulation": {"steps": 10}},
        )
        # Using plural flags with comma-separated values triggers matrix run
        ret = evaluation.main(["--scenarios", "cologne1,ingolstadt1", "--algorithms", "ft,mp", "--seeds", "42,99"])
        assert ret == 0
        assert len(matrix_calls) == 1
        assert matrix_calls[0]["scenarios"] == ["cologne1", "ingolstadt1"]
        assert matrix_calls[0]["algorithms"] == ["Fixed-Time", "Max-Pressure"]
        assert matrix_calls[0]["seeds"] == [42, 99]

        # Singular flag with comma-separated values also triggers matrix run
        ret = evaluation.main(["--scenario", "cologne1,cologne3", "--algorithm", "Fixed-Time", "--seed", "42"])
        assert ret == 0
        assert len(matrix_calls) == 2
        assert matrix_calls[1]["scenarios"] == ["cologne1", "cologne3"]


    def test_aggregate_benchmark_statistics_rejects_controlled_tls_mismatch(self):
        base_record = {
            "scenario_name": "cologne3",
            "algorithm": "Fixed-Time",
            "seed": 42,
            "status": "SUCCESS",
            "is_valid": True,
            "controlled_tls_ids": ["360082", "360086", "360089"],
            "mean_waiting_time": 20.0,
            "completion_rate_pct": 98.0,
            "total_steps": 3600,
            "effective_config": {"steps": 3600, "step_length": 1.0},
            "observation_window_begin": 0.0,
            "observation_window_end": 3600.0,
            "observation_window_duration": 3600.0,
        }
        mismatched_ql = {
            "scenario_name": "cologne3",
            "algorithm": "Q-Learning",
            "seed": 42,
            "status": "SUCCESS",
            "is_valid": True,
            "controlled_tls_ids": ["360082"],
            "mean_waiting_time": 15.0,
            "completion_rate_pct": 99.0,
        }
        mismatched_ql["total_steps"] = 3600
        mismatched_ql["effective_config"] = {"steps": 3600, "step_length": 1.0}
        mismatched_ql["observation_window_begin"] = 0.0
        mismatched_ql["observation_window_end"] = 3600.0
        mismatched_ql["observation_window_duration"] = 3600.0
        stats = aggregate_benchmark_statistics([base_record, mismatched_ql], baseline_algo="Fixed-Time")
        assert "Controlled TLS mismatch" in stats["cologne3"]["Q-Learning"]["comparison_unavailable_reason"]
        assert "improvements_vs_baseline" not in stats["cologne3"]["Q-Learning"]

        matched_ql = {
            **base_record, **mismatched_ql,
            "controlled_tls_ids": ["360082", "360086", "360089"],
        }
        stats = aggregate_benchmark_statistics([base_record, matched_ql], baseline_algo="Fixed-Time")
        assert "Q-Learning" in stats["cologne3"]
        assert "mean_waiting_time_improvement_pct" in stats["cologne3"]["Q-Learning"]["improvements_vs_baseline"]

    def test_broken_checkpoint_does_not_crash_fingerprint_or_benchmark_matrix(self, tmp_path):
        bad_ckpt = tmp_path / "bad_agent.json"
        bad_ckpt.write_text(json.dumps({"config": {"num_actions": 1e309}, "q_table": {}}), encoding="utf-8")
        cfg = ExperimentConfig(scenario_name="cologne1", algorithm="Q-Learning", model_path=str(bad_ckpt))
        fp = compute_run_fingerprint(cfg)
        assert isinstance(fp, str)

    def test_cli_does_not_fail_due_to_past_historical_records(self, tmp_path, monkeypatch):
        # Historical record that failed
        monkeypatch.chdir(tmp_path)
        eval_dir = tmp_path / "results" / "evaluation"
        eval_dir.mkdir(parents=True)
        master = eval_dir / "benchmark_runs.json"
        master.write_text(json.dumps([{
            "scenario_name": "old_scenario",
            "algorithm": "Fixed-Time",
            "seed": 999,
            "status": "FAILED",
            "is_valid": False,
            "run_fingerprint": "old_failed_hash",
        }]), encoding="utf-8")

        monkeypatch.setattr(
            evaluation,
            "run_single_experiment",
            lambda cfg: {"status": "SUCCESS", "is_valid": True, "scenario_name": cfg.scenario_name, "algorithm": cfg.algorithm, "seed": cfg.seed},
        )
        monkeypatch.setattr(
            evaluation,
            "load_experiments_yaml",
            lambda path: {"simulation": {"steps": 10}},
        )
        ret = evaluation.main(["--run-all", "--scenario", "cologne1", "--algorithm", "Fixed-Time", "--seed", "42", "--config", str(tmp_path / "dummy.yaml")])
        assert ret == 0
        persisted = json.loads(master.read_text(encoding="utf-8"))
        assert any(record.get("run_fingerprint") == "old_failed_hash" for record in persisted)
        assert any(record.get("scenario_name") == "cologne1" for record in persisted)


def comparable_record(algorithm="Fixed-Time", seed=42, **changes):
    return {
        "scenario_name": "audit", "algorithm": algorithm, "seed": seed,
        "status": "SUCCESS", "is_valid": True, "mean_waiting_time": 100.0,
        "total_steps": 3600, "effective_config": {"steps": 3600, "step_length": 1.0},
        "observation_window_begin": 0.0, "observation_window_end": 3600.0,
        "observation_window_duration": 3600.0, **changes,
    }


@pytest.mark.parametrize("changes", [
    {"seed": 99}, {"total_steps": 60},
    {"effective_config": {"steps": 60, "step_length": 1.0}},
    {"effective_config": {"steps": 3600, "step_length": 0.5}},
    {"observation_window_begin": 100.0, "observation_window_end": 3700.0},
    {"observation_window_duration": 60.0},
    {"metric_schema_version": "old"},
])
def test_improvements_reject_incompatible_experiments(changes):
    with pytest.raises(ValueError, match="Cannot compare"):
        aggregate_benchmark_statistics([
            comparable_record(), comparable_record("Max-Pressure", **changes),
        ])


def test_improvements_require_simulation_metadata():
    records = [{key: value for key, value in comparable_record(algo).items()
                if key not in ("effective_config", "total_steps")}
               for algo in ("Fixed-Time", "Max-Pressure")]
    stats = aggregate_benchmark_statistics(records)["audit"]["Max-Pressure"]
    assert not stats.get("improvements_vs_baseline")


def test_improvements_match_seeds_independently_of_record_order():
    records = [comparable_record(seed=1), comparable_record(seed=2),
               comparable_record("Max-Pressure", seed=2, mean_waiting_time=20.0),
               comparable_record("Max-Pressure", seed=1, mean_waiting_time=40.0)]
    stats = aggregate_benchmark_statistics(records)["audit"]["Max-Pressure"]
    assert stats["improvements_vs_baseline"]["mean_waiting_time_improvement_pct"] == 70.0


def test_partial_failures_preserve_summary_without_unpaired_improvements():
    result = aggregate_benchmark_statistics([
        comparable_record(seed=1), comparable_record(seed=2),
        comparable_record("Max-Pressure", seed=1),
        comparable_record("Max-Pressure", seed=2, status="FAILED", is_valid=False),
    ])["audit"]["Max-Pressure"]
    assert result["status"] == "PARTIAL"
    assert result["failed_seeds"] == [2]
    assert "comparison_unavailable_reason" in result
    assert "improvements_vs_baseline" not in result


def test_real_short_runs_keep_summary_status_and_comparison_window_consistent(tmp_path):
    records = []
    for algorithm in ("Fixed-Time", "Max-Pressure"):
        record, summary = run_single_experiment(
            ExperimentConfig("cologne1", algorithm, steps=60, output_dir=str(tmp_path)),
            include_summary=True,
        )
        assert record["status"] == "SUCCESS"
        assert summary["tripinfo_reconciliation_status"] == "MATCH"
        assert record["service_status"] == summary["service_status"] == "LOW_COMPLETION"
        for key in ("observation_window_begin", "observation_window_end", "observation_window_duration"):
            assert record[key] == summary[key]
        assert record["observation_window_duration"] == 60.0
        records.append(record)
    stats = aggregate_benchmark_statistics(records)["cologne1"]["Max-Pressure"]
    assert "throughput_vph_improvement_pct" in stats["improvements_vs_baseline"]


def test_direct_experiment_execution(tmp_path):
    cfg = ExperimentConfig("cologne1", "Fixed-Time", steps=10, output_dir=str(tmp_path))
    record, summary = run_single_experiment(cfg, include_summary=True)
    assert record["status"] == "SUCCESS"
    assert record["is_valid"] is True
    assert record["total_steps"] == 10
    assert summary["total_steps"] == 10


def test_checkpoint_preserves_watchdog_in_evaluation(tmp_path):
    path = tmp_path / "model.json"
    QLearningController(8, max_red=37).save(path)
    agent, env = create_evaluation_session("cologne1", str(path))
    try:
        assert agent.max_red == env.config.max_red == 37
        assert env.runner.tls_managers[env.tls_id].watchdog.max_red == 37
    finally:
        env.close()
    agent.load(path)
    QLearningController(8, max_red=None).save(path)
    agent.load(path)
    assert agent.max_red is None


def test_run_all_honors_explicit_single_filters(monkeypatch):
    captured = {}

    def run(**kwargs):
        captured.update(kwargs)
        return [], {}

    monkeypatch.setattr(evaluation, "run_full_benchmark_matrix", run)
    assert evaluation.main(["--run-all", "--scenario", "cologne3", "--algorithm", "mp", "--seed", "0"]) == 0
    assert captured["scenarios"] == ["cologne3"]
    assert captured["algorithms"] == ["Max-Pressure"]
    assert captured["seeds"] == [0]


def test_checkpoint_resolution_failure_returns_failed_record(monkeypatch):
    def resolve(_):
        raise ValueError("Historical checkpoint is stale")

    monkeypatch.setattr(evaluation, "resolve_model_path", resolve)
    result = evaluation.run_single_experiment(evaluation.ExperimentConfig("cologne1", "ql"))
    assert result["status"] == "FAILED"
    assert "stale" in result["error_message"]


@pytest.mark.parametrize("overrides, expected", [({}, None), ({"max_green": 40}, 40), ({"max_green": None}, None)])
def test_simulation_inherits_checkpoint_and_explicit_overrides(tmp_path, overrides, expected):
    from src.rl.environment import TrafficSignalRLEnv
    probe = TrafficSignalRLEnv("cologne1")
    path = str(tmp_path / "agent.json")
    QLearningController(probe.num_actions, min_green=7, max_green=None, decision_interval=10).save(path)
    config = ExperimentConfig("cologne1", "Q-Learning", steps=0, model_path=path, timing_overrides=overrides)
    record, summary = run_single_experiment(config, include_summary=True)
    assert record["status"] == "SUCCESS"
    assert summary["min_green"] == 7
    assert summary["max_green"] == expected
    assert summary["decision_interval"] == 10


def test_fingerprint_uses_effective_timing(tmp_path):
    path = str(tmp_path / "agent.json")
    QLearningController(2, max_green=None).save(path)
    inherited = ExperimentConfig("cologne1", "Q-Learning", model_path=path)
    explicit = replace(inherited, timing_overrides={"max_green": None})
    assert compute_run_fingerprint(inherited) == compute_run_fingerprint(explicit)


def test_benchmark_rejects_diagnostic_run_without_tripinfo():
    result = run_single_experiment(ExperimentConfig("cologne1", "Fixed-Time", use_tripinfo=False, steps=0))
    assert result["status"] == "FAILED"


def test_full_summary_distinguishes_no_demand():
    _, summary = run_single_experiment(ExperimentConfig("cologne1", "Fixed-Time", steps=0), include_summary=True)
    assert summary["service_status"] == summary["completion_status"] == "NO_DEMAND"


@pytest.mark.parametrize("data", [{"config": None, "q_table": {}}, {"config": {"num_actions": 2}, "q_table": None}])
def test_malformed_checkpoint_does_not_stop_benchmark_matrix(tmp_path, monkeypatch, data):
    path = tmp_path / "broken.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    config = tmp_path / "experiment.yaml"
    config.write_text("simulation:\n  steps: 0\n", encoding="utf-8")
    monkeypatch.setattr(evaluation, "resolve_q_learning_checkpoint", lambda _: str(path))
    records, _ = evaluation.run_full_benchmark_matrix(
        scenarios=["cologne1"], algorithms=["Q-Learning", "Fixed-Time"], seeds=[42],
        config_file=str(config), output_dir=str(tmp_path / "evaluation"), generate_report=False,
    )
    assert [record["status"] for record in records] == ["FAILED", "SUCCESS"]


@pytest.mark.parametrize("steps", [0, 17, None])
def test_master_benchmark_window_overrides_single_simulation(tmp_path, steps):
    from src.training.train_q_learning import TrainingConfig
    path = tmp_path / "master.yaml"
    path.write_text(
        "simulation:\n  steps: 999\nsafety:\n  max_green: null\n  max_red: 37\n"
        f"benchmark:\n  steps: {steps if steps is not None else 'null'}\n",
        encoding="utf-8",
    )
    simulation = load_experiments_yaml(str(path))["simulation"]
    assert simulation["steps"] == steps
    assert simulation["max_green"] is None
    assert simulation["max_red"] == 37
    training_cfg = TrainingConfig.from_yaml("configs/config.yaml")
    benchmark = load_experiments_yaml()["simulation"]
    assert benchmark["steps"] == 3600
    for key in ("yellow_duration", "all_red_duration", "min_green", "max_green", "max_red"):
        assert getattr(training_cfg, key) == benchmark[key]


def test_missing_explicit_benchmark_config_is_not_replaced():
    with pytest.raises(FileNotFoundError):
        load_experiments_yaml("missing-custom-experiment.yaml")


def test_corrupt_resume_history_remains_available(tmp_path):
    path = tmp_path / "benchmark_runs.json"
    path.write_text("[broken", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot resume"):
        run_full_benchmark_matrix(output_dir=str(tmp_path))
    assert path.read_text(encoding="utf-8") == "[broken"


def test_requested_report_failure_is_visible(tmp_path, monkeypatch):
    from src.evaluation import compare_algorithms

    def fail(**kwargs):
        raise OSError("report disk unavailable")

    monkeypatch.setattr(compare_algorithms, "generate_comparison_report", fail)
    with pytest.raises(OSError, match="report disk unavailable"):
        evaluation.run_full_benchmark_matrix(
            scenarios=["cologne1"], algorithms=["Fixed-Time"], seeds=[42],
            config_file="configs/config.yaml", output_dir=str(tmp_path),
        )
    assert (tmp_path / "benchmark_runs.json").is_file()

