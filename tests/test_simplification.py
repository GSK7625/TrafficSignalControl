"""Behavior checks for the shared configuration, execution and accounting paths."""

from dataclasses import replace
import json

import pytest

from src.data_inspector import get_window_demand, inspect_scenario
from src.metrics import MetricsCollector
from src.runner import SimulationRunner
from src.config import get_scenario_config
from src.training.train_q_learning import TrainingConfig, ValidationCheckpointSelector, train_q_learning
from src.controllers.q_learning import QLearningController
from src.evaluation.experiment_runner import ExperimentConfig, run_single_experiment, compute_run_fingerprint
from src.rl.environment import TrafficSignalRLEnv


def test_window_demand_excludes_departures_at_end():
    info = replace(inspect_scenario("cologne1"), departure_times=[0, 9, 10, 19, 20])
    assert get_window_demand(info, 10, 20)["scheduled_demand_in_window"] == 2
    assert get_window_demand(info, 10, 10)["scheduled_demand_in_window"] == 0


def test_runner_accounts_only_demand_in_short_window():
    runner = SimulationRunner(get_scenario_config("cologne1", max_steps=10, use_tripinfo=True))
    summary = runner.run()
    expected = get_window_demand(runner.scenario_info, runner.config.begin_time, runner.sim_time)["scheduled_demand_in_window"]
    assert summary["scheduled_demand_in_window"] == expected
    assert summary["demand_denominator"] == expected
    assert summary["tripinfo_reconciliation_status"] == "MATCH"


@pytest.mark.parametrize("ids", [["other"], [], ["arrived", "extra"], ["arrived", "arrived"]])
def test_tripinfo_must_match_arrived_ids(tmp_path, ids):
    path = tmp_path / "tripinfo.xml"
    trips = ''.join(f'<tripinfo id="{vid}" duration="5" waitingTime="1" timeLoss="2"/>' for vid in ids)
    path.write_text(f'<tripinfos>{trips}</tripinfos>', encoding="utf-8")
    collector = MetricsCollector("test", str(path))
    collector.completed_ids = {"arrived"}
    collector._cumulative_completed = 1
    assert collector.reconcile_tripinfo()["tripinfo_reconciliation_status"] == "MISMATCH"
    assert collector.get_summary()["is_valid"] is False


def test_departure_and_arrival_in_same_step_does_not_query_removed_vehicle(mock_traci):
    mock_traci.simulation.getDepartedIDList.return_value = ["short"]
    mock_traci.simulation.getArrivedIDList.return_value = ["short"]
    mock_traci.vehicle.getDeparture.side_effect = RuntimeError("vehicle already removed")
    collector = MetricsCollector("test")
    collector.record_step(1, 1.0, [], mock_traci)
    assert collector.completed_ids == {"short"}
    assert collector._cumulative_completed == 1


def test_training_yaml_preserves_explicit_null_override(tmp_path):
    path = tmp_path / "training.yaml"
    path.write_text("max_green: 25\nhyperparameters:\n  seed: 17\nmodel_dir: custom_models\n", encoding="utf-8")
    config = TrainingConfig.from_yaml(str(path), max_green=None)
    assert config.max_green is None
    assert config.seed == 17
    assert config.model_dir == "custom_models"


def test_training_rejects_ambiguous_config_and_overrides(tmp_path):
    with pytest.raises(TypeError, match="config"):
        train_q_learning(config=TrainingConfig(num_episodes=1, model_dir=str(tmp_path / "models"), log_dir=str(tmp_path / "logs")), num_episodes=2)


@pytest.mark.parametrize("overrides, expected", [({}, None), ({"max_green": 40}, 40), ({"max_green": None}, None)])
def test_simulation_inherits_checkpoint_and_explicit_overrides(tmp_path, overrides, expected):
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


def test_chart_generation_reads_custom_summary(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from src.evaluation.plot_comparison import generate_all_phase6_charts
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"custom": {"Fixed-Time": {
        "mean_queue_length": {"mean": 2, "std": 0},
        "total_completed_vehicles": {"mean": 5, "std": 0},
    }}}), encoding="utf-8")
    files = generate_all_phase6_charts(summary_file=str(path), output_dir=str(tmp_path / "figures"), dashboard_only=True)
    assert len(files) == 1
    assert files[0].endswith("custom_benchmark_dashboard.png")


def test_retired_cyclic_mode_is_rejected_in_runner_override():
    with pytest.raises(ValueError, match="control_mode"):
        SimulationRunner(get_scenario_config("cologne1"), control_mode="cyclic")


@pytest.mark.parametrize("key", ["min_green", "yellow_duration", "all_red_duration", "decision_interval"])
def test_legacy_checkpoint_nulls_keep_loader_defaults(tmp_path, key):
    from src.evaluation.experiment_runner import effective_timing
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"config": {"num_actions": 2, key: None}, "q_table": {}}), encoding="utf-8")
    config = ExperimentConfig("cologne1", "Q-Learning", model_path=str(path))
    assert effective_timing(config)[key] == QLearningController.from_checkpoint(str(path)).get_config()[key]


def test_full_summary_distinguishes_no_demand():
    _, summary = run_single_experiment(ExperimentConfig("cologne1", "Fixed-Time", steps=0), include_summary=True)
    assert summary["service_status"] == summary["completion_status"] == "NO_DEMAND"


def test_diagnostic_summary_cannot_select_best_checkpoint():
    result = {"reward": 0, "summary": {"is_valid": True, "total_completed_vehicles": 100,
              "scheduled_demand_in_window": 100, "tripinfo_required": False}}
    assert ValidationCheckpointSelector().consider(1, [result]) is False


@pytest.mark.parametrize("data", [{"config": None, "q_table": {}}, {"config": {"num_actions": 2}, "q_table": None}])
def test_malformed_checkpoint_does_not_stop_benchmark_matrix(tmp_path, monkeypatch, data):
    from src.evaluation import experiment_runner as evaluation
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
