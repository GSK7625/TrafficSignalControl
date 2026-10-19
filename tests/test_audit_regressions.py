"""Failures found in the project audit, tested at their public boundaries."""
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.metrics import MetricsCollector


def test_required_metric_failure_is_not_a_zero_queue():
    backend = MagicMock()
    backend.simulation.getDepartedIDList.return_value = []
    backend.simulation.getArrivedIDList.return_value = []
    backend.vehicle.getIDList.return_value = []
    backend.lane.getLastStepHaltingNumber.side_effect = RuntimeError("connection lost")
    collector = MetricsCollector("audit")
    with pytest.raises(RuntimeError, match="connection lost"):
        collector.record_step(1, 1.0, ["lane"], backend)
    assert collector.step_records == []


def test_observation_failure_is_not_a_zero_queue():
    from src.config import get_scenario_config
    from src.controllers.base import ControlledLink
    from src.runner import SimulationRunner
    runner = SimulationRunner(get_scenario_config("cologne1"))
    tls = runner.scenario_info.tls_ids[0]
    runner.tls_controlled_links[tls] = [ControlledLink(0, "in", "out")]
    runner.traci = MagicMock()
    runner.traci.simulation.getPendingVehicles.return_value = []
    runner.traci.lane.getLastStepHaltingNumber.side_effect = RuntimeError("lane missing")
    with pytest.raises(RuntimeError, match="lane missing"):
        runner.get_observation()


def test_invalid_summary_is_not_promoted_to_success(tmp_path, monkeypatch):
    import src.evaluation.experiment_runner as evaluation
    runner = MagicMock()
    runner.run.return_value = {"is_valid": False, "measurement_errors": 3, "total_steps": 10}
    monkeypatch.setattr(evaluation, "SimulationRunner", lambda *a, **k: runner)
    result = evaluation.run_single_experiment(evaluation.ExperimentConfig("cologne1", "Fixed-Time", output_dir=str(tmp_path)))
    assert result["is_valid"] is False
    assert result["status"] == "FAILED"


@pytest.mark.parametrize("xml,status", [
    ("<tripinfos>broken", "PARSE_ERROR"),
    (None, "MISSING"),
])
def test_tripinfo_reconciliation_does_not_claim_false_match(tmp_path, xml, status):
    path = tmp_path / "tripinfo.xml"
    if xml is not None:
        path.write_text(xml)
    collector = MetricsCollector("audit", str(path))
    collector._cumulative_completed = 1
    assert collector.reconcile_tripinfo()["tripinfo_reconciliation_status"] == status


def test_early_summary_does_not_cache_missing_tripinfo(tmp_path):
    path = tmp_path / "tripinfo.xml"
    collector = MetricsCollector("audit", str(path))
    collector.get_summary()
    path.write_text("<tripinfos/>")
    assert collector.reconcile_tripinfo()["tripinfo_reconciliation_status"] == "MATCH"


def test_evaluation_requires_an_existing_checkpoint(tmp_path):
    from src.controllers import make_controller
    from src.evaluation.evaluate_rl import create_evaluation_session
    path = str(tmp_path / "missing.json")
    for create in (lambda: make_controller("q-learning", model_path=path),
                   lambda: create_evaluation_session("cologne1", path)):
        with pytest.raises(FileNotFoundError):
            create()


def test_checkpoint_shape_is_checked_at_load(tmp_path):
    from src.controllers.q_learning import QLearningController
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"config": {"num_actions": 3}, "q_table": {"(0, 0, 0, 1)": [10]}}))
    with pytest.raises(ValueError, match="shape"):
        QLearningController.from_checkpoint(str(path))


def test_checkpoint_rejects_unknown_override_and_changed_state_encoding(tmp_path):
    from src.controllers.q_learning import QLearningController
    path = str(tmp_path / "agent.json")
    QLearningController(2).save(path)
    with pytest.raises(TypeError):
        QLearningController.from_checkpoint(path, min_grean=9)
    with pytest.raises(ValueError, match="discretizer"):
        QLearningController.from_checkpoint(path, min_green=9)


def test_original_mode_does_not_claim_control_of_tls():
    from src.config import get_scenario_config
    from src.runner import SimulationRunner
    runner = SimulationRunner(get_scenario_config("cologne1", control_mode="original"))
    assert runner.get_controlled_tls_ids() == set()


def test_q_cli_respects_the_environment_action_mask():
    from src.controllers.q_learning import QLearningController
    agent = QLearningController(3, eval_mode=True)
    agent.q_table[(0, 0, 0, 2)] = np.array([100.0, 1.0, 50.0])
    obs = SimpleNamespace(current_phase=0, time_in_phase=60.0, is_in_transition=False,
                          green_phases=["Grr", "rGr", "rrG"], controlled_links=[],
                          in_lane_queues={}, valid_actions=[1, 2])
    assert agent.select_action({"tls": obs}) == {"tls": 2}


@pytest.mark.parametrize("actions", [[], [999]])
def test_empty_or_invalid_action_mask_is_not_expanded(actions):
    from src.controllers.q_learning import QLearningController
    with pytest.raises(ValueError):
        QLearningController(2, eval_mode=True).select_action_for_state((0, 0, 0, 0), actions)


def test_phase_manager_provides_valid_actions_at_boundaries():
    from src.phase_manager import SafePhaseManager
    manager = SafePhaseManager("tls", ["Grr", "rGr", "rrG"], min_green=5, max_green=60)
    assert manager.valid_actions() == [0]
    manager.update(5)
    assert manager.valid_actions() == [0, 1, 2]
    manager.update(55)
    assert manager.valid_actions() == [1, 2]
    manager.request_phase(2)
    assert manager.valid_actions() == [2]


def test_primary_collision_api_counts_vehicle_involvements():
    from src.config import get_scenario_config
    from src.runner import SimulationRunner
    runner = SimulationRunner(get_scenario_config("cologne1", control_mode="agent"))
    runner.traci = MagicMock()
    runner.traci.simulation.getTime.return_value = 1.0
    runner.traci.simulation.getStartingTeleportNumber.return_value = 0
    runner.traci.simulation.getCollisions.return_value = [SimpleNamespace(collider="a", victim="b")]
    runner.metrics_collector.record_step = MagicMock()
    runner.is_running = True
    runner.step({})
    assert runner.total_collision_vehicle_involvements == 2
    assert runner.unique_colliding_vehicles == {"a", "b"}


def test_finalize_is_idempotent_and_closes_on_accounting_error(monkeypatch):
    from src.config import get_scenario_config
    from src.runner import SimulationRunner
    runner = SimulationRunner(get_scenario_config("cologne1"))
    backend = MagicMock()
    backend.vehicle.getIDList.return_value = ["a"]
    backend.simulation.getPendingVehicles.return_value = []
    runner.traci = backend
    runner.is_running = True
    assert runner.finalize()["running_vehicles_at_end"] == 1
    assert runner.finalize()["running_vehicles_at_end"] == 1
    runner.reset()
    runner.traci = backend
    runner.is_running = True
    monkeypatch.setattr(runner.metrics_collector, "population_snapshot", lambda: (_ for _ in ()).throw(RuntimeError("accounting failed")))
    with pytest.raises(RuntimeError, match="accounting failed"):
        runner.finalize()
    assert runner.is_running is False


def test_env_reset_clears_summary_and_zero_limit_never_steps():
    from src.rl.environment import TrafficSignalRLEnv
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


def test_env_rejects_out_of_range_actions():
    from src.rl.environment import TrafficSignalRLEnv
    env = TrafficSignalRLEnv("cologne1", max_steps=5)
    try:
        env.reset()
        with pytest.raises(ValueError):
            env.step(env.num_actions)
    finally:
        env.close()


def test_fingerprint_changes_with_steps_and_model_contents(tmp_path):
    from dataclasses import replace
    from src.evaluation.experiment_runner import ExperimentConfig, compute_run_fingerprint
    path = tmp_path / "model.json"
    path.write_text("first")
    config = ExperimentConfig("cologne1", "Q-Learning", steps=20, model_path=str(path))
    first = compute_run_fingerprint(config, str(path))
    assert compute_run_fingerprint(replace(config, steps=99), str(path)) != first
    path.write_text("second")
    assert compute_run_fingerprint(config, str(path)) != first


def test_aggregation_preserves_failed_seeds_and_deduplicates_retries():
    from src.evaluation.experiment_runner import aggregate_benchmark_statistics
    good = {"scenario_name": "test", "algorithm": "Fixed-Time", "seed": 1,
            "status": "SUCCESS", "is_valid": True, "mean_queue_length": 10.0}
    failed = {**good, "seed": 2, "status": "FAILED", "is_valid": False, "error_message": "lost connection"}
    result = aggregate_benchmark_statistics([good, good, failed])["test"]["Fixed-Time"]
    assert result["status"] == "PARTIAL"
    assert result["run_count"] == 1
    assert result["failed_count"] == 1
    assert result["error_messages"] == ["lost connection"]


@pytest.mark.parametrize("result", [{}, {"reward": float("nan"), "completion_rate": 1.0},
                                    {"reward": 0.0, "summary": {"scheduled_demand_in_window": 0}}])
def test_selector_rejects_missing_nonfinite_or_no_demand_results(result):
    from src.training.train_q_learning import ValidationCheckpointSelector
    assert ValidationCheckpointSelector().consider(1, [result]) is False


def test_checkpoint_restores_reward_configuration(tmp_path):
    from src.controllers.q_learning import QLearningController
    from src.evaluation.evaluate_rl import create_evaluation_session
    from src.rl.environment import TrafficSignalRLEnv
    probe = TrafficSignalRLEnv("cologne1")
    agent = QLearningController(probe.num_actions)
    agent.reward_config = {"name": "NegativePressureReward", "scale": 2.0, "pending_weight": 0.5}
    path = str(tmp_path / "agent.json")
    agent.save(path)
    loaded, env = create_evaluation_session("cologne1", path)
    assert env.reward_fn.get_config() == {"name": "NegativePressureReward", "scale": 2.0, "pending_weight": 0.5}
    env.reward_fn.record_step({"in": 3}, pending_vehicles=4)
    assert env.reward_fn.calculate_reward() == -10.0


@pytest.mark.parametrize("options", [{"step_length": 0}, {"step_length": -1},
                                      {"decision_interval": -1}, {"yellow_duration": 0}])
def test_config_rejects_invalid_inputs_once(options):
    from src.config import SimulationConfig
    with pytest.raises(ValueError):
        SimulationConfig("cologne1", **options)


def test_environment_rejects_misspelled_options():
    from src.rl.environment import TrafficSignalRLEnv
    with pytest.raises(TypeError):
        TrafficSignalRLEnv("cologne1", decison_interval=99)


def test_experiments_loader_does_not_hide_bad_yaml(tmp_path):
    from src.evaluation.experiment_runner import load_experiments_yaml
    path = tmp_path / "bad.yaml"
    path.write_text("simulation: [unclosed")
    with pytest.raises(Exception):
        load_experiments_yaml(str(path))


def test_matrix_honors_yaml_timing_and_does_not_reuse_old_steps(tmp_path, monkeypatch):
    import src.evaluation.experiment_runner as evaluation
    path = tmp_path / "experiment.yaml"
    path.write_text("simulation:\n  steps: 10\n  step_length: 0.5\n  use_tripinfo: false\n  max_green: null\n")
    seen = []
    def run(config):
        seen.append(config)
        return {"scenario_name": "cologne1", "algorithm": "Fixed-Time", "seed": 42,
                "status": "SUCCESS", "is_valid": True, "total_steps": config.steps}
    monkeypatch.setattr(evaluation, "run_single_experiment", run)
    args = dict(scenarios=["cologne1"], algorithms=["Fixed-Time"], seeds=[42], config_file=str(path), output_dir=str(tmp_path))
    evaluation.run_full_benchmark_matrix(**args)
    assert seen[0].step_length == 0.5
    assert seen[0].use_tripinfo is False
    assert seen[0].max_green is None
    evaluation.run_full_benchmark_matrix(**args)
    assert len(seen) == 1
    path.write_text(path.read_text().replace("steps: 10", "steps: 99"))
    evaluation.run_full_benchmark_matrix(**args)
    assert [config.steps for config in seen] == [10, 99]


def test_report_handles_missing_metrics_and_zero_completion(tmp_path):
    from src.evaluation.experiment_runner import aggregate_benchmark_statistics
    from src.evaluation.compare_algorithms import generate_comparison_report
    records = [{"scenario_name": "cologne1", "algorithm": "Fixed-Time", "seed": 1,
                "status": "SUCCESS", "is_valid": True, "completion_rate_pct": 0.0,
                "collision_count": 0, "teleport_count": 0}]
    path = tmp_path / "summary.json"
    path.write_text(json.dumps(aggregate_benchmark_statistics(records)))
    report = tmp_path / "report.md"
    generate_comparison_report(eval_dir=str(tmp_path), phase6_summary=str(path), output_report=str(report))
    text = report.read_text(encoding="utf-8")
    assert "N/A" in text
    assert "LOW_COMPLETION" in text


def test_zero_baseline_does_not_invent_a_relative_percentage():
    from src.evaluation.experiment_runner import compute_improvement
    assert compute_improvement(0, 4) is None
    assert compute_improvement(float("nan"), 4) is None


def test_no_baseline_does_not_generate_improvement_chart(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from src.evaluation.plot_comparison import plot_group_a_comparisons
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"cologne3": {"Q-Learning": {"status": "SUCCESS", "is_valid": True,
                                                             "mean_queue_length": {"mean": 2.0, "std": 0}}}}))
    files = plot_group_a_comparisons(str(path), str(tmp_path))
    assert [__import__("pathlib").Path(file).name for file in files] == ["cologne3_queue_length_comparison.png"]


def test_phase_allocation_combines_seeds_before_plotting(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use("Agg")
    import src.evaluation.plot_comparison as charts
    runs = [{"scenario_name": "cologne1", "algorithm": "Fixed-Time", "is_valid": True,
             "output_dir": None, "phase_durations_per_tls": {"tls": phases}}
            for phases in ({"0": 10, "1": 90}, {"0": 90, "1": 10})]
    path = tmp_path / "runs.json"
    path.write_text(json.dumps(runs))
    heights = []
    original_save = charts.plt.Figure.savefig
    def capture(figure, *args, **kwargs):
        heights.append([bar.get_height() for bar in figure.axes[0].patches])
        return original_save(figure, *args, **kwargs)
    monkeypatch.setattr(charts.plt.Figure, "savefig", capture)
    files = charts.plot_group_b_dynamics(str(path), str(tmp_path))
    assert len(files) == 1
    assert heights == [[50.0, 50.0]]


def test_close_error_does_not_hide_simulation_error():
    from src.config import get_scenario_config
    from src.runner import SimulationRunner
    runner = SimulationRunner(get_scenario_config("cologne1"))
    runner.traci = MagicMock()
    runner.traci.start.side_effect = RuntimeError("simulation failed")
    with pytest.raises(RuntimeError, match="simulation failed"):
        runner.run()


def test_rl_environment_import_is_independent_of_import_order():
    import subprocess, sys
    result = subprocess.run([sys.executable, "-c", "from src.rl.environment import TrafficSignalRLEnv"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_missing_sumo_version_is_unknown(monkeypatch):
    import src.evaluation.experiment_runner as evaluation
    def unavailable(*args, **kwargs):
        raise FileNotFoundError("sumo")
    monkeypatch.setattr(evaluation.subprocess, "check_output", unavailable)
    assert evaluation.get_sumo_version() == "unknown"


def test_selector_does_not_infer_perfect_completion_from_completed_count():
    from src.training.train_q_learning import ValidationCheckpointSelector
    assert not ValidationCheckpointSelector().consider(1, [{"reward": 1.0, "completed_vehicles": 1}])


def test_runner_checks_q_action_count_before_simulation():
    from src.controllers.q_learning import QLearningController
    from src.config import get_scenario_config
    from src.runner import SimulationRunner
    with pytest.raises(ValueError, match="actions"):
        SimulationRunner(get_scenario_config("cologne1"), controller=QLearningController(99))


def test_nonfinite_tripinfo_is_invalid(tmp_path):
    path = tmp_path / "tripinfo.xml"
    path.write_text('<tripinfos><tripinfo id="a" duration="nan" waitingTime="nan" timeLoss="nan"/></tripinfos>')
    collector = MetricsCollector("audit", str(path))
    collector._cumulative_completed = 1
    assert collector.reconcile_tripinfo()["tripinfo_reconciliation_status"] == "PARSE_ERROR"


def test_no_arrivals_has_unavailable_waiting_mean(tmp_path):
    path = tmp_path / "tripinfo.xml"
    path.write_text("<tripinfos/>")
    collector = MetricsCollector("audit", str(path))
    collector.reconcile_tripinfo()
    assert collector.get_summary()["tripinfo_mean_waiting_time"] is None


def test_controller_cannot_expand_configured_tls_scope():
    from src.config import get_scenario_config
    from src.controllers.max_pressure import MaxPressureController
    from src.runner import SimulationRunner
    config = get_scenario_config("cologne3")
    target = config.scenario_info.tls_ids[0]
    config.target_tls_id = target
    runner = SimulationRunner(config, controller=MaxPressureController())
    try:
        runner.start()
        runner.step()
        assert runner.get_controlled_tls_ids() == {target}
        other = next(tls for tls in runner.tls_managers if tls != target)
        with pytest.raises(ValueError, match="controlled"):
            runner.step({other: 0})
    finally:
        runner.close()


def test_zero_demand_run_is_not_a_failed_completion(tmp_path):
    from src.evaluation.experiment_runner import ExperimentConfig, run_single_experiment
    result = run_single_experiment(ExperimentConfig("cologne1", "Fixed-Time", steps=0, output_dir=str(tmp_path)))
    assert result["tripinfo_mean_waiting_time"] is None
    assert result["tripinfo_mean_travel_time"] is None


def test_nonfinite_rewards_do_not_break_strict_json_export():
    from src.evaluation.experiment_runner import aggregate_benchmark_statistics
    result = aggregate_benchmark_statistics([{"scenario_name": "audit", "algorithm": "Q-Learning", "seed": 1,
                                              "status": "SUCCESS", "is_valid": True, "episode_reward": float("nan")}])
    assert result["audit"]["Q-Learning"]["episode_reward"]["mean"] is None
    json.dumps(result, allow_nan=False)
