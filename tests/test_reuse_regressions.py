"""Behavioral regressions for shared controller, training and evaluation setup."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.controllers import make_controller
from src.controllers.q_learning import QLearningController
from src.evaluation.evaluate_rl import run_greedy_eval_loop


def test_load_checkpoint_restores_optional_target(tmp_path):
    path = str(tmp_path / "agent.json")
    original = QLearningController(num_actions=3, target_tls_id="junction-2")
    original.save(path)
    loaded = QLearningController(num_actions=2)
    loaded.load(path)
    assert loaded.target_tls_id == "junction-2"
    assert loaded.num_actions == 3
    assert loaded.action_counts.shape == (3,)


@pytest.mark.parametrize("algorithm", ["max-pressure", "q-learning"])
def test_factory_distinguishes_default_and_disabled_max_green(algorithm, tmp_path):
    path = str(tmp_path / "agent.json")
    QLearningController(2).save(path)
    default, _ = make_controller(algorithm, model_path=path)
    disabled, _ = make_controller(algorithm, model_path=path, max_green=None)
    assert default.max_green == 60.0
    assert disabled.max_green is None


def test_checkpoint_override_can_disable_max_green(tmp_path):
    path = str(tmp_path / "agent.json")
    QLearningController(num_actions=2).save(path)
    assert QLearningController.from_checkpoint(path, max_green=None).max_green is None


def test_legacy_checkpoint_uses_safety_and_discretizer_defaults(tmp_path):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"config": {"num_actions": 2, "min_green": 7}, "q_table": {}}))
    agent = QLearningController.from_checkpoint(str(path))
    assert agent.max_green == 60.0
    assert agent.discretizer.min_green == 7.0


@pytest.mark.parametrize("fails", [False, True])
def test_greedy_evaluation_does_not_change_learning_state(fails):
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


@pytest.fixture
def training_env(monkeypatch):
    import src.training.train_q_learning as training

    env = MagicMock()
    env.num_actions = 2
    env.tls_id = "junction-2"
    info = {"discrete_state": (0, 0, 0, 0), "valid_actions": [0, 1]}
    env.reset.return_value = (None, info)
    env.step.return_value = (None, -1.0, True, False, info)
    env.get_summary.return_value = {"total_completed_vehicles": 96, "mean_waiting_time": 1.0}
    monkeypatch.setattr(training, "TrafficSignalRLEnv", lambda **_: env)
    return env


@pytest.mark.parametrize("threshold, accepted", [(0.99, False), (0.90, True)])
def test_training_honors_completion_threshold(tmp_path, monkeypatch, training_env, threshold, accepted):
    import src.training.train_q_learning as training

    monkeypatch.setattr(training, "evaluate_greedy_q_learning", lambda **_: {
        "reward": -2.0, "completion_rate": 0.96, "validation_viable": True,
        "summary": {"is_valid": True, "total_completed_vehicles": 96, "scheduled_demand_in_window": 100},
    })
    result = training.train_q_learning(
        num_episodes=1, val_interval=1, val_seeds=[7], min_val_completion=threshold,
        model_dir=str(tmp_path / "models"), log_dir=str(tmp_path / "logs"),
    )
    assert result["best_is_viable"] is accepted
    assert (tmp_path / "models" / "q_learning_ingolstadt1_best.json").exists() is accepted


def test_validation_details_have_separate_seed_directories(tmp_path, monkeypatch, training_env):
    import src.training.train_q_learning as training

    evaluations = []

    def evaluate(**kwargs):
        evaluations.append(kwargs)
        return {"reward": -2.0, "completion_rate": 1.0, "validation_viable": True, "is_valid": True}

    monkeypatch.setattr(training, "evaluate_greedy_q_learning", evaluate)
    result = training.train_q_learning(
        num_episodes=1, val_interval=1, val_seeds=[7, 8], save_run_details=True,
        model_dir=str(tmp_path / "models"), log_dir=str(tmp_path / "logs"),
    )
    assert [e["seed"] for e in evaluations] == [7, 8]
    assert [e["output_dir"] for e in evaluations] == [
        str(tmp_path / "logs" / "validation" / "ep1_seed7"),
        str(tmp_path / "logs" / "validation" / "ep1_seed8"),
    ]
    saved = json.loads(Path(result["final_checkpoint_path"]).read_text(encoding="utf-8"))
    assert saved["config"]["target_tls_id"] == "junction-2"


def test_selector_rejects_invalid_measurements():
    from src.training.train_q_learning import ValidationCheckpointSelector

    result = {"reward": 100, "completion_rate": 1.0, "summary": {"is_valid": False}}
    assert not ValidationCheckpointSelector().consider(1, [result])


def test_validation_keeps_legacy_top_level_completed_count():
    from src.training.train_q_learning import is_validation_viable

    assert is_validation_viable({
        "summary": {"scheduled_demand_in_window": 100}, "completed_vehicles": 96,
    }) == (True, [])


def test_evaluation_setup_shares_checkpoint_target_and_disabled_timing(tmp_path, monkeypatch):
    import src.evaluation.evaluate_rl as evaluation

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

    monkeypatch.setattr(evaluation, "TrafficSignalRLEnv", make_env)
    agent, actual_env = evaluation.create_evaluation_session(
        "cologne3", path, seed=17, timing_overrides={"decision_interval": 10.0},
    )
    assert actual_env is env
    assert options["target_tls_id"] == "junction-2"
    assert options["max_green"] is None
    assert options["decision_interval"] == agent.decision_interval == 10.0
    assert agent.discretizer is env.state_discretizer
    assert agent.discretizer.serving_thresholds == (2.0, 8.0)


def test_evaluation_setup_closes_env_on_action_count_mismatch(tmp_path, monkeypatch):
    import src.evaluation.evaluate_rl as evaluation

    path = str(tmp_path / "agent.json")
    QLearningController(num_actions=3).save(path)
    env = MagicMock()
    env.num_actions = 2
    monkeypatch.setattr(evaluation, "TrafficSignalRLEnv", lambda **_: env)
    with pytest.raises(ValueError, match="actions"):
        evaluation.create_evaluation_session("cologne1", path)
    env.close.assert_called_once()


def test_checkpoint_metadata_keeps_explicit_disabled_max_green(tmp_path):
    from src.evaluation.evaluate_rl import load_checkpoint_env_params

    path = str(tmp_path / "agent.json")
    QLearningController(num_actions=2, max_green=None).save(path)
    assert load_checkpoint_env_params(path)["max_green"] is None


def test_completion_statistics_uses_realized_demand_and_distinct_policies():
    from src.metrics import completion_statistics, service_status, REPORT_COMPLETION_THRESHOLD

    completed, demand, rate = completion_statistics({
        "total_completed_vehicles": 96, "running_vehicles_at_end": 3, "pending_insertion_vehicles": 1,
    })
    assert (completed, demand, rate) == (96, 100, 0.96)
    assert service_status(rate) == "COMPLETED"
    assert service_status(rate, REPORT_COMPLETION_THRESHOLD) == "LOW_COMPLETION"


def test_rl_environment_uses_shared_all_red_default():
    import inspect
    from src.config import DEFAULT_SAFETY_CONFIG
    from src.rl.environment import TrafficSignalRLEnv

    assert inspect.signature(TrafficSignalRLEnv).parameters["all_red_duration"].default == DEFAULT_SAFETY_CONFIG["all_red_duration"]


def test_comparison_plots_skip_failed_algorithms(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    from src.evaluation.plot_comparison import plot_group_a_comparisons

    metrics = {
        key: {"mean": 10.0, "std": 1.0} for key in (
            "tripinfo_mean_waiting_time", "mean_queue_length", "total_completed_vehicles",
            "tripinfo_total_time_loss", "completion_rate_pct",
        )
    }
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"cologne1": {
        "Fixed-Time": metrics, "Q-Learning": {"status": "FAILED", "is_valid": False},
    }}))
    plots = plot_group_a_comparisons(str(path), str(tmp_path / "plots"))
    assert len(plots) == 5
    assert {Path(p).name for p in plots} == {
        f"cologne1_{suffix}_comparison.png" for suffix in (
            "waiting_time", "queue_length", "throughput", "time_loss", "completion_rate",
        )
    }


def test_dynamics_without_saved_run_details_does_not_crash(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.evaluation.plot_comparison import plot_group_b_dynamics

    path = tmp_path / "runs.json"
    path.write_text(json.dumps([{
        "scenario_name": "cologne1", "algorithm": "Fixed-Time", "is_valid": True, "output_dir": None,
    }]))
    try:
        assert plot_group_b_dynamics(str(path), str(tmp_path / "plots")) == []
    finally:
        plt.close("all")


@pytest.mark.parametrize("name", ["ql", "Q Learning", "q_learning", "QLearning"])
def test_algorithm_aliases_use_same_factory_and_benchmark_path(name, monkeypatch):
    from src.evaluation.experiment_runner import ExperimentConfig, resolve_effective_decision_interval, resolve_model_path

    agent, _ = make_controller(name)
    assert isinstance(agent, QLearningController)
    cfg = ExperimentConfig(scenario_name="cologne1", algorithm=name)
    assert resolve_effective_decision_interval(cfg) == 5.0
    assert resolve_model_path(cfg) == "models/q_learning/q_learning_cologne1_best.json"


def test_dashboard_handles_unavailable_waiting_metrics(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.evaluation.plot_comparison import plot_benchmark_dashboard

    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"cologne1": {"Fixed-Time": {
        "tripinfo_mean_waiting_time": {"mean": None, "std": None, "available": False},
        "mean_queue_length": {"mean": 10.0, "std": 1.0},
        "total_completed_vehicles": {"mean": 0.0, "std": 0.0},
    }}}))
    try:
        figures = plot_benchmark_dashboard(str(path), str(tmp_path / "plots"))
        assert len(figures) == 1
        assert Path(figures[0]).is_file()
    finally:
        plt.close("all")
