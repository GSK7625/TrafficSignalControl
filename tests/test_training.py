"""Unit tests for Q-Learning training and ValidationCheckpointSelector."""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

import src.training.train_q_learning as training
from src.training.train_q_learning import ValidationCheckpointSelector, is_validation_viable


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf"), -0.1, 1.1])
@pytest.mark.parametrize("entry", ["config", "selector", "viability"])
def test_completion_threshold_must_be_finite_fraction(threshold, entry):
    with pytest.raises(ValueError, match="completion"):
        if entry == "config":
            training.TrainingConfig(min_val_completion=threshold)
        elif entry == "selector":
            ValidationCheckpointSelector(threshold)
        else:
            is_validation_viable({"completed_vehicles": 1, "total_demand": 100}, threshold)


@pytest.mark.parametrize("threshold", [0.0, 1.0])
def test_completion_threshold_accepts_boundaries(threshold):
    training.TrainingConfig(min_val_completion=threshold)
    assert ValidationCheckpointSelector(threshold).min_completion == threshold


@pytest.fixture
def training_env(monkeypatch):
    """Mock environment for training loop tests."""
    env = MagicMock()
    env.num_actions = 2
    env.tls_id = "junction-2"
    info = {"discrete_state": (0, 0, 0, 0), "valid_actions": [0, 1]}
    env.reset.return_value = (None, info)
    env.step.return_value = (None, -1.0, True, False, info)
    env.get_summary.return_value = {"total_completed_vehicles": 96, "mean_waiting_time": 1.0}
    monkeypatch.setattr(training, "TrafficSignalRLEnv", lambda **_: env)
    return env


class TestQLearningTraining:
    """Test suite for Q-Learning training orchestrator."""

    @pytest.mark.parametrize("threshold, accepted", [(0.99, False), (0.90, True)])
    def test_training_honors_completion_threshold(self, tmp_path, monkeypatch, training_env, threshold, accepted):
        monkeypatch.setattr(training, "evaluate_greedy_q_learning", lambda **_: {
            "reward": -2.0, "completion_rate": 0.96, "validation_viable": True,
            "summary": {
                "is_valid": True, "total_completed_vehicles": 96, "scheduled_demand_in_window": 100,
                "tripinfo_required": True, "tripinfo_reconciliation_status": "MATCH",
            },
        })
        result = training.train_q_learning(
            num_episodes=1, val_interval=1, val_seeds=[7], min_val_completion=threshold,
            model_dir=str(tmp_path / "models"), log_dir=str(tmp_path / "logs"),
        )
        assert result["best_is_viable"] is accepted
        assert (tmp_path / "models" / "q_learning_ingolstadt1_best.json").exists() is accepted

    def test_validation_details_have_separate_seed_directories(self, tmp_path, monkeypatch, training_env):
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

    def test_training_with_explicit_training_config(self, tmp_path, monkeypatch, training_env):
        from src.training.train_q_learning import TrainingConfig

        monkeypatch.setattr(training, "evaluate_greedy_q_learning", lambda **kw: {"reward": -1.0, "completion_rate": 1.0, "validation_viable": True, "is_valid": True})

        cfg = TrainingConfig(
            scenario_name="ingolstadt1",
            num_episodes=1,
            val_interval=1,
            learning_rate=0.2,
            gamma=0.9,
            model_dir=str(tmp_path / "models"),
            log_dir=str(tmp_path / "logs"),
        )
        result = training.train_q_learning(config=cfg)
        assert result["hyperparameters"]["learning_rate"] == 0.2
        assert result["hyperparameters"]["gamma"] == 0.9
        assert (tmp_path / "models" / "q_learning_ingolstadt1_final.json").exists()

    @pytest.mark.parametrize("episodes", [0, -1])
    def test_training_rejects_non_positive_episodes(self, tmp_path, monkeypatch, training_env, episodes):
        checkpoint_dir = tmp_path / "models"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_ckpt = checkpoint_dir / "q_learning_ingolstadt1_final.json"
        final_ckpt.write_text('{"q_table": {"test": 99}}', encoding="utf-8")

        with pytest.raises(ValueError, match="num_episodes must be positive"):
            training.train_q_learning(
                num_episodes=episodes,
                model_dir=str(checkpoint_dir),
                log_dir=str(tmp_path / "logs"),
            )
        # Checkpoint should remain untouched
        assert json.loads(final_ckpt.read_text(encoding="utf-8")) == {"q_table": {"test": 99}}



class TestValidationCheckpointSelector:
    """Test suite for ValidationCheckpointSelector and viability logic."""

    def test_selector_rejects_invalid_measurements(self):
        result = {"reward": 100, "completion_rate": 1.0, "summary": {"is_valid": False}}
        assert not ValidationCheckpointSelector().consider(1, [result])

    @pytest.mark.parametrize("summary_meta, should_accept", [
        ({}, False),
        ({"tripinfo_required": False}, False),
        ({"tripinfo_required": True, "tripinfo_reconciliation_status": "MISSING"}, False),
        ({"tripinfo_required": True, "tripinfo_reconciliation_status": "MISMATCH"}, False),
        ({"tripinfo_required": True, "tripinfo_reconciliation_status": "NOT_REQUIRED"}, False),
        ({"tripinfo_required": True, "tripinfo_reconciliation_status": "MATCH"}, True),
    ])
    def test_selector_strictly_requires_tripinfo_match(self, summary_meta, should_accept):
        result = {
            "reward": 10.0,
            "completion_rate": 1.0,
            "summary": {
                "is_valid": True,
                "scheduled_demand_in_window": 100,
                "total_completed_vehicles": 100,
                **summary_meta,
            },
        }
        assert ValidationCheckpointSelector().consider(1, [result]) is should_accept

    def test_validation_keeps_legacy_top_level_completed_count(self):
        assert is_validation_viable({
            "summary": {"scheduled_demand_in_window": 100}, "completed_vehicles": 96,
        }) == (True, [])

    @pytest.mark.parametrize("result", [
        {},
        {"reward": float("nan"), "completion_rate": 1.0},
        {"reward": 0.0, "summary": {"scheduled_demand_in_window": 0}},
    ])
    def test_selector_rejects_missing_nonfinite_or_no_demand_results(self, result):
        assert ValidationCheckpointSelector().consider(1, [result]) is False

    def test_selector_does_not_infer_perfect_completion_from_completed_count(self):
        assert not ValidationCheckpointSelector().consider(1, [{"reward": 1.0, "completed_vehicles": 1}])

    def test_evaluate_greedy_q_learning_passes_target_tls_and_matches_actions(self, monkeypatch):
        from src.controllers.q_learning import QLearningController
        from src.training.train_q_learning import evaluate_greedy_q_learning

        env = MagicMock()
        env.num_actions = 4
        env.tls_id = "360086"
        env.reset.return_value = (None, {"discrete_state": (0, 0, 0, 0), "valid_actions": [0, 1, 2, 3]})
        env.step.return_value = (None, 0.0, True, False, {"discrete_state": (0, 0, 0, 0), "valid_actions": [0, 1, 2, 3]})
        env.get_summary.return_value = {"is_valid": True, "total_completed_vehicles": 10, "scheduled_demand_in_window": 10}

        created_kwargs = {}
        def mock_env_factory(**kwargs):
            created_kwargs.update(kwargs)
            return env

        monkeypatch.setattr(training, "TrafficSignalRLEnv", mock_env_factory)

        agent = QLearningController(num_actions=4, target_tls_id="360086")
        res = evaluate_greedy_q_learning(agent, scenario_name="cologne3")
        assert created_kwargs.get("target_tls_id") == "360086"
        assert res["validation_viable"] is True

    def test_evaluate_greedy_q_learning_rejects_action_mismatch(self, monkeypatch):
        from src.controllers.q_learning import QLearningController
        from src.training.train_q_learning import evaluate_greedy_q_learning

        env = MagicMock()
        env.num_actions = 3
        env.tls_id = "360082"
        monkeypatch.setattr(training, "TrafficSignalRLEnv", lambda **kw: env)

        agent = QLearningController(num_actions=4, target_tls_id="360086")
        with pytest.raises(ValueError, match="actions"):
            evaluate_greedy_q_learning(agent, scenario_name="cologne3")
        env.close.assert_called_once()
