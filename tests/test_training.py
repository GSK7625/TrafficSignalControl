"""Unit tests for Q-Learning training and ValidationCheckpointSelector."""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

import src.training.train_q_learning as training
from src.training.train_q_learning import ValidationCheckpointSelector, is_validation_viable


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
            "summary": {"is_valid": True, "total_completed_vehicles": 96, "scheduled_demand_in_window": 100},
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

        evaluations = []
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



class TestValidationCheckpointSelector:
    """Test suite for ValidationCheckpointSelector and viability logic."""

    def test_selector_rejects_invalid_measurements(self):
        result = {"reward": 100, "completion_rate": 1.0, "summary": {"is_valid": False}}
        assert not ValidationCheckpointSelector().consider(1, [result])

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
