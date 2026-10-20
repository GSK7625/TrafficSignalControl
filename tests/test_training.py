"""Unit tests for Q-Learning training and ValidationCheckpointSelector."""

import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

import src.training.train_q_learning as training
from src.training.train_q_learning import ValidationCheckpointSelector, is_validation_viable


def test_project_benchmark_seeds_are_separate_from_training_and_validation():
    from src.config import load_yaml_config
    config = load_yaml_config("configs/config.yaml")
    benchmark_seeds = set(config["benchmark"]["seeds"])
    for scenario in config["benchmark"]["scenarios"]:
        cfg = training.TrainingConfig.from_yaml("configs/config.yaml", scenario_name=scenario)
        training_seeds = set(range(cfg.seed + 1, cfg.seed + cfg.num_episodes + 1))
        validation_seeds = set(cfg.val_seeds or [cfg.seed + 9999, cfg.seed + 10000])
        assert benchmark_seeds.isdisjoint(training_seeds | validation_seeds)


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

    def test_dataset_options_keep_shared_values_and_cli_precedence(self, tmp_path):
        path = tmp_path / "scenarios.yaml"
        path.write_text("""
q_learning:
  hyperparameters:
    gamma: 0.95
    learning_rate: 0.1
  discretization:
    group_by: phase
    serving_queue_thresholds: [0.5, 3]
  scenarios:
    cologne1:
      hyperparameters:
        learning_rate: 0.05
      discretization:
        group_by: aggregate
        serving_queue_thresholds: [2, 8]
""", encoding="utf-8")
        cologne = training.TrainingConfig.from_yaml(str(path), scenario_name="cologne1")
        ingolstadt = training.TrainingConfig.from_yaml(str(path), scenario_name="ingolstadt1")
        assert cologne.group_by == "aggregate" and cologne.serving_thresholds == (2, 8)
        assert cologne.learning_rate == 0.05 and cologne.gamma == 0.95
        assert ingolstadt.group_by == "phase" and ingolstadt.learning_rate == 0.1
        explicit = training.TrainingConfig.from_yaml(str(path), scenario_name="cologne1", group_by="phase")
        assert explicit.group_by == "phase"

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

    def test_training_config_group_by_support(self, tmp_path):
        from src.training.train_q_learning import TrainingConfig

        yaml_file = tmp_path / "q_learning.yaml"
        yaml_file.write_text("""
discretization:
  group_by: branch
  max_effective_green: 30.0
""", encoding="utf-8")

        cfg = TrainingConfig.from_yaml(str(yaml_file))
        assert cfg.group_by == "branch"
        assert cfg.max_effective_green == 30.0

        with pytest.raises(ValueError, match="group_by must be one of"):
            TrainingConfig(group_by="unknown_mode")

    def test_training_config_metrics_aligned_reward(self, tmp_path, monkeypatch, training_env):
        from src.training.train_q_learning import TrainingConfig, _create_reward_fn
        from src.rl.rewards import MetricsAlignedReward

        yaml_file = tmp_path / "config_reward.yaml"
        yaml_file.write_text("""
q_learning:
  reward:
    name: MetricsAlignedReward
    scale: 2.0
    w_completed: 3.0
    w_queue: 0.1
    w_waiting: 0.02
    w_switch: 1.0
    w_pending: 0.05
""", encoding="utf-8")

        cfg = TrainingConfig.from_yaml(str(yaml_file), scenario_name="ingolstadt1", num_episodes=1, val_interval=1, model_dir=str(tmp_path / "models"), log_dir=str(tmp_path / "logs"))
        assert cfg.reward_name == "MetricsAlignedReward"
        assert cfg.reward_scale == 2.0
        assert cfg.w_completed == 3.0
        assert cfg.w_queue == 0.1

        reward_fn = _create_reward_fn(cfg)
        assert isinstance(reward_fn, MetricsAlignedReward)
        assert reward_fn.scale == 2.0
        assert reward_fn.w_completed == 3.0

        monkeypatch.setattr(training, "evaluate_greedy_q_learning", lambda **kw: {"reward": 5.0, "completion_rate": 1.0, "validation_viable": True, "is_valid": True})
        result = training.train_q_learning(config=cfg)
        saved = json.loads(Path(result["final_checkpoint_path"]).read_text(encoding="utf-8"))
        assert saved["reward_config"]["name"] == "MetricsAlignedReward"

    def test_training_checkpoint_preserves_group_by(self, tmp_path, monkeypatch, training_env):
        from src.training.train_q_learning import TrainingConfig

        monkeypatch.setattr(training, "evaluate_greedy_q_learning", lambda **kw: {"reward": -1.0, "completion_rate": 1.0, "validation_viable": True, "is_valid": True})

        cfg = TrainingConfig(
            scenario_name="ingolstadt1",
            num_episodes=1,
            val_interval=1,
            group_by="aggregate",
            model_dir=str(tmp_path / "models"),
            log_dir=str(tmp_path / "logs"),
        )
        result = training.train_q_learning(config=cfg)
        final_ckpt = Path(result["final_checkpoint_path"])
        assert final_ckpt.exists()
        import json
        saved = json.loads(final_ckpt.read_text(encoding="utf-8"))
        assert saved["config"]["discretizer_config"]["group_by"] == "aggregate"

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

    def test_training_yaml_watchdog_reaches_training_validation_and_checkpoint(self, tmp_path, monkeypatch):
        from src.controllers.q_learning import QLearningController
        path = tmp_path / "train.yaml"
        path.write_text("max_red: 37\nhyperparameters:\n  num_episodes: 1\n", encoding="utf-8")
        cfg = training.TrainingConfig.from_yaml(
            str(path), val_interval=1, val_seeds=[7],
            model_dir=str(tmp_path / "models"), log_dir=str(tmp_path / "logs"),
        )
        observed = []
        env = MagicMock()
        env.num_actions, env.tls_id = 2, "junction"
        info = {"discrete_state": (0, 0, 0, 0), "valid_actions": [0, 1]}
        env.reset.return_value = (None, info)
        env.step.return_value = (None, -1, True, False, info)
        env.get_summary.return_value = {"mean_waiting_time": 0, "is_valid": True}

        def make_env(**options):
            observed.append(options)
            return env

        monkeypatch.setattr(training, "TrafficSignalRLEnv", make_env)
        training.train_q_learning(cfg)
        assert len(observed) == 2
        assert all(options["max_red"] == 37 for options in observed)
        saved = QLearningController.from_checkpoint(str(tmp_path / "models" / "q_learning_ingolstadt1_final.json"))
        assert saved.max_red == 37

    def test_invalid_training_measurements_do_not_write_final_checkpoint(self, tmp_path, monkeypatch):
        env = MagicMock()
        env.num_actions, env.tls_id = 2, "junction"
        info = {"discrete_state": (0, 0, 0, 0), "valid_actions": [0, 1]}
        env.reset.return_value = (None, info)
        env.step.return_value = (None, -1, True, False, info)
        env.get_summary.return_value = {"is_valid": False, "measurement_errors": 1}
        monkeypatch.setattr(training, "TrafficSignalRLEnv", lambda **_: env)
        with pytest.raises(ValueError, match="Invalid training measurements"):
            training.train_q_learning(num_episodes=1, model_dir=str(tmp_path / "models"), log_dir=str(tmp_path / "logs"))
        assert not list((tmp_path / "models").glob("*.json"))
        env.close.assert_called_once()

    def test_training_yaml_preserves_explicit_null_override(self, tmp_path):
        from src.training.train_q_learning import TrainingConfig
        path = tmp_path / "training.yaml"
        path.write_text("max_green: 25\nhyperparameters:\n  seed: 17\nmodel_dir: custom_models\n", encoding="utf-8")
        config = TrainingConfig.from_yaml(str(path), max_green=None)
        assert config.max_green is None
        assert config.seed == 17
        assert config.model_dir == "custom_models"

    def test_training_rejects_ambiguous_config_and_overrides(self, tmp_path):
        from src.training.train_q_learning import TrainingConfig, train_q_learning
        with pytest.raises(TypeError, match="config"):
            train_q_learning(config=TrainingConfig(num_episodes=1, model_dir=str(tmp_path / "models"), log_dir=str(tmp_path / "logs")), num_episodes=2)

    def test_diagnostic_summary_cannot_select_best_checkpoint(self):
        result = {"reward": 0, "summary": {"is_valid": True, "total_completed_vehicles": 100,
                  "scheduled_demand_in_window": 100, "tripinfo_required": False}}
        assert ValidationCheckpointSelector().consider(1, [result]) is False

