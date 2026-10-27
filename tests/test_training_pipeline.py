"""Unit tests for Q-Learning training pipeline."""

import os
import tempfile
import pytest

from src.controllers.q_learning import QLearningController
from src.data_inspector import inspect_scenario
from src.training.train_q_learning import train_q_learning_agent


def test_train_q_learning_single_episode_short():
    """Verify train_q_learning_agent executes short episode and learns non-empty Q-table."""
    cfg_path = os.path.join("data", "cologne1", "cologne1.sumocfg")
    info = inspect_scenario("cologne1", data_dir="data")

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_q_model.json")
        controller, stats = train_q_learning_agent(
            scenario_name="cologne1",
            data_dir="data",
            episodes=1,
            max_steps_per_episode=60,
            save_path=model_path,
            lr=0.2,
            gamma=0.9,
            epsilon_start=0.5,
            epsilon_min=0.1,
            epsilon_decay=0.9,
        )

        assert isinstance(controller, QLearningController)
        assert len(stats) == 1
        assert "episode" in stats[0]
        assert "reward" in stats[0]
        assert "mean_queue" in stats[0]
        assert "throughput" in stats[0]
        assert len(controller.q_table) > 0
        assert os.path.isfile(model_path)
