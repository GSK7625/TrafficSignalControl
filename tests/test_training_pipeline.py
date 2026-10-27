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


def test_train_q_learning_with_and_without_libsumo():
    """Verify train_q_learning_agent executes with both libsumo and traci backends."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Test with TraCI (use_libsumo=False)
        model_traci = os.path.join(tmpdir, "model_traci.json")
        ctrl_traci, stats_traci = train_q_learning_agent(
            scenario_name="cologne1",
            data_dir="data",
            episodes=1,
            max_steps_per_episode=30,
            save_path=model_traci,
            use_libsumo=False,
            verbose=False,
        )
        assert len(stats_traci) == 1
        assert os.path.isfile(model_traci)

        # 2. Test with libsumo (use_libsumo=True)
        model_libsumo = os.path.join(tmpdir, "model_libsumo.json")
        ctrl_libsumo, stats_libsumo = train_q_learning_agent(
            scenario_name="cologne1",
            data_dir="data",
            episodes=1,
            max_steps_per_episode=30,
            save_path=model_libsumo,
            use_libsumo=True,
            verbose=False,
        )
        assert len(stats_libsumo) == 1
        assert os.path.isfile(model_libsumo)


def test_train_q_learning_resume_from_model():
    """Verify train_q_learning_agent can resume from an existing model and retain states/actions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        stage1_path = os.path.join(tmpdir, "stage1.json")
        stage2_path = os.path.join(tmpdir, "stage2.json")

        # Stage 1: train 1 episode
        ctrl1, stats1 = train_q_learning_agent(
            scenario_name="cologne1",
            data_dir="data",
            episodes=1,
            max_steps_per_episode=30,
            save_path=stage1_path,
            use_libsumo=True,
            verbose=False,
        )
        assert os.path.isfile(stage1_path)
        stage1_states = len(ctrl1.q_table)
        assert stage1_states > 0

        # Stage 2: resume with init_model
        ctrl2, stats2 = train_q_learning_agent(
            scenario_name="cologne1",
            data_dir="data",
            episodes=1,
            max_steps_per_episode=30,
            init_model=stage1_path,
            save_path=stage2_path,
            epsilon_start=0.2,
            epsilon_min=0.08,
            use_libsumo=True,
            verbose=False,
        )
        assert os.path.isfile(stage2_path)
        assert len(ctrl2.q_table) >= stage1_states
        assert "action_count" in stats2[0]


