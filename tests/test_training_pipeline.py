"""Unit tests for Q-Learning training pipeline."""

import os
import tempfile

from src.controllers.q_learning import QLearningController
from src.training.train_q_learning import train_q_learning_agent


def test_train_q_learning_single_episode_short():
    """Verify train_q_learning_agent executes short episode and learns non-empty Q-table."""
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
        _, stats_traci = train_q_learning_agent(
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
        _, stats_libsumo = train_q_learning_agent(
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
        ctrl1, _ = train_q_learning_agent(
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


def test_training_loop_updates_switch_actions(tmp_path):
    """Training with exploration must update switch actions (a != curr_phase) in Q-table."""
    save_file = str(tmp_path / "test_model.json")
    ctrl, stats = train_q_learning_agent(
        scenario_name="ingolstadt1",
        data_dir="data",
        episodes=1,
        max_steps_per_episode=100,
        save_path=save_file,
        lr=0.2,
        gamma=0.9,
        epsilon_start=1.0,
        epsilon_min=1.0,  # Force pure exploration
        min_green=5.0,
        decision_interval=5.0,
        verbose=False,
    )

    # Check non-zero updates where action != state[0]
    switch_updates = 0
    for state, q_vals in ctrl.q_table.items():
        curr_p = state[0]
        for a, val in enumerate(q_vals):
            if a != curr_p and val != 0.0:
                switch_updates += 1

    assert switch_updates > 0, "No switch action (a != curr_phase) was ever updated during training!"




def test_training_pipeline_multi_episode_seed_variation(tmp_path):
    """Verify train_q_learning_agent executes multi-episode training with seed variation."""
    save_file = str(tmp_path / "test_multi_seed_model.json")
    ctrl, stats = train_q_learning_agent(
        scenario_name="cologne1",
        data_dir="data",
        episodes=2,
        max_steps_per_episode=30,
        save_path=save_file,
        seeds=[101, 202],
        use_libsumo=True,
        verbose=False,
    )
    assert len(stats) == 2
    assert stats[0]["episode"] == 1
    assert stats[1]["episode"] == 2
    assert os.path.isfile(save_file)
