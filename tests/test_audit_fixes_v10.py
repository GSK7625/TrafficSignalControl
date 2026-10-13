"""Unit tests for tenth audit bug fixes (Issue 1 to Issue 6)."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import yaml

from src.controllers.base import IntersectionObservation
from src.controllers.dqn import DQNController
from src.evaluation.experiment_runner import (
    ExperimentConfig,
    aggregate_benchmark_statistics,
    compute_cohort_fingerprint,
    compute_run_fingerprint,
    run_single_experiment,
)
from src.phase_manager import PhaseState, SafePhaseManager
from src.runner import SimulationRunner
from src.config import SimulationConfig, get_scenario_config
import numpy as np


class TestAuditFixesV10(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_audit_v10_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # Issue 1 (P1): Multi-seed statistics aggregation
    # -------------------------------------------------------------
    def test_issue1_multi_seed_statistics_aggregation(self):
        """Issue 1: Runs across multiple seeds for the same configuration must aggregate

        into a single cohort with correct mean, std, min, max, and run_count.
        """
        cfg_seed42 = ExperimentConfig(
            scenario_name="ingolstadt1",
            algorithm="DQN",
            seed=42,
            training_seed=42,
        )
        cfg_seed101 = ExperimentConfig(
            scenario_name="ingolstadt1",
            algorithm="DQN",
            seed=101,
            training_seed=42,
        )
        cfg_seed2024 = ExperimentConfig(
            scenario_name="ingolstadt1",
            algorithm="DQN",
            seed=2024,
            training_seed=42,
        )

        fp42 = compute_run_fingerprint(cfg_seed42)
        fp101 = compute_run_fingerprint(cfg_seed101)
        fp2024 = compute_run_fingerprint(cfg_seed2024)
        # Fingerprints must differ across seeds for resume caching
        self.assertNotEqual(fp42, fp101)
        self.assertNotEqual(fp101, fp2024)

        cohort_fp42 = compute_cohort_fingerprint(cfg_seed42)
        cohort_fp101 = compute_cohort_fingerprint(cfg_seed101)
        cohort_fp2024 = compute_cohort_fingerprint(cfg_seed2024)
        # Cohort fingerprint must be identical across seeds!
        self.assertEqual(cohort_fp42, cohort_fp101)
        self.assertEqual(cohort_fp101, cohort_fp2024)

        run_42 = {
            "scenario_name": "ingolstadt1",
            "algorithm": "DQN",
            "seed": 42,
            "fingerprint": fp42,
            "cohort_id": cohort_fp42,
            "is_valid": True,
            "status": "SUCCESS",
            "tripinfo_mean_waiting_time": 10.0,
            "mean_queue_length": 2.0,
        }
        run_101 = {
            "scenario_name": "ingolstadt1",
            "algorithm": "DQN",
            "seed": 101,
            "fingerprint": fp101,
            "cohort_id": cohort_fp101,
            "is_valid": True,
            "status": "SUCCESS",
            "tripinfo_mean_waiting_time": 20.0,
            "mean_queue_length": 4.0,
        }
        run_2024 = {
            "scenario_name": "ingolstadt1",
            "algorithm": "DQN",
            "seed": 2024,
            "fingerprint": fp2024,
            "cohort_id": cohort_fp2024,
            "is_valid": True,
            "status": "SUCCESS",
            "tripinfo_mean_waiting_time": 30.0,
            "mean_queue_length": 6.0,
        }

        stats = aggregate_benchmark_statistics([run_42, run_101, run_2024])
        ingolstadt_dqn = stats["ingolstadt1"]["DQN"]

        # All 3 seeds must be united under the single primary algorithm entry
        self.assertEqual(ingolstadt_dqn["run_count"], 3)
        self.assertEqual(ingolstadt_dqn["seeds"], [42, 101, 2024])
        # Mean of [10, 20, 30] is 20.0
        self.assertEqual(ingolstadt_dqn["tripinfo_mean_waiting_time"]["mean"], 20.0)
        # Sample std with ddof=1 is 10.0
        self.assertEqual(ingolstadt_dqn["tripinfo_mean_waiting_time"]["std"], 10.0)
        self.assertEqual(ingolstadt_dqn["tripinfo_mean_waiting_time"]["min"], 10.0)
        self.assertEqual(ingolstadt_dqn["tripinfo_mean_waiting_time"]["max"], 30.0)
        # No artificial multiple cohorts split
        self.assertFalse(ingolstadt_dqn.get("has_multiple_cohorts", False))

    # -------------------------------------------------------------
    # Issue 2 (P2): Circular import of RL environment in fresh process
    # -------------------------------------------------------------
    def test_issue2_no_circular_import_in_fresh_python_process(self):
        """Issue 2: Directly importing TrafficSignalRLEnv in a fresh process must not raise ImportError."""
        cmd = [
            sys.executable,
            "-c",
            "from src.rl.environment import TrafficSignalRLEnv; "
            "from src.controllers import QLearningController, DQNController; "
            "print('SUCCESS')",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(
            proc.returncode,
            0,
            f"Import failed with error:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}",
        )
        self.assertIn("SUCCESS", proc.stdout)

    # -------------------------------------------------------------
    # Issue 3 (P2): RL Environment & Runner cleanup on exception
    # -------------------------------------------------------------
    @patch("src.evaluation.experiment_runner.TrafficSignalRLEnv")
    def test_issue3_env_closed_when_reset_fails(self, mock_env_cls):
        """Issue 3: If an exception occurs during experiment execution, env.close() must be called in finally."""
        mock_env = MagicMock()
        mock_env.observation_dim = 10
        mock_env.num_actions = 2
        mock_env.obs_builder = MagicMock()
        mock_env.reset.side_effect = RuntimeError("Simulated SUMO startup crash")
        mock_env_cls.return_value = mock_env

        cfg = ExperimentConfig(
            scenario_name="ingolstadt1",
            algorithm="DQN",
            seed=42,
            output_dir=self.test_dir,
        )

        record = run_single_experiment(cfg)
        self.assertEqual(record["status"], "FAILED")
        self.assertEqual(record["execution_status"], "FAILED")
        self.assertFalse(record["is_valid"])
        # Crucial check: env.close() MUST have been called despite the exception!
        mock_env.close.assert_called_once()

    # -------------------------------------------------------------
    # Issue 4 (P2): SafePhaseManager Original SUMO phase sync
    # -------------------------------------------------------------
    def test_issue4_sync_external_phase_handles_clearance_and_green(self):
        """Issue 4: sync_external_phase must map SUMO signal states to valid green indices

        and not assign out-of-bounds indices during yellow clearance.
        """
        green_phases = ["GGrr", "rrGG", "GrGr"]
        mgr = SafePhaseManager(
            tls_id="tls_0",
            green_phases=green_phases,
            initial_phase=0,
        )
        self.assertEqual(mgr.get_current_phase(), 0)
        self.assertEqual(mgr.get_current_state(), "GGrr")

        # 1. External controller enters yellow clearance: state "yyrr", raw SUMO phase index 4
        mgr.sync_external_phase(phase_index=4, signal_state="yyrr", step_duration=1.0)
        # Current phase must NOT become 4! Must remain valid green index
        self.assertEqual(mgr.get_current_phase(), 0)
        # Signal string must reflect actual state
        self.assertEqual(mgr.get_current_state(), "yyrr")
        # Must be in transition
        self.assertTrue(mgr.is_in_transition())
        # Yellow duration should not be added to green phase durations
        self.assertEqual(mgr.phase_durations[0], 0.0)
        self.assertEqual(mgr.phase_switch_count, 0)

        # 2. External controller enters next green: state "rrGG", raw SUMO phase index 2
        mgr.sync_external_phase(phase_index=2, signal_state="rrGG", step_duration=1.0)
        self.assertEqual(mgr.get_current_phase(), 1)
        self.assertEqual(mgr.get_current_state(), "rrGG")
        self.assertFalse(mgr.is_in_transition())
        self.assertEqual(mgr.phase_durations[1], 1.0)
        self.assertEqual(mgr.phase_switch_count, 1)

    # -------------------------------------------------------------
    # Issue 5 (P2): run(controller=...) updates controlled TLS IDs
    # -------------------------------------------------------------
    def test_issue5_run_with_controller_updates_controlled_tls_ids(self):
        """Issue 5: Calling runner.run(controller=...) must update _controlled_tls_ids

        to target only the controller's target_tls_id.
        """
        sim_cfg = get_scenario_config(
            dataset_name="ingolstadt1",
            control_mode="original",
            max_steps=10,
        )
        runner = SimulationRunner(sim_cfg)
        # Without controller at init, runner is in original mode with empty controlled TLS
        self.assertEqual(runner._controlled_tls_ids, set())

        # Create mock targeted controller
        mock_controller = MagicMock()
        mock_controller.target_tls_id = "tls_target_only"

        def fake_step():
            runner.current_step += 1

        with patch.object(runner, "start"), \
             patch.object(runner, "step", side_effect=fake_step), \
             patch.object(runner, "close"):
            runner.run(num_steps=1, controller=mock_controller)

        # Must restrict control strictly to the target TLS
        self.assertEqual(runner._controlled_tls_ids, {"tls_target_only"})
        self.assertEqual(runner.control_mode, "agent")
        self.assertTrue(runner._active_control_engaged)

    # -------------------------------------------------------------
    # Issue 6 (P2): CLI arguments override YAML configuration in training
    # -------------------------------------------------------------
    def test_issue6_cli_arguments_override_yaml_in_training(self):
        """Issue 6: In train_q_learning and train_dqn, user-specified CLI arguments

        must override values defined in YAML configs.
        """
        from src.training.train_q_learning import train_q_learning
        from src.training.train_dqn import train_dqn

        # Create temporary YAML config setting episodes: 0, lr: 0.001
        yaml_path = os.path.join(self.test_dir, "config_test.yaml")
        yaml_content = {
            "hyperparameters": {
                "num_episodes": 0,
                "learning_rate": 0.001,
                "seed": 999,
            },
            "decision_interval": 10.0,
            "min_green": 7.0,
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f)

        # 1. Test Q-Learning precedence
        with patch("src.training.train_q_learning.TrafficSignalRLEnv") as mock_env_cls, \
             patch("src.training.train_q_learning.QLearningController") as mock_agent_cls:
            mock_env = MagicMock()
            mock_env.num_actions = 2
            mock_env.reset.return_value = (np.zeros(5), {"valid_actions": [0, 1], "discrete_state": 0})
            mock_env.step.return_value = (np.zeros(5), 1.0, True, False, {"valid_actions": [0, 1], "discrete_state": 0})
            mock_env.get_summary.return_value = {}
            mock_env_cls.return_value = mock_env
            mock_agent = MagicMock()
            mock_agent.q_table = {}
            mock_agent.total_updates = 0
            mock_agent.epsilon = 0.5
            mock_agent.select_action.return_value = 0
            mock_agent_cls.return_value = mock_agent

            # User explicitly passes num_episodes=2, seed=42 on CLI; YAML has episodes=0, seed=999
            train_q_learning(
                scenario_name="ingolstadt1",
                num_episodes=2,
                seed=42,
                config_path=yaml_path,
                model_dir=self.test_dir,
                log_dir=self.test_dir,
            )

            # Check that TrafficSignalRLEnv received seed=42 (from CLI, overriding 999)
            _, first_env_kwargs = mock_env_cls.call_args_list[0]
            self.assertEqual(first_env_kwargs["seed"], 42)
            # Check that QLearningController received lr=0.001 (from YAML since CLI was None)
            _, agent_kwargs = mock_agent_cls.call_args
            self.assertEqual(agent_kwargs["learning_rate"], 0.001)

            # Check saved training log has num_episodes == 2 (CLI won!)
            log_path = os.path.join(self.test_dir, "q_learning_ingolstadt1_log.json")
            with open(log_path, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            self.assertEqual(log_data["num_episodes"], 2)
            self.assertEqual(log_data["seed"], 42)

        # 2. Test DQN precedence
        with patch("src.training.train_dqn.TrafficSignalRLEnv") as mock_dqn_env_cls, \
             patch("src.training.train_dqn.DQNController") as mock_dqn_agent_cls:
            mock_env = MagicMock()
            mock_env.observation_dim = 10
            mock_env.action_dim = 2
            mock_env.reset.return_value = (np.zeros(10), {"valid_actions": [0, 1], "action_mask": np.array([1, 1])})
            mock_env.step.return_value = (np.zeros(10), 1.0, True, False, {"valid_actions": [0, 1], "action_mask": np.array([1, 1])})
            mock_env.get_summary.return_value = {}
            mock_dqn_env_cls.return_value = mock_env
            mock_agent = MagicMock()
            mock_agent.total_updates = 0
            mock_agent.epsilon = 0.5
            mock_agent.replay_buffer = MagicMock()
            mock_agent.replay_buffer.__len__.return_value = 100
            mock_agent.select_action.return_value = 0
            mock_dqn_agent_cls.return_value = mock_agent

            # User explicitly passes num_episodes=3, lr=0.05 on CLI; YAML has episodes=0, lr=0.001
            train_dqn(
                scenario_name="ingolstadt1",
                num_episodes=3,
                learning_rate=0.05,
                config_path=yaml_path,
                model_dir=self.test_dir,
                log_dir=self.test_dir,
            )

            # Check that TrafficSignalRLEnv received decision_interval=10.0 (from YAML since CLI was None)
            _, env_kwargs = mock_dqn_env_cls.call_args
            self.assertEqual(env_kwargs["decision_interval"], 10.0)

            # Check saved training log has num_episodes == 3, lr == 0.05 (CLI won!)
            dqn_log_path = os.path.join(self.test_dir, "dqn_ingolstadt1_log.json")
            with open(dqn_log_path, "r", encoding="utf-8") as f:
                dqn_log_data = json.load(f)
            self.assertEqual(dqn_log_data["num_episodes"], 3)
            self.assertEqual(dqn_log_data["hyperparameters"]["learning_rate"], 0.05)


if __name__ == "__main__":
    unittest.main()
