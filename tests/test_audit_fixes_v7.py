"""Unit tests for seventh audit bug fixes (TDD Red -> Green phase)."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

from src.config import get_scenario_config, SimulationConfig
from src.runner import SimulationRunner
from src.rl.environment import TrafficSignalRLEnv as TrafficSignalEnv


class TestAuditFixesV7(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_audit_v7_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # Task 1: F05 & F01 - decision_interval validation & pacing
    # -------------------------------------------------------------
    def test_environment_decision_interval_validation_rejects_non_positive(self):
        """F05: TrafficSignalEnv must reject decision_interval <= 0 or NaN/Inf."""
        with self.assertRaises(ValueError):
            TrafficSignalEnv(scenario_name="ingolstadt1", decision_interval=0.0)
        with self.assertRaises(ValueError):
            TrafficSignalEnv(scenario_name="ingolstadt1", decision_interval=-1.0)
        with self.assertRaises(ValueError):
            TrafficSignalEnv(scenario_name="ingolstadt1", decision_interval=float("nan"))

    def test_runner_decision_interval_pacing(self):
        """F01: SimulationRunner must pace controller action selection by decision_interval."""
        cfg = get_scenario_config("ingolstadt1", output_dir=self.test_dir, decision_interval=5.0)
        self.assertEqual(cfg.decision_interval, 5.0)

        mock_ctrl = MagicMock()
        mock_ctrl.select_action.return_value = {cfg.dataset_name: 0}
        runner = SimulationRunner(cfg, controller=mock_ctrl)
        self.assertEqual(runner.decision_interval, 5.0)

        # Mock traci simulation
        mock_traci = MagicMock()
        mock_traci.simulation.getTime.side_effect = lambda: 57600.0 + runner.current_step
        mock_traci.simulation.getDepartedIDList.return_value = []
        mock_traci.simulation.getArrivedIDList.return_value = []
        mock_traci.vehicle.getIDList.return_value = []
        mock_traci.lane.getLastStepHaltingNumber.return_value = 0
        mock_traci.lane.getLastStepVehicleNumber.return_value = 0
        mock_traci.trafficlight.getControlledLinks.return_value = []

        import src.runner as runner_module
        orig_traci = runner_module.traci
        runner_module.traci = mock_traci
        try:
            runner.start()
            # Step 10 times (10 seconds with step_length=1.0)
            # With decision_interval=5.0, select_action should only be called twice (at step 0 and step 5)
            for _ in range(10):
                runner.step()
            runner.close()

            self.assertEqual(
                mock_ctrl.select_action.call_count,
                2,
                f"Controller should be called exactly 2 times in 10s with 5s cadence, got {mock_ctrl.select_action.call_count}",
            )
        finally:
            runner_module.traci = orig_traci

    # -------------------------------------------------------------
    # Task 2: F04 - TraCI error handling in observation extraction
    # -------------------------------------------------------------
    def test_runner_get_observation_records_traci_error(self):
        """F04: TraCI errors in get_observation or getControlledLinks must record error and mark run invalid."""
        from src.runner import ControlledLink
        cfg = get_scenario_config("ingolstadt1", output_dir=self.test_dir)
        runner = SimulationRunner(cfg)
        tl_id = runner.scenario_info.tls_ids[0]
        runner.tls_controlled_links = {
            tl_id: [ControlledLink(signal_index=0, from_lane="bad_in_lane", to_lane="bad_out_lane")]
        }

        mock_traci = MagicMock()
        mock_traci.lane.getLastStepHaltingNumber.side_effect = RuntimeError("TraCI link broken")
        mock_traci.lane.getLastStepVehicleNumber.side_effect = RuntimeError("TraCI link broken")

        import src.runner as runner_module
        orig_traci = runner_module.traci
        runner_module.traci = mock_traci
        try:
            obs = runner.get_observation()
            self.assertIn(tl_id, obs)
            # Run must be flagged as invalid with recorded errors
            self.assertFalse(runner.metrics_collector.is_valid)
            self.assertGreater(runner.metrics_collector.measurement_errors, 0)
            self.assertTrue(
                any("bad_in_lane" in msg for msg in runner.metrics_collector.error_messages)
            )
        finally:
            runner_module.traci = orig_traci

    # -------------------------------------------------------------
    # Task 3: F02 & F03 - System Delay, Safety, and Classification
    # -------------------------------------------------------------
    def test_validate_run_metrics_classifies_safety_and_service(self):
        """F03: validate_run_metrics must classify safety incidents and service completion."""
        from src.evaluation.experiment_runner import validate_run_metrics

        # 1. Run with collisions
        summary_coll = {
            "mean_waiting_time": 13.57,
            "total_completed_vehicles": 1997,
            "total_steps": 3600,
            "collision_count": 4,
            "teleport_count": 4,
        }
        is_valid, anomalies, flags = validate_run_metrics(summary_coll, total_demand=2015, return_flags=True)
        self.assertTrue(is_valid)
        self.assertEqual(flags["safety_status"], "INCIDENTS_DETECTED")
        self.assertEqual(flags["service_status"], "COMPLETED")
        self.assertIn("4 collisions and 4 teleports detected", anomalies[0])

        # 2. Run with starvation (DQN Ingolstadt1: 1118/1716 = 65.15%)
        summary_starved = {
            "mean_waiting_time": 0.67,
            "total_completed_vehicles": 1118,
            "total_steps": 3600,
            "collision_count": 0,
            "teleport_count": 0,
        }
        is_valid, anomalies, flags = validate_run_metrics(summary_starved, total_demand=1716, return_flags=True)
        self.assertTrue(is_valid)
        self.assertEqual(flags["safety_status"], "SAFE")
        self.assertEqual(flags["service_status"], "PARTIAL")
        self.assertTrue(any("Completion rate 65.2%" in a or "65.1%" in a for a in anomalies))

    def test_metrics_collector_tracks_pending_and_system_delay(self):
        """F02: SimulationMetricsCollector must track pending vehicles and system total delay."""
        from src.metrics import SimulationMetricsCollector

        collector = SimulationMetricsCollector(scenario_name="ingolstadt1")
        mock_traci = MagicMock()
        mock_traci.simulation.getTime.return_value = 1.0
        mock_traci.simulation.getDepartedIDList.return_value = ["v1"]
        mock_traci.simulation.getArrivedIDList.return_value = []
        mock_traci.vehicle.getIDList.return_value = ["v1"]
        mock_traci.vehicle.getAccumulatedWaitingTime.return_value = 5.0
        mock_traci.vehicle.getTimeLoss.return_value = 5.0
        mock_traci.simulation.getPendingVehicles.return_value = ["v2", "v3"]
        mock_traci.lane.getLastStepHaltingNumber.return_value = 1

        collector.record_step(step=1, sim_time=1.0, controlled_lanes=["l1"], traci_module=mock_traci, step_length=1.0)
        summary = collector.get_summary()

        self.assertIn("system_total_delay", summary)
        self.assertIn("completed_mean_waiting_time", summary)
        self.assertIn("system_mean_delay", summary)
        self.assertIn("pending_insertion_vehicles", summary)
        self.assertEqual(summary["pending_insertion_vehicles"], 2)
        # Running wait (5.0) + pending backlog delay (2 vehicles * 1s = 2.0s) = 7.0s
        self.assertGreaterEqual(summary["system_total_delay"], 7.0)

    # -------------------------------------------------------------
    # Task 4: F07 - RL Environment Artifact Export
    # -------------------------------------------------------------
    def test_rl_environment_exports_artifacts(self):
        """F07: TrafficSignalEnv must export step_metrics.csv, summary.json, and phase metadata."""
        env = TrafficSignalEnv(scenario_name="ingolstadt1", output_dir=self.test_dir, decision_interval=5.0)
        # Mock runner execution and close
        env.runner.tls_controlled_links = {}
        env.runner.metrics_collector.record_step(
            step=1, sim_time=57601.0, controlled_lanes=[], traci_module=MagicMock(), step_length=1.0
        )
        summary = env._finalize_episode()

        self.assertIn("phase_change_count", summary)
        self.assertIn("phase_durations_per_tls", summary)
        self.assertIn("output_csv", summary)
        self.assertTrue(os.path.isfile(summary["output_csv"]), f"CSV must exist at {summary.get('output_csv')}")
        self.assertTrue(os.path.isfile(summary["output_json"]), f"JSON must exist at {summary.get('output_json')}")

    # -------------------------------------------------------------
    # Task 5: F12 - Terminated vs Truncated in DQN
    # -------------------------------------------------------------
    def test_dqn_target_preserves_bootstrap_on_truncation(self):
        """F12: DQN target calculation must preserve bootstrap value when truncated=True but terminated=False."""
        import numpy as np
        import torch
        from src.controllers.dqn import DQNController

        controller = DQNController(
            observation_dim=4,
            action_dim=2,
            batch_size=1,
            learning_starts=1,
            gamma=0.9,
            eval_mode=False,
        )

        state = np.zeros(4, dtype=np.float32)
        next_state = np.ones(4, dtype=np.float32)

        # Push transition where time limit expired (truncated=True, terminated=False)
        controller.replay_buffer.push(
            state=state,
            action=0,
            reward=10.0,
            next_state=next_state,
            done=True,
            valid_next_mask=np.array([True, True]),
            terminated=False,
            truncated=True,
        )

        # Sample and verify target computation
        states, actions, rewards, next_states, dones, next_masks, terminated_flags = controller.replay_buffer.sample(
            batch_size=1, device=controller.device, return_terminated=True
        )
        self.assertEqual(terminated_flags.item(), 0.0, "Terminated flag must be 0 for truncation")

    # -------------------------------------------------------------
    # Task 6: F06 & F14 - Fingerprint Resume, Sample ddof=1, Frozen Q-table
    # -------------------------------------------------------------
    def test_run_fingerprint_changes_when_config_or_checkpoint_changes(self):
        """F06: compute_run_fingerprint must be sensitive to config changes, seeds, and checkpoint weights."""
        from src.evaluation.experiment_runner import compute_run_fingerprint, ExperimentConfig

        cfg1 = ExperimentConfig(scenario_name="ingolstadt1", algorithm="DQN", seed=42)
        cfg2 = ExperimentConfig(scenario_name="ingolstadt1", algorithm="DQN", seed=42, decision_interval=1.0)
        cfg3 = ExperimentConfig(scenario_name="ingolstadt1", algorithm="DQN", seed=101)

        fp1 = compute_run_fingerprint(cfg1)
        fp2 = compute_run_fingerprint(cfg2)
        fp3 = compute_run_fingerprint(cfg3)

        self.assertNotEqual(fp1, fp2, "Fingerprint must change when decision_interval changes")
        self.assertNotEqual(fp1, fp3, "Fingerprint must change when seed changes")

    def test_q_table_not_mutated_in_eval_mode(self):
        """F14: Q-learning controller must not mutate q_table for unseen states in eval_mode."""
        from src.controllers.q_learning import QLearningController

        ctrl = QLearningController(num_actions=3, eval_mode=True)
        unseen_state = (0, 1, 2, 0)
        self.assertNotIn(unseen_state, ctrl.q_table)

        # Select action in eval mode
        action = ctrl.select_action_for_state(unseen_state)
        self.assertIn(action, [0, 1, 2])
        # Q-table must still be clean and not have unseen_state permanently added
        self.assertNotIn(unseen_state, ctrl.q_table)

    def test_aggregate_benchmark_statistics_uses_sample_std(self):
        """F06: aggregate_benchmark_statistics must use sample standard deviation (ddof=1)."""
        from src.evaluation.experiment_runner import aggregate_benchmark_statistics

        # Mock two runs with waiting times 10.0 and 20.0
        # Population std would be 5.0, sample std (ddof=1) is 7.07
        runs = [
            {
                "scenario_name": "ingolstadt1",
                "algorithm": "Fixed-Time",
                "seed": 42,
                "is_valid": True,
                "tripinfo_mean_waiting_time": 10.0,
            },
            {
                "scenario_name": "ingolstadt1",
                "algorithm": "Fixed-Time",
                "seed": 101,
                "is_valid": True,
                "tripinfo_mean_waiting_time": 20.0,
            },
        ]
        stats = aggregate_benchmark_statistics(runs, baseline_algo="Fixed-Time")
        mean_wait_stat = stats["ingolstadt1"]["Fixed-Time"]["tripinfo_mean_waiting_time"]
        self.assertEqual(mean_wait_stat["mean"], 15.0)
        self.assertEqual(mean_wait_stat["std"], 7.07, "Must use sample std ddof=1 (7.07 instead of 5.00)")

    # -------------------------------------------------------------
    # Task 7: F13 & F08 - YAML Loader and Config Integration
    # -------------------------------------------------------------
    def test_yaml_config_loader(self):
        """F13: load_yaml_config must load and parse YAML files correctly."""
        from src.config import load_yaml_config

        cfg_dqn = load_yaml_config("configs/dqn.yaml")
        self.assertEqual(cfg_dqn["algorithm"], "DQN")
        self.assertEqual(cfg_dqn["hyperparameters"]["learning_rate"], 0.001)

        cfg_ql = load_yaml_config("configs/q_learning.yaml")
        self.assertEqual(cfg_ql["algorithm"], "QLearning")
        self.assertEqual(cfg_ql["hyperparameters"]["learning_rate"], 0.1)

        with self.assertRaises(FileNotFoundError):
            load_yaml_config("configs/non_existent.yaml")

    def test_requirements_and_readme_exist(self):
        """F13: requirements.txt and README.md must exist in root workspace."""
        self.assertTrue(os.path.isfile("requirements.txt"), "requirements.txt must exist at root")
        self.assertTrue(os.path.isfile("README.md"), "README.md must exist at root")
        with open("requirements.txt", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("torch", content)
        self.assertIn("pyyaml", content)
        self.assertIn("pytest", content)
