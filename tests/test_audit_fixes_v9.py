"""Unit tests for ninth audit bug fixes (Issue 1 to Issue 8)."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from src.config import SimulationConfig, get_scenario_config
from src.controllers.base import IntersectionObservation
from src.controllers.dqn import DQNController
from src.controllers.fixed_time import FixedTimeController
from src.controllers.q_learning import QLearningController
from src.evaluation.evaluate_rl import load_checkpoint_env_params
from src.evaluation.experiment_runner import (
    ExperimentConfig,
    aggregate_benchmark_statistics,
    compute_run_fingerprint,
    run_full_benchmark_matrix,
)
from src.rl.discretization import StateDiscretizer
from src.rl.observations import ObservationBuilder
from src.runner import SimulationRunner


class TestAuditFixesV9(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_audit_v9_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # Issue 1 (P1): Resume logic does not skip differing fingerprints
    # -------------------------------------------------------------
    @patch("src.evaluation.experiment_runner.run_single_experiment")
    def test_issue1_resume_does_not_skip_when_fingerprint_differs(self, mock_run):
        """Issue 1: When resume=True, runner must NOT skip a task if its fingerprint differs from existing runs."""
        master_file = os.path.join(self.test_dir, "benchmark_runs.json")
        # Existing completed run has an old fingerprint
        old_record = {
            "scenario_name": "cologne1",
            "algorithm": "DQN",
            "seed": 42,
            "fingerprint": "old_fp_aaaa1111",
            "is_valid": True,
            "status": "SUCCESS",
            "tripinfo_mean_waiting_time": 12.5,
        }
        with open(master_file, "w", encoding="utf-8") as f:
            json.dump([old_record], f)

        mock_run.return_value = {
            "scenario_name": "cologne1",
            "algorithm": "DQN",
            "seed": 42,
            "fingerprint": "new_fp_bbbb2222",
            "is_valid": True,
            "status": "SUCCESS",
            "tripinfo_mean_waiting_time": 9.5,
        }

        # Run with matching scenario, algo, seed, but code/config yields a new fingerprint
        with patch("src.evaluation.experiment_runner.compute_run_fingerprint", return_value="new_fp_bbbb2222"):
            run_full_benchmark_matrix(
                scenarios=["cologne1"],
                algorithms=["DQN"],
                seeds=[42],
                output_dir=self.test_dir,
                resume=True,
            )

        # Must execute the experiment because new_fp_bbbb2222 was never run!
        mock_run.assert_called_once()

    # -------------------------------------------------------------
    # Issue 2 (P1): DQN multi-intersection target TLS restriction
    # -------------------------------------------------------------
    def test_issue2_dqn_target_tls_restriction(self):
        """Issue 2: DQNController with target_tls_id must only select actions for that intersection."""
        builder = ObservationBuilder(
            incoming_lanes=["lane_tls0_0", "lane_tls0_1"],
            num_green_phases=2,
            min_green=5.0,
        )
        controller = DQNController(
            observation_dim=builder.observation_dim,
            action_dim=2,
            obs_builder=builder,
            target_tls_id="tls_0",
            eval_mode=True,
        )

        obs_tls0 = IntersectionObservation(
            tls_id="tls_0",
            current_phase=0,
            time_in_phase=10.0,
            is_in_transition=False,
            green_phases=["GGrr", "rrGG"],
            controlled_links=[],
            in_lane_queues={"lane_tls0_0": 5.0, "lane_tls0_1": 2.0},
            out_lane_queues={},
            in_lane_vehicles={},
            out_lane_vehicles={},
            sim_time=10.0,
            step=10,
        )

        obs_tls1 = IntersectionObservation(
            tls_id="tls_1",
            current_phase=0,
            time_in_phase=10.0,
            is_in_transition=False,
            green_phases=["GGG", "rrr", "yyy"],
            controlled_links=[],
            in_lane_queues={"lane_tls1_0": 8.0, "lane_tls1_1": 4.0},
            out_lane_queues={},
            in_lane_vehicles={},
            out_lane_vehicles={},
            sim_time=10.0,
            step=10,
        )

        actions = controller.select_action({"tls_0": obs_tls0, "tls_1": obs_tls1})
        self.assertIn("tls_0", actions)
        self.assertNotIn("tls_1", actions)

    # -------------------------------------------------------------
    # Issue 3 (P2): Aggregation partitions cohorts without mixing
    # -------------------------------------------------------------
    def test_issue3_aggregate_benchmark_statistics_no_cohort_mixing(self):
        """Issue 3: aggregate_benchmark_statistics must partition cohorts and never blend distinct configurations."""
        run_a = {
            "scenario_name": "cologne1",
            "algorithm": "DQN",
            "seed": 42,
            "fingerprint": "fp10_aaaa1111",
            "cohort_id": "cohort_a",
            "is_valid": True,
            "status": "SUCCESS",
            "tripinfo_mean_waiting_time": 10.0,
            "mean_queue_length": 2.0,
        }
        run_b = {
            "scenario_name": "cologne1",
            "algorithm": "DQN",
            "seed": 43,
            "fingerprint": "fp100_bbbb2222",
            "cohort_id": "cohort_b",
            "is_valid": True,
            "status": "SUCCESS",
            "tripinfo_mean_waiting_time": 100.0,
            "mean_queue_length": 20.0,
        }

        stats = aggregate_benchmark_statistics([run_a, run_b])
        cologne_dqn = stats["cologne1"]["DQN"]

        # The latest cohort is under "DQN" (value 100.0)
        self.assertEqual(cologne_dqn["tripinfo_mean_waiting_time"]["mean"], 100.0)
        self.assertEqual(cologne_dqn["seeds"], [43])
        self.assertEqual(cologne_dqn["run_count"], 1)

        # Check that individual cohorts are also present and not blended into 55.0
        self.assertNotEqual(cologne_dqn["tripinfo_mean_waiting_time"]["mean"], 55.0)
        cohort_a_key = "DQN (fp: fp10_aaa)"
        self.assertIn(cohort_a_key, stats["cologne1"])
        self.assertEqual(stats["cologne1"][cohort_a_key]["tripinfo_mean_waiting_time"]["mean"], 10.0)
        self.assertEqual(stats["cologne1"][cohort_a_key]["seeds"], [42])

    # -------------------------------------------------------------
    # Issue 4 (P2): Q-Learning discretizer passed to agent
    # -------------------------------------------------------------
    def test_issue4_q_learning_controller_discretizer_configuration(self):
        """Issue 4: QLearningController must accept custom StateDiscretizer and save it in get_config."""
        custom_disc = StateDiscretizer(
            min_green=9.0,
            max_effective_green=35.0,
            serving_queue_thresholds=(2.0, 7.0),
            waiting_queue_thresholds=(2.0, 8.0, 20.0),
        )
        agent = QLearningController(
            num_actions=3,
            discretizer=custom_disc,
            seed=42,
            eval_mode=False,
        )
        cfg = agent.get_config()
        self.assertIn("discretizer_config", cfg)
        self.assertEqual(cfg["discretizer_config"]["min_green"], 9.0)
        self.assertEqual(cfg["discretizer_config"]["max_effective_green"], 35.0)
        self.assertEqual(tuple(cfg["discretizer_config"]["serving_thresholds"]), (2.0, 7.0))

    # -------------------------------------------------------------
    # Issue 5 (P2): DQNController.load restores observation scaling
    # -------------------------------------------------------------
    def test_issue5_dqn_load_restores_observation_scaling(self):
        """Issue 5: DQNController.load must restore max_queue_scale, max_time_scale, and min_green."""
        builder = ObservationBuilder(
            incoming_lanes=["lane_1", "lane_2"],
            num_green_phases=2,
            min_green=5.0,
            max_queue_scale=20.0,
            max_time_scale=60.0,
        )
        agent = DQNController(
            observation_dim=builder.observation_dim,
            action_dim=2,
            obs_builder=builder,
        )

        ckpt_path = os.path.join(self.test_dir, "custom_dqn.pt")
        # Save a checkpoint with custom normalization scales
        builder_custom = ObservationBuilder(
            incoming_lanes=["lane_1", "lane_2"],
            num_green_phases=2,
            min_green=8.0,
            max_queue_scale=99.0,
            max_time_scale=150.0,
        )
        agent_saved = DQNController(
            observation_dim=builder_custom.observation_dim,
            action_dim=2,
            obs_builder=builder_custom,
        )
        agent_saved.save(ckpt_path)

        # Load into initial agent (which had 20.0 and 60.0)
        agent.load(ckpt_path)
        self.assertEqual(agent.obs_builder.max_queue_scale, 99.0)
        self.assertEqual(agent.obs_builder.max_time_scale, 150.0)
        self.assertEqual(agent.obs_builder.min_green, 8.0)

    # -------------------------------------------------------------
    # Issue 6 (P2): RL evaluation synchronizes environment params
    # -------------------------------------------------------------
    def test_issue6_load_checkpoint_env_params(self):
        """Issue 6: load_checkpoint_env_params must inspect checkpoint and extract observation scales and thresholds."""
        # 1. DQN checkpoint
        dqn_ckpt_path = os.path.join(self.test_dir, "dqn_test.pt")
        builder = ObservationBuilder(
            incoming_lanes=["lane_0"],
            num_green_phases=2,
            min_green=6.0,
            max_queue_scale=85.0,
            max_time_scale=110.0,
        )
        dqn_agent = DQNController(
            observation_dim=builder.observation_dim,
            action_dim=2,
            obs_builder=builder,
        )
        dqn_agent.save(dqn_ckpt_path)

        dqn_params = load_checkpoint_env_params(dqn_ckpt_path)
        self.assertEqual(dqn_params.get("max_queue_scale"), 85.0)
        self.assertEqual(dqn_params.get("max_time_scale"), 110.0)
        self.assertEqual(dqn_params.get("min_green"), 6.0)

        # 2. Q-Learning checkpoint
        ql_ckpt_path = os.path.join(self.test_dir, "ql_test.json")
        custom_disc = StateDiscretizer(
            min_green=7.0,
            max_effective_green=30.0,
            serving_queue_thresholds=(3.0, 9.0),
            waiting_queue_thresholds=(2.0, 5.0, 12.0),
        )
        ql_agent = QLearningController(
            num_actions=2,
            discretizer=custom_disc,
        )
        ql_agent.save(ql_ckpt_path)

        ql_params = load_checkpoint_env_params(ql_ckpt_path)
        self.assertEqual(ql_params.get("min_green"), 7.0)
        self.assertIn("state_discretizer", ql_params)
        self.assertEqual(ql_params["state_discretizer"].serving_thresholds, (3.0, 9.0))

    # -------------------------------------------------------------
    # Issue 7 (P2): SimulationRunner original mode syncs phase stats
    # -------------------------------------------------------------
    def test_issue7_runner_original_mode_syncs_external_phase(self):
        """Issue 7: In control_mode='original', phase changes in SUMO must be synchronized via sync_external_phase."""
        cfg = get_scenario_config(
            dataset_name="cologne1",
            control_mode="original",
            step_length=1.0,
        )
        runner = SimulationRunner(cfg)
        tls_id = list(runner.tls_managers.keys())[0]
        mgr = runner.tls_managers[tls_id]

        mock_traci = MagicMock()
        mock_traci.simulation.getTime.side_effect = [1.0, 2.0, 3.0]
        mock_traci.simulation.getStartingTeleportNumber.return_value = 0
        mock_traci.simulation.getCollisions.return_value = []
        mock_traci.simulation.getCollidingVehiclesIDList.return_value = []
        mock_traci.simulation.getCollidingVehiclesNumber.return_value = 0

        # Simulate SUMO switching from phase 0 to phase 2
        mock_traci.trafficlight.getPhase.side_effect = [0, 2, 2]

        runner.is_running = True
        with patch("src.runner.traci", mock_traci):
            # Step 1: phase 0
            runner.step()
            self.assertEqual(mgr.get_current_phase(), 0)
            self.assertEqual(mgr.phase_switch_count, 0)

            # Step 2: phase 2 (switched by native SUMO)
            runner.step()
            self.assertEqual(mgr.get_current_phase(), 2)
            self.assertEqual(mgr.phase_switch_count, 1)

            # Step 3: still phase 2
            runner.step()
            self.assertEqual(mgr.phase_durations[2], 2.0)

    # -------------------------------------------------------------
    # Issue 8 (P2): Fixed-time controller respects offsets
    # -------------------------------------------------------------
    def test_issue8_fixed_time_offsets_applied(self):
        """Issue 8: FixedTimeController must extend initial cycle by offset and clear offset_remaining once reached."""
        controller = FixedTimeController(
            default_green_time=30.0,
            offsets={"tls_0": 15.0},
        )
        obs = IntersectionObservation(
            tls_id="tls_0",
            current_phase=0,
            time_in_phase=30.0,
            is_in_transition=False,
            green_phases=["GG", "rr"],
            controlled_links=[],
            in_lane_queues={},
            out_lane_queues={},
            in_lane_vehicles={},
            out_lane_vehicles={},
            sim_time=30.0,
            step=30,
        )

        # At elapsed 30s, base duration is 30s + offset 15s = 45s. Must hold phase 0!
        actions = controller.select_action({"tls_0": obs})
        self.assertEqual(actions["tls_0"], 0)
        self.assertEqual(controller._offset_remaining["tls_0"], 15.0)

        # At elapsed 45s, offset is satisfied. Must transition to phase 1 and reset offset_remaining!
        obs.time_in_phase = 45.0
        obs.sim_time = 45.0
        obs.step = 45
        actions = controller.select_action({"tls_0": obs})
        self.assertEqual(actions["tls_0"], 1)
        self.assertEqual(controller._offset_remaining["tls_0"], 0.0)


if __name__ == "__main__":
    unittest.main()
