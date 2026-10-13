"""Unit tests for eleventh audit bug fixes (Issue 1 to Issue 4)."""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from src.controllers.dqn import DQNController
from src.controllers.q_learning import QLearningController
from src.data_inspector import inspect_scenario
from src.evaluation.evaluate_rl import evaluate_agent, load_checkpoint_env_params
from src.evaluation.experiment_runner import aggregate_benchmark_statistics
from src.runner import extract_green_phases


class TestAuditFixesV11(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_audit_v11_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # Issue 1 (P1): Yellow clearance phases strictly excluded from green actions
    # -------------------------------------------------------------
    def test_issue1_no_yellow_phases_in_extract_green_phases(self):
        """Issue 1: extract_green_phases must strictly eliminate all yellow clearance phases
        even when their duration is >= 5 seconds.
        """
        from src.data_inspector import PhaseInfo
        for scenario in ["cologne1", "cologne3", "ingolstadt1", "grid4x4"]:
            info = inspect_scenario(scenario)
            total_yellow_found = 0
            for tl_id, phases in info.phases_per_tls.items():
                green_phases = extract_green_phases(scenario, tl_id, phases)
                for g in green_phases:
                    if any(c in ("y", "Y") for c in g):
                        total_yellow_found += 1
            self.assertEqual(
                total_yellow_found,
                0,
                f"Expected 0 yellow clearance phases in green_phases for {scenario}, found {total_yellow_found}",
            )

        # Specifically verify edge-case where a yellow clearance phase has duration >= 5.0s (e.g. '(5.0, rryyyyg)')
        synthetic_phases = [
            PhaseInfo(index=0, duration=30.0, state="rrGGGGg"),
            PhaseInfo(index=1, duration=5.0, state="rryyyyg"),
            PhaseInfo(index=2, duration=3.0, state="rrrryyy"),
            PhaseInfo(index=3, duration=30.0, state="rrrrGGG"),
            PhaseInfo(index=4, duration=3.0, state="yyyrrrr"),
            PhaseInfo(index=5, duration=30.0, state="GGGrrrr"),
        ]
        extracted = extract_green_phases("mock_scenario", "tls_1", synthetic_phases)
        self.assertNotIn("rryyyyg", extracted)
        self.assertNotIn("rrrryyy", extracted)
        self.assertNotIn("yyyrrrr", extracted)
        # Verify genuine green phases are preserved
        self.assertIn("rrGGGGg", extracted)
        self.assertIn("rrrrGGG", extracted)
        self.assertIn("GGGrrrr", extracted)

    # -------------------------------------------------------------
    # Issue 2 (P2): Independent evaluation restores custom transition parameters
    # -------------------------------------------------------------
    def test_issue2_evaluate_agent_and_checkpoint_custom_transitions(self):
        """Issue 2: evaluate_agent and load_checkpoint_env_params must restore and apply

        decision_interval, yellow_duration, all_red_duration, and min_green from checkpoint.
        """
        # 1. Test JSON checkpoint (Q-Learning)
        q_ckpt_path = os.path.join(self.test_dir, "q_model.json")
        q_data = {
            "config": {
                "decision_interval": 10.0,
                "yellow_duration": 7.0,
                "all_red_duration": 2.0,
                "min_green": 9.0,
            },
            "q_table": {},
        }
        with open(q_ckpt_path, "w", encoding="utf-8") as f:
            json.dump(q_data, f)

        params_q = load_checkpoint_env_params(q_ckpt_path)
        self.assertEqual(params_q["decision_interval"], 10.0)
        self.assertEqual(params_q["yellow_duration"], 7.0)
        self.assertEqual(params_q["all_red_duration"], 2.0)
        self.assertEqual(params_q["min_green"], 9.0)

        # 2. Test PT checkpoint (DQN)
        dqn_ckpt_path = os.path.join(self.test_dir, "dqn_model.pt")
        dqn_data = {
            "q_network_state_dict": {},
            "target_network_state_dict": {},
            "optimizer_state_dict": {},
            "config": {
                "decision_interval": 12.0,
                "yellow_duration": 6.0,
                "all_red_duration": 3.0,
                "min_green": 8.0,
            },
        }
        torch.save(dqn_data, dqn_ckpt_path)

        params_dqn = load_checkpoint_env_params(dqn_ckpt_path)
        self.assertEqual(params_dqn["decision_interval"], 12.0)
        self.assertEqual(params_dqn["yellow_duration"], 6.0)
        self.assertEqual(params_dqn["all_red_duration"], 3.0)
        self.assertEqual(params_dqn["min_green"], 8.0)

        # 3. Test evaluate_agent passing them to TrafficSignalRLEnv
        with patch("src.evaluation.evaluate_rl.TrafficSignalRLEnv") as mock_env_cls, \
             patch("src.evaluation.evaluate_rl.QLearningController") as mock_agent_cls:
            mock_env = MagicMock()
            mock_env.reset.return_value = (np.zeros(5), {"valid_actions": [0], "discrete_state": 0})
            mock_env.step.return_value = (np.zeros(5), 1.0, True, False, {"valid_actions": [0], "discrete_state": 0})
            mock_env.get_summary.return_value = {"mean_waiting_time": 15.0}
            mock_env_cls.return_value = mock_env

            mock_agent = MagicMock()
            mock_agent.select_action_for_state.return_value = 0
            mock_agent_cls.return_value = mock_agent

            evaluate_agent(
                scenario_name="ingolstadt1",
                algorithm="q_learning",
                model_path=q_ckpt_path,
                output_dir=self.test_dir,
            )

            _, env_kwargs = mock_env_cls.call_args
            self.assertEqual(env_kwargs["decision_interval"], 10.0)
            self.assertEqual(env_kwargs["yellow_duration"], 7.0)
            self.assertEqual(env_kwargs["all_red_duration"], 2.0)
            self.assertEqual(env_kwargs["min_green"], 9.0)

    # -------------------------------------------------------------
    # Issue 3 (P2): DQNController.load restores full training hyperparameters
    # -------------------------------------------------------------
    def test_issue3_dqn_load_restores_all_training_hyperparameters(self):
        """Issue 3: DQNController.load must restore batch_size, learning_starts,

        target_update_interval, buffer_size, max_grad_norm, and target_tls_id.
        """
        ckpt_path = os.path.join(self.test_dir, "dqn_custom_hparams.pt")
        # Create an initial agent with custom hyperparameters and save it
        agent_saved = DQNController(
            observation_dim=8,
            action_dim=2,
            batch_size=1,
            learning_starts=1,
            target_update_interval=7,
            buffer_size=5000,
            max_grad_norm=2.5,
            target_tls_id="tls_custom_target",
            decision_interval=10.0,
            yellow_duration=5.0,
            all_red_duration=2.0,
            min_green=8.0,
        )
        agent_saved.save(ckpt_path)

        # Create a fresh agent with default hyperparameters
        fresh_agent = DQNController(
            observation_dim=8,
            action_dim=2,
            batch_size=32,
            learning_starts=64,
            target_update_interval=100,
            buffer_size=20000,
            max_grad_norm=10.0,
        )
        self.assertEqual(fresh_agent.batch_size, 32)
        self.assertEqual(fresh_agent.learning_starts, 64)
        self.assertEqual(fresh_agent.target_update_interval, 100)

        # Load checkpoint into fresh agent
        fresh_agent.load(ckpt_path)

        # Hyperparameters must be fully restored!
        self.assertEqual(fresh_agent.batch_size, 1)
        self.assertEqual(fresh_agent.learning_starts, 1)
        self.assertEqual(fresh_agent.target_update_interval, 7)
        self.assertEqual(fresh_agent.buffer_size, 5000)
        self.assertEqual(fresh_agent.replay_buffer.capacity, 5000)
        self.assertEqual(fresh_agent.max_grad_norm, 2.5)
        self.assertEqual(fresh_agent.target_tls_id, "tls_custom_target")
        self.assertEqual(fresh_agent.decision_interval, 10.0)
        self.assertEqual(fresh_agent.yellow_duration, 5.0)
        self.assertEqual(fresh_agent.all_red_duration, 2.0)
        self.assertEqual(fresh_agent.min_green, 8.0)

    # -------------------------------------------------------------
    # Issue 4 (P2): aggregate_benchmark_statistics selects chronologically latest cohort
    # -------------------------------------------------------------
    def test_issue4_aggregate_benchmark_statistics_selects_chronologically_latest_cohort(self):
        """Issue 4: In history sequence A -> B -> A, primary cohort must be A (most recent), not B."""
        run_a1 = {
            "scenario_name": "cologne1",
            "algorithm": "DQN",
            "seed": 42,
            "cohort_id": "cohort_A",
            "is_valid": True,
            "status": "SUCCESS",
            "tripinfo_mean_waiting_time": 10.0,
        }
        run_b = {
            "scenario_name": "cologne1",
            "algorithm": "DQN",
            "seed": 42,
            "cohort_id": "cohort_B",
            "is_valid": True,
            "status": "SUCCESS",
            "tripinfo_mean_waiting_time": 50.0,
        }
        run_a2 = {
            "scenario_name": "cologne1",
            "algorithm": "DQN",
            "seed": 101,
            "cohort_id": "cohort_A",
            "is_valid": True,
            "status": "SUCCESS",
            "tripinfo_mean_waiting_time": 20.0,
        }

        # History is A -> B -> A
        stats = aggregate_benchmark_statistics([run_a1, run_b, run_a2])
        primary_dqn = stats["cologne1"]["DQN"]

        # Primary cohort must be A because A was run last!
        self.assertEqual(primary_dqn["cohort_id"], "cohort_A")
        self.assertEqual(primary_dqn["tripinfo_mean_waiting_time"]["mean"], 15.0)
        self.assertEqual(primary_dqn["seeds"], [42, 101])
        self.assertEqual(primary_dqn["run_count"], 2)


if __name__ == "__main__":
    unittest.main()
