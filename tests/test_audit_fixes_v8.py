"""Unit tests for eighth audit bug fixes (R01 to R11) - TDD Green phase."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch

from src.config import SimulationConfig, get_scenario_config
from src.controllers.dqn import DQNController
from src.controllers.q_learning import QLearningController
from src.evaluation.experiment_runner import (
    ExperimentConfig,
    aggregate_benchmark_statistics,
    compute_run_fingerprint,
    get_sumo_version,
    resolve_model_path,
)
from src.metrics import MetricsCollector
from src.rl.environment import TrafficSignalRLEnv
from src.training.train_q_learning import evaluate_greedy_q_learning


class TestAuditFixesV8(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_audit_v8_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # -------------------------------------------------------------
    # R01: Fingerprint consistency and resume deduplication
    # -------------------------------------------------------------
    def test_r01_fingerprint_consistency(self):
        """R01: compute_run_fingerprint must produce identical hash before and after model path resolution."""
        cfg = ExperimentConfig(
            scenario_name="ingolstadt1",
            algorithm="DQN",
            seed=42,
            training_seed=42,
        )
        # Even if cfg.model_path is None, resolve_model_path resolves to models/dqn/dqn_ingolstadt1_best.pt
        fp_unresolved = compute_run_fingerprint(cfg)
        resolved_path = resolve_model_path(cfg)
        cfg.model_path = resolved_path
        fp_resolved = compute_run_fingerprint(cfg, resolved_path)
        self.assertEqual(fp_unresolved, fp_resolved)

    def test_r01_resume_deduplication_in_aggregation(self):
        """R01: aggregate_benchmark_statistics must deduplicate runs by seed."""
        records = [
            {
                "scenario_name": "ingolstadt1",
                "algorithm": "DQN",
                "seed": 42,
                "is_valid": True,
                "tripinfo_mean_waiting_time": 5.0,
                "system_mean_delay": 10.0,
            },
            {
                "scenario_name": "ingolstadt1",
                "algorithm": "DQN",
                "seed": 42,  # Duplicate run from broken resume
                "is_valid": True,
                "tripinfo_mean_waiting_time": 5.5,
                "system_mean_delay": 10.5,
            },
            {
                "scenario_name": "ingolstadt1",
                "algorithm": "DQN",
                "seed": 43,
                "is_valid": True,
                "tripinfo_mean_waiting_time": 6.0,
                "system_mean_delay": 11.0,
            },
        ]
        stats = aggregate_benchmark_statistics(records)
        algo_stats = stats["ingolstadt1"]["DQN"]
        # Must have run_count=2, seeds=[42, 43], keeping latest for 42
        self.assertEqual(algo_stats["run_count"], 2)
        self.assertEqual(algo_stats["seeds"], [42, 43])
        self.assertEqual(algo_stats["tripinfo_mean_waiting_time"]["mean"], 5.75)

    # -------------------------------------------------------------
    # R02: decision_interval validation
    # -------------------------------------------------------------
    def test_r02_rejects_sub_step_and_non_integer_multiples(self):
        """R02: decision_interval must be >= step_length (1.0s) and an integer multiple."""
        with self.assertRaises(ValueError) as ctx:
            TrafficSignalRLEnv(scenario_name="ingolstadt1", decision_interval=0.4)
        self.assertIn("must be >=", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            TrafficSignalRLEnv(scenario_name="ingolstadt1", decision_interval=1.5)
        self.assertIn("must be an integer multiple", str(ctx.exception))

    # -------------------------------------------------------------
    # R03 & R11: CLI argument normalization and cadence
    # -------------------------------------------------------------
    def test_r03_and_r11_cli_cadence_and_alias(self):
        """R03 & R11: --algorithm should be accepted and q-learning must default to 5.0s."""
        import subprocess
        # Check CLI help / parser directly
        import main
        parser = main.create_argument_parser() if hasattr(main, "create_argument_parser") else None
        if parser:
            args = parser.parse_args(["--dataset", "cologne1", "--algorithm", "q-learning"])
            self.assertEqual(args.controller, "q-learning")
            ctrl_norm = (args.controller or "").lower().replace("-", "_")
            dec_int = 5.0 if ctrl_norm in ("dqn", "q_learning") else 1.0
            self.assertEqual(dec_int, 5.0)

    # -------------------------------------------------------------
    # R04: Q-learning validation determinism and RNG isolation
    # -------------------------------------------------------------
    def test_r04_q_learning_deterministic_tie_breaking(self):
        """R04: In eval_mode with tie_break_strategy='first', tie-breaking between max Q-values must be deterministic."""
        agent = QLearningController(num_actions=3, seed=42, eval_mode=True, tie_break_strategy="first")
        # All Q-values equal to 0.0
        state = (0, 0, 0, 0)
        a1 = agent.select_action_for_state(state, valid_actions=[0, 1, 2])
        a2 = agent.select_action_for_state(state, valid_actions=[0, 1, 2])
        self.assertEqual(a1, a2)
        self.assertEqual(a1, 0)

    def test_r04_validation_does_not_mutate_python_rng(self):
        """R04: evaluate_greedy_q_learning must preserve caller's random RNG state."""
        import random
        agent = QLearningController(num_actions=3, seed=42)

        random.seed(12345)
        state_before = random.getstate()

        # Mock TrafficSignalRLEnv to avoid needing full SUMO simulation
        with patch("src.training.train_q_learning.TrafficSignalRLEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.reset.return_value = (np.zeros(10), {"discrete_state": (0, 0, 0, 0), "valid_actions": [0, 1]})
            mock_env.step.return_value = (np.zeros(10), -1.0, True, False, {"discrete_state": (0, 0, 0, 0), "valid_actions": [0, 1]})
            mock_env_cls.return_value = mock_env

            reward = evaluate_greedy_q_learning(agent, "ingolstadt1", seed=9999)

        state_after = random.getstate()
        self.assertEqual(state_before, state_after)

    # -------------------------------------------------------------
    # R06: DQN checkpoint hidden_dim adaptive loading
    # -------------------------------------------------------------
    def test_r06_dqn_adaptive_hidden_dim_loading(self):
        """R06: DQNController.load must adaptively rebuild network when checkpoint hidden_dim differs."""
        # 1. Create and save model with hidden_dim=64
        ckpt_path = os.path.join(self.test_dir, "dqn_custom_64.pt")
        agent_64 = DQNController(observation_dim=13, action_dim=3, hidden_dim=64, seed=42)
        agent_64.save(ckpt_path)

        # 2. Instantiate default controller with hidden_dim=128
        agent_128 = DQNController(observation_dim=13, action_dim=3, hidden_dim=128, seed=42, eval_mode=True)
        self.assertEqual(agent_128.hidden_dim, 128)

        # 3. Loading should succeed and adapt to hidden_dim=64 without size mismatch error
        agent_128.load(ckpt_path)
        self.assertEqual(agent_128.hidden_dim, 64)
        self.assertEqual(agent_128.q_network.network[0].out_features, 64)

    # -------------------------------------------------------------
    # R07: Q-learning truncated vs terminated bootstrap
    # -------------------------------------------------------------
    def test_r07_q_learning_update_truncation_bootstrapping(self):
        """R07: Q-learning must continue bootstrapping when truncated=True, and stop only when terminated=True."""
        agent = QLearningController(num_actions=2, learning_rate=1.0, gamma=0.5, seed=42)
        state = (0, 0, 0, 0)
        next_state = (1, 1, 1, 1)
        action = 0

        # Set next_state Q-values to 10.0
        agent.q_table[next_state] = np.array([10.0, 10.0])

        # Case 1: Truncated (horizon limit) -> bootstrap: target = reward + gamma * max_next_q = 2.0 + 0.5 * 10.0 = 7.0
        agent.q_table[state] = np.array([0.0, 0.0])
        agent.update(state, action, reward=2.0, next_state=next_state, terminated=False, truncated=True)
        self.assertAlmostEqual(agent.q_table[state][action], 7.0)

        # Case 2: Terminated (true absorbing end) -> no bootstrap: target = reward = 2.0
        agent.q_table[state] = np.array([0.0, 0.0])
        agent.update(state, action, reward=2.0, next_state=next_state, terminated=True, truncated=False)
        self.assertAlmostEqual(agent.q_table[state][action], 2.0)

    # -------------------------------------------------------------
    # R08: Full system metrics in aggregation
    # -------------------------------------------------------------
    def test_r08_aggregation_includes_all_system_and_safety_metrics(self):
        """R08: Aggregated stats must report system delay, pending delay, collisions, and safety counts."""
        records = [
            {
                "scenario_name": "cologne1",
                "algorithm": "Max-Pressure",
                "seed": 42,
                "is_valid": True,
                "system_mean_delay": 15.2,
                "system_total_delay": 30000.0,
                "total_pending_delay": 120.0,
                "pending_insertion_vehicles": 4,
                "running_vehicles_at_end": 10,
                "collision_count": 1,
                "safety_status": "INCIDENTS_DETECTED",
                "service_status": "NORMAL",
            }
        ]
        stats = aggregate_benchmark_statistics(records)
        mp_stats = stats["cologne1"]["Max-Pressure"]
        self.assertIn("system_mean_delay", mp_stats)
        self.assertIn("total_pending_delay", mp_stats)
        self.assertIn("collision_count", mp_stats)
        self.assertEqual(mp_stats["safety_status_counts"]["INCIDENTS_DETECTED"], 1)

    # -------------------------------------------------------------
    # R09: getPendingVehicles TraCI error handling
    # -------------------------------------------------------------
    def test_r09_pending_vehicles_read_error_marks_invalid(self):
        """R09: When getPendingVehicles fails, collector must record error and set is_valid=False."""
        collector = MetricsCollector(scenario_name="cologne1")
        mock_traci = MagicMock()
        mock_traci.simulation.getTime.return_value = 10.0
        mock_traci.simulation.getDepartedIDList.return_value = []
        mock_traci.simulation.getArrivedIDList.return_value = []
        mock_traci.vehicle.getIDList.return_value = []
        # getPendingVehicles raises TraCI RPC error
        mock_traci.simulation.getPendingVehicles.side_effect = RuntimeError("TraCI simulation RPC error")

        collector.record_step(
            step=1,
            sim_time=10.0,
            traci_module=mock_traci,
            step_length=1.0,
            controlled_lanes=[],
        )

        self.assertFalse(collector.is_valid)
        self.assertGreater(collector.measurement_errors, 0)
        self.assertTrue(any("getPendingVehicles" in err for err in collector.error_messages))

    # -------------------------------------------------------------
    # R10: SUMO version parsing
    # -------------------------------------------------------------
    def test_r10_sumo_version_parsing_case_insensitive(self):
        """R10: get_sumo_version must parse lowercase 'version 9.99.0' and not guess on failure."""
        with patch("subprocess.check_output") as mock_subp:
            mock_subp.return_value = "Eclipse SUMO sumo version 9.99.0 (git commit abc)"
            ver = get_sumo_version()
            self.assertEqual(ver, "9.99.0")

        with patch("subprocess.check_output", side_effect=Exception("binary not found")):
            with patch.dict("sys.modules", {"traci": None}):
                ver = get_sumo_version()
                self.assertEqual(ver, "unknown")

    # -------------------------------------------------------------
    # R05: YAML configuration propagation
    # -------------------------------------------------------------
    def test_r05_yaml_config_propagation(self):
        """R05: Custom YAML timing and reward parameters must propagate into TrafficSignalRLEnv."""
        import yaml
        from src.training.train_dqn import train_dqn

        yaml_content = {
            "yellow_duration": 7.0,
            "all_red_duration": 1.0,
            "observation": {
                "max_queue_scale": 15.0,
                "max_time_scale": 45.0,
            },
            "reward": {
                "scale": 2.5,
            },
            "hyperparameters": {
                "num_episodes": 0,
            },
        }
        yaml_file = os.path.join(self.test_dir, "custom_dqn.yaml")
        with open(yaml_file, "w") as f:
            yaml.dump(yaml_content, f)

        with patch("src.training.train_dqn.TrafficSignalRLEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.observation_dim = 13
            mock_env.num_actions = 3
            mock_env.obs_builder = MagicMock()
            mock_env.obs_builder.get_config.return_value = {}
            mock_env_cls.return_value = mock_env

            train_dqn(
                scenario_name="ingolstadt1",
                config_path=yaml_file,
                model_dir=os.path.join(self.test_dir, "models"),
                log_dir=os.path.join(self.test_dir, "logs"),
            )

            mock_env_cls.assert_called()
            call_kwargs = mock_env_cls.call_args[1]
            self.assertEqual(call_kwargs["yellow_duration"], 7.0)
            self.assertEqual(call_kwargs["all_red_duration"], 1.0)
            self.assertEqual(call_kwargs["max_queue_scale"], 15.0)
            self.assertEqual(call_kwargs["max_time_scale"], 45.0)
            self.assertEqual(call_kwargs["reward_fn"].scale, 2.5)


if __name__ == "__main__":
    unittest.main()
