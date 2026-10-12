"""Unit tests for Phase 6 Experiment Runner, metrics validation, and statistical aggregation."""

import math
import os
import unittest
import numpy as np

from src.evaluation.experiment_runner import (
    SCENARIO_DEMAND,
    compute_improvement,
    validate_run_metrics,
    aggregate_benchmark_statistics,
)
from src.controllers.q_learning import QLearningController
from src.controllers.dqn import DQNController


class TestExperimentRunnerMetrics(unittest.TestCase):
    """Tests improvement calculations, anomaly detection, and statistical aggregation."""

    def test_compute_improvement_lower_is_better(self):
        # Waiting time reduced from 60.0 to 30.0 -> 50% improvement
        imp = compute_improvement(60.0, 30.0, higher_is_better=False)
        self.assertAlmostEqual(imp, 50.0, places=2)

        # Waiting time increased from 60.0 to 90.0 -> -50% (degradation)
        imp_neg = compute_improvement(60.0, 90.0, higher_is_better=False)
        self.assertAlmostEqual(imp_neg, -50.0, places=2)

    def test_compute_improvement_higher_is_better(self):
        # Throughput increased from 1000 to 1500 -> 50% improvement
        imp = compute_improvement(1000.0, 1500.0, higher_is_better=True)
        self.assertAlmostEqual(imp, 50.0, places=2)

        # Throughput decreased from 1000 to 800 -> -20%
        imp_neg = compute_improvement(1000.0, 800.0, higher_is_better=True)
        self.assertAlmostEqual(imp_neg, -20.0, places=2)

    def test_compute_improvement_edge_cases(self):
        # Baseline is 0
        self.assertEqual(compute_improvement(0.0, 10.0, higher_is_better=False), 0.0)
        self.assertEqual(compute_improvement(0.0, 10.0, higher_is_better=True), 0.0)

        # NaN / Infinity inputs
        self.assertEqual(compute_improvement(float("nan"), 10.0), 0.0)
        self.assertEqual(compute_improvement(10.0, float("nan")), 0.0)
        self.assertEqual(compute_improvement(float("inf"), 10.0), 0.0)
        self.assertEqual(compute_improvement(10.0, float("-inf")), 0.0)

    def test_validate_run_metrics_valid(self):
        valid_summary = {
            "total_steps": 3600,
            "total_completed_vehicles": 1950,
            "mean_waiting_time": 25.5,
            "tripinfo_mean_waiting_time": 25.2,
            "mean_queue_length": 5.2,
            "mean_travel_time": 60.4,
            "total_time_loss": 50000.0,
        }
        is_valid, anomalies = validate_run_metrics(valid_summary, total_demand=2015)
        self.assertTrue(is_valid)
        self.assertEqual(len(anomalies), 0)

    def test_validate_run_metrics_anomalies(self):
        # Negative waiting time
        bad_summary = {
            "total_steps": 3600,
            "total_completed_vehicles": 1500,
            "mean_waiting_time": -5.0,
            "mean_queue_length": 2.0,
        }
        is_valid, anomalies = validate_run_metrics(bad_summary, total_demand=1716)
        self.assertFalse(is_valid)
        self.assertTrue(any("Negative waiting time" in a for a in anomalies))

        # Negative queue length
        bad_queue = {
            "total_steps": 3600,
            "total_completed_vehicles": 1500,
            "mean_waiting_time": 10.0,
            "mean_queue_length": -1.5,
        }
        is_valid, anomalies = validate_run_metrics(bad_queue, total_demand=1716)
        self.assertFalse(is_valid)
        self.assertTrue(any("Negative queue length" in a for a in anomalies))

        # Impossible completion rate (> 105%)
        overflow_completed = {
            "total_steps": 3600,
            "total_completed_vehicles": 3000,
            "mean_waiting_time": 10.0,
            "mean_queue_length": 2.0,
        }
        is_valid, anomalies = validate_run_metrics(overflow_completed, total_demand=1716)
        self.assertFalse(is_valid)
        self.assertTrue(any("Completion rate exceeds 100%" in a for a in anomalies))

        # NaN metric
        nan_summary = {
            "total_steps": 3600,
            "total_completed_vehicles": 1500,
            "mean_waiting_time": float("nan"),
            "mean_queue_length": 2.0,
        }
        is_valid, anomalies = validate_run_metrics(nan_summary, total_demand=1716)
        self.assertFalse(is_valid)
        self.assertTrue(any("NaN" in a for a in anomalies))

        # Total lock: 0 completed in 3600s
        locked_summary = {
            "total_steps": 3600,
            "total_completed_vehicles": 0,
            "mean_waiting_time": 100.0,
            "mean_queue_length": 50.0,
        }
        is_valid, anomalies = validate_run_metrics(locked_summary, total_demand=1716)
        self.assertFalse(is_valid)
        self.assertTrue(any("0 completed vehicles" in a for a in anomalies))

    def test_aggregate_benchmark_statistics(self):
        runs = [
            {
                "scenario_name": "cologne1",
                "algorithm": "Fixed-Time",
                "seed": 42,
                "is_valid": True,
                "tripinfo_mean_waiting_time": 60.0,
                "mean_queue_length": 30.0,
                "total_completed_vehicles": 1900,
                "completion_rate_pct": 94.3,
                "throughput_vph": 1900.0,
                "tripinfo_mean_travel_time": 100.0,
                "tripinfo_total_time_loss": 150000.0,
                "teleport_count": 0,
                "phase_change_count": 120,
            },
            {
                "scenario_name": "cologne1",
                "algorithm": "Fixed-Time",
                "seed": 101,
                "is_valid": True,
                "tripinfo_mean_waiting_time": 62.0,
                "mean_queue_length": 32.0,
                "total_completed_vehicles": 1910,
                "completion_rate_pct": 94.8,
                "throughput_vph": 1910.0,
                "tripinfo_mean_travel_time": 102.0,
                "tripinfo_total_time_loss": 152000.0,
                "teleport_count": 0,
                "phase_change_count": 120,
            },
            {
                "scenario_name": "cologne1",
                "algorithm": "DQN",
                "seed": 42,
                "is_valid": True,
                "tripinfo_mean_waiting_time": 12.0,
                "mean_queue_length": 6.0,
                "total_completed_vehicles": 1995,
                "completion_rate_pct": 99.0,
                "throughput_vph": 1995.0,
                "tripinfo_mean_travel_time": 50.0,
                "tripinfo_total_time_loss": 60000.0,
                "teleport_count": 0,
                "phase_change_count": 350,
                "episode_reward": -4500.0,
            },
        ]

        summary = aggregate_benchmark_statistics(runs, baseline_algo="Fixed-Time")
        self.assertIn("cologne1", summary)
        cologne_stats = summary["cologne1"]

        # Check Fixed-Time mean waiting time: (60 + 62) / 2 = 61.0
        self.assertEqual(cologne_stats["Fixed-Time"]["tripinfo_mean_waiting_time"]["mean"], 61.0)
        self.assertEqual(cologne_stats["Fixed-Time"]["tripinfo_mean_waiting_time"]["min"], 60.0)
        self.assertEqual(cologne_stats["Fixed-Time"]["tripinfo_mean_waiting_time"]["max"], 62.0)
        self.assertEqual(cologne_stats["Fixed-Time"]["run_count"], 2)

        # Check DQN improvement vs Fixed-Time:
        # Waiting time: (61.0 - 12.0) / 61.0 * 100 ~ 80.33%
        dqn_improvements = cologne_stats["DQN"]["improvements_vs_baseline"]
        self.assertIn("tripinfo_mean_waiting_time_improvement_pct", dqn_improvements)
        self.assertAlmostEqual(dqn_improvements["tripinfo_mean_waiting_time_improvement_pct"], 80.33, places=1)

    def test_scenario_ground_truth_demand(self):
        self.assertEqual(SCENARIO_DEMAND["cologne1"], 2015)
        self.assertEqual(SCENARIO_DEMAND["ingolstadt1"], 1716)
        self.assertEqual(SCENARIO_DEMAND["cologne3"], 4494)


class TestCheckpointLoadingAndEvalMode(unittest.TestCase):
    """Tests that RL controllers load checkpoints and run in strict evaluation mode."""

    def test_q_learning_checkpoint_eval_mode(self):
        model_path = "models/q_learning/q_learning_ingolstadt1_best.json"
        if os.path.isfile(model_path):
            agent = QLearningController(num_actions=3, seed=42, eval_mode=True)
            agent.load(model_path)
            agent.eval_mode = True
            agent.epsilon = 0.0
            self.assertTrue(agent.eval_mode)
            self.assertEqual(agent.epsilon, 0.0)
            self.assertGreater(len(agent.q_table), 0)

            # Verification: update in eval_mode should do nothing (return 0.0)
            state = list(agent.q_table.keys())[0]
            err = agent.update(state=state, action=0, reward=-10.0, next_state=state, done=False)
            self.assertEqual(err, 0.0)

    def test_dqn_checkpoint_eval_mode(self):
        model_path = "models/dqn/dqn_ingolstadt1_best.pt"
        if os.path.isfile(model_path):
            agent = DQNController(
                observation_dim=13,
                action_dim=3,
                seed=42,
                eval_mode=True,
            )
            agent.load(model_path)
            agent.eval_mode = True
            agent.epsilon = 0.0
            self.assertTrue(agent.eval_mode)
            self.assertEqual(agent.epsilon, 0.0)

            # Verification: train_step in eval_mode returns None without updating
            loss = agent.train_step()
            self.assertIsNone(loss)


if __name__ == "__main__":
    unittest.main()
