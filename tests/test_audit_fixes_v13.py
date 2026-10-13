"""Tests for audit fixes v13:

1. Decoupling baseline decision cadence from RL decision interval.
2. Checkpoint safety constraint validation and accurate metadata logging.
3. CLI error handling for missing RL checkpoints.
4. Distinguishing clearance transition duration from actual yellow duration.
5. Reporting terminology and metric definitions.
"""

import json
import os
import pytest
from unittest.mock import MagicMock, patch

from src.evaluation.experiment_runner import (
    ExperimentConfig,
    run_single_experiment,
    resolve_effective_decision_interval,
)


class TestAuditFixesV13:
    """Audit fixes v13 test suite."""

    def test_baseline_decision_interval_cadence_resolution(self):
        """Task 1: Test that baseline algorithms resolve to 1.0s decision interval by default,

        while RL algorithms default to 5.0s, preventing the 30s -> 32s elongation.
        """
        # Baseline: Fixed-Time with decision_interval=None should resolve to 1.0s
        cfg_ft = ExperimentConfig(scenario_name="cologne1", algorithm="Fixed-Time")
        assert resolve_effective_decision_interval(cfg_ft) == 1.0

        # Baseline: Max-Pressure with decision_interval=None should resolve to 1.0s
        cfg_mp = ExperimentConfig(scenario_name="cologne1", algorithm="Max-Pressure")
        assert resolve_effective_decision_interval(cfg_mp) == 1.0

        # RL: Q-Learning without checkpoint params should default to 5.0s
        cfg_ql = ExperimentConfig(scenario_name="cologne1", algorithm="Q-Learning")
        assert resolve_effective_decision_interval(cfg_ql) == 5.0

        # RL: DQN without checkpoint params should default to 5.0s
        cfg_dqn = ExperimentConfig(scenario_name="cologne1", algorithm="DQN")
        assert resolve_effective_decision_interval(cfg_dqn) == 5.0

        # Explicit override in config should always be respected
        cfg_custom = ExperimentConfig(scenario_name="cologne1", algorithm="Fixed-Time", decision_interval=2.5)
        assert resolve_effective_decision_interval(cfg_custom) == 2.5

    def test_checkpoint_constraint_mismatch_raises_value_error(self, tmp_path):
        """Task 2: If checkpoint specifies safety constraints different from ExperimentConfig,

        run_single_experiment must raise ValueError (or record as failed with error).
        """
        # Create a dummy Q-Learning checkpoint with min_green=10.0 in config (cfg default is 5.0)
        dummy_ckpt = {
            "config": {
                "num_actions": 8,
                "learning_rate": 0.1,
                "gamma": 0.99,
                "epsilon": 0.0,
                "epsilon_min": 0.0,
                "epsilon_decay": 0.995,
                "min_green": 10.0,
                "yellow_duration": 3.0,
                "all_red_duration": 0.0,
                "decision_interval": 5.0,
            },
            "q_table": {},
            "action_counts": [0] * 8,
            "total_updates": 0,
        }
        ckpt_file = tmp_path / "q_mismatch.json"
        with open(ckpt_file, "w") as f:
            json.dump(dummy_ckpt, f)

        cfg = ExperimentConfig(
            scenario_name="cologne1",
            algorithm="Q-Learning",
            model_path=str(ckpt_file),
            min_green=5.0,  # Conflict: 5.0 vs 10.0
        )
        rec = run_single_experiment(cfg)
        assert rec["status"] == "FAILED"
        assert not rec["is_valid"]
        assert "Safety constraint mismatch" in rec["error_message"]
        assert "min_green" in rec["error_message"]

    def test_checkpoint_constraint_match_records_accurate_metadata(self, tmp_path):
        """Task 2: When checkpoint constraints match ExperimentConfig,

        result dictionary records accurate effective parameters.
        """
        dummy_ckpt = {
            "config": {
                "num_actions": 8,
                "learning_rate": 0.1,
                "gamma": 0.99,
                "epsilon": 0.0,
                "epsilon_min": 0.0,
                "epsilon_decay": 0.995,
                "min_green": 5.0,
                "yellow_duration": 3.0,
                "all_red_duration": 0.0,
                "decision_interval": 5.0,
            },
            "q_table": {},
            "action_counts": [0] * 8,
            "total_updates": 0,
        }
        ckpt_file = tmp_path / "q_match.json"
        with open(ckpt_file, "w") as f:
            json.dump(dummy_ckpt, f)

        cfg = ExperimentConfig(
            scenario_name="cologne1",
            algorithm="Q-Learning",
            model_path=str(ckpt_file),
            min_green=5.0,
            yellow_duration=3.0,
            all_red_duration=0.0,
        )

        with patch("src.evaluation.experiment_runner.TrafficSignalRLEnv") as mock_env_cls:
            mock_env = MagicMock()
            mock_env.num_actions = 8
            mock_env.state_discretizer = MagicMock()
            mock_env.reset.return_value = ([0.0], {"valid_actions": [0], "discrete_state": (0, 0, 0, 0)})
            mock_env.step.return_value = ([0.0], 0.0, True, False, {"valid_actions": [0], "discrete_state": (0, 0, 0, 0)})
            mock_env.get_summary.return_value = {
                "total_steps": 3600,
                "total_completed_vehicles": 1950,
                "mean_waiting_time": 12.0,
                "system_mean_delay": 15.0,
                "system_total_delay": 75.0,
                "mean_queue_length": 2.0,
                "mean_travel_time": 40.0,
                "total_time_loss": 100.0,
            }
            mock_env_cls.return_value = mock_env

            rec = run_single_experiment(cfg)
            assert rec["status"] == "SUCCESS"
            assert rec["is_valid"]
            assert rec["min_green"] == 5.0
            assert rec["yellow_duration"] == 3.0
            assert rec["all_red_duration"] == 0.0
            assert rec["decision_interval"] == 5.0

    def test_cli_missing_checkpoint_fails_without_flag(self):
        """Task 3: Running main.py with missing checkpoint must fail with sys.exit(1)

        unless --allow-untrained is explicitly supplied.
        """
        import subprocess

        # 1. Without --allow-untrained: must exit with code 1 and error message
        cmd = [
            ".\\.venv\\Scripts\\python.exe",
            "main.py",
            "--dataset", "cologne1",
            "--controller", "q-learning",
            "--model", "non_existent_model_checkpoint.json",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode != 0
        assert "not found" in res.stdout or "not found" in res.stderr
        assert "--allow-untrained" in res.stdout or "--allow-untrained" in res.stderr

        # 2. With --allow-untrained: must accept and proceed (run 2 steps)
        cmd_allowed = cmd + ["--allow-untrained", "--steps", "2"]
        res_allowed = subprocess.run(cmd_allowed, capture_output=True, text=True)
        assert res_allowed.returncode == 0
        assert "using untrained agent (--allow-untrained supplied)" in res_allowed.stdout

    def test_transition_vs_actual_yellow_tracking(self):
        """Task 4: SafePhaseManager and summary metrics must accurately distinguish

        between clearance transition time and actual yellow light time.
        """
        from src.phase_manager import SafePhaseManager

        # Cologne1 phases:
        # Phase 0: srrrrrrrrrGGGGGrrrrr
        # Phase 1: GGGGGrrrrrsrrrrrrrrr
        # Phase 2: GGGggrrrrrGGGggrrrrr (1->2 has NO 'y' characters!)
        phases = [
            "srrrrrrrrrGGGGGrrrrr",
            "GGGGGrrrrrsrrrrrrrrr",
            "GGGggrrrrrGGGggrrrrr",
        ]
        mgr = SafePhaseManager(
            tls_id="tls1",
            green_phases=phases,
            yellow_duration=3.0,
            all_red_duration=0.0,
            min_green=5.0,
            initial_phase=0,
        )

        # Advance phase 0 for 5s green
        for _ in range(5):
            mgr.update(1.0)
        assert mgr.total_transition_time == 0.0
        assert mgr.actual_yellow_time == 0.0

        # Switch 0 -> 1 (contains yellow 'y')
        assert mgr.request_phase(1)
        for _ in range(3):
            mgr.update(1.0)
        assert mgr.get_current_phase() == 1
        assert mgr.total_transition_time == 3.0
        assert mgr.actual_yellow_time == 3.0

        # Advance phase 1 for 5s green
        for _ in range(5):
            mgr.update(1.0)

        # Switch 1 -> 2 (does NOT contain yellow 'y' - holding delay only)
        assert mgr.request_phase(2)
        for _ in range(3):
            mgr.update(1.0)
        assert mgr.get_current_phase() == 2
        # Transition time increased by 3.0s -> total 6.0s
        assert mgr.total_transition_time == 6.0
        # Actual yellow time remained 3.0s (no new yellow light displayed!)
        assert mgr.actual_yellow_time == 3.0

    def test_reporting_terminology_and_metric_definitions(self, tmp_path):
        """Task 5: Report generation must not refer to Fixed-Time as 'chu kỳ 30s'

        and must correctly expose completed_mean_waiting_time and system_mean_delay.
        """
        from src.experiments.run_experiments import write_markdown_report
        from src.metrics import MetricsCollector

        # Check write_markdown_report terminology
        report_file = tmp_path / "report.md"
        mock_results = {
            "cologne1": {
                "Fixed-Time": {"mean_waiting_time": 68.29, "system_mean_delay": 82.65},
            }
        }
        write_markdown_report(mock_results, str(report_file))
        content = report_file.read_text(encoding="utf-8")
        assert "chu kỳ 30s" not in content
        assert "30s xanh mỗi pha" in content

        # Check metrics summary structure
        collector = MetricsCollector("cologne1")
        summary = collector.get_summary()
        assert "completed_mean_waiting_time" in summary
        assert "system_mean_delay" in summary
        assert "system_total_delay" in summary
        assert "total_pending_delay" in summary






