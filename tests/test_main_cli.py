"""Unit tests for main.py command-line interface."""

import json
import sys
from unittest.mock import patch, MagicMock
import pytest

import main


class TestMainCLI:
    """Test suite for main.py argument parsing and entrypoint execution."""

    def test_parse_args_no_args_exits_with_code_1(self, capsys):
        with patch.object(sys, "argv", ["main.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main.parse_args()
            assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "usage:" in captured.err.lower() or "sumo traffic signal control platform" in captured.err.lower()

    def test_parse_args_valid_arguments(self):
        with patch.object(sys, "argv", ["main.py", "-a", "max-pressure", "-d", "ingolstadt1", "-s", "50", "--gui"]):
            args = main.parse_args()
            assert args.algorithm == "max-pressure"
            assert args.dataset == "ingolstadt1"
            assert args.steps == 50
            assert args.gui is True
            assert args.model is None

    def test_parse_args_invalid_algorithm_exits(self):
        with patch.object(sys, "argv", ["main.py", "-a", "unsupported-algo"]):
            with pytest.raises(SystemExit) as exc_info:
                main.parse_args()
            assert exc_info.value.code != 0

    @patch("main.run_single_experiment")
    def test_main_success_prints_json_summary(self, mock_run, capsys):
        mock_run.return_value = (
            {"status": "SUCCESS", "is_valid": True},
            {"mean_waiting_time": 12.34, "completion_rate_pct": 99.5},
        )
        with patch.object(sys, "argv", ["main.py", "-a", "fixed-time", "-s", "10"]):
            main.main()

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["mean_waiting_time"] == 12.34
        assert data["completion_rate_pct"] == 99.5
        mock_run.assert_called_once()
        config = mock_run.call_args[0][0]
        assert config.algorithm == "Fixed-Time"
        assert config.steps == 10

    @patch("main.run_single_experiment")
    def test_main_failed_experiment_raises_runtime_error(self, mock_run):
        mock_run.return_value = (
            {"status": "FAILED", "error_message": "Simulation crashed unexpectedly"},
            {},
        )
        with patch.object(sys, "argv", ["main.py", "-a", "fixed-time"]):
            with pytest.raises(RuntimeError, match="Simulation crashed unexpectedly"):
                main.main()

    @patch("main.run_single_experiment")
    def test_main_missing_model_provides_helpful_hint(self, mock_run):
        mock_run.return_value = (
            {"status": "FAILED", "error_message": "Model file not found: models/q_learning/best.json"},
            {},
        )
        with patch.object(sys, "argv", ["main.py", "-a", "q-learning", "-d", "cologne3"]):
            with pytest.raises(RuntimeError) as exc_info:
                main.main()
            msg = str(exc_info.value)
            assert "Gợi ý: Kịch bản 'cologne3' chưa có checkpoint sẵn." in msg
            assert "python -m src.training.train_q_learning --dataset cologne3" in msg
