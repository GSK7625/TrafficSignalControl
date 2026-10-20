"""Unit tests for main.py command-line interface."""

import json
import sys
from unittest.mock import patch, MagicMock
import pytest

import main


class TestMainCLI:
    """Test suite for main.py argument parsing and entrypoint execution."""

    def test_parse_args_no_args_loads_yaml_defaults(self):
        with patch.object(sys, "argv", ["main.py"]):
            args = main.parse_args()
            assert args.config == "configs/config.yaml"
            assert args.algorithm is None

    def test_main_missing_config_raises_file_not_found(self):
        with patch.object(sys, "argv", ["main.py", "--config", "nonexistent_config.yaml"]):
            with pytest.raises(FileNotFoundError, match="Bắt buộc phải có file cấu hình YAML"):
                main.main()

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

    @patch("main.run_single_experiment")
    def test_main_cli_overrides_dictionary_and_preserves_yaml(self, mock_run, tmp_path):
        custom_yaml = tmp_path / "custom.yaml"
        custom_yaml.write_text(
            "simulation:\n"
            "  algorithm: fixed-time\n"
            "  dataset: cologne1\n"
            "  gui: false\n"
            "  steps: 50\n"
            "  seed: 99\n"
            "  output_dir: custom_results\n"
            "safety:\n"
            "  yellow_duration: 4.0\n",
            encoding="utf-8",
        )
        mock_run.return_value = ({"status": "SUCCESS", "is_valid": True}, {})

        # CLI overrides algorithm to max-pressure and steps to 200, but omits seed and output_dir
        with patch.object(sys, "argv", ["main.py", "-c", str(custom_yaml), "-a", "max-pressure", "-s", "200"]):
            main.main()

        mock_run.assert_called_once()
        cfg = mock_run.call_args[0][0]
        assert cfg.algorithm == "Max-Pressure"
        assert cfg.steps == 200
        # Preserved from YAML
        assert cfg.scenario_name == "cologne1"
        assert cfg.seed == 99
        assert cfg.output_dir == "custom_results"
        assert cfg.timing_overrides["yellow_duration"] == 4.0

    def test_main_raises_when_algorithm_unspecified(self, tmp_path):
        empty_yaml = tmp_path / "empty.yaml"
        empty_yaml.write_text("simulation: {}\n", encoding="utf-8")
        with patch.object(sys, "argv", ["main.py", "-c", str(empty_yaml)]):
            with pytest.raises(ValueError, match="Chưa xác định thuật toán điều khiển"):
                main.main()

    def test_cli_forwards_explicit_disabled_safety_limits(self, tmp_path, monkeypatch):
        path = tmp_path / "config.yaml"
        path.write_text("simulation:\n  algorithm: max-pressure\nsafety:\n  max_green: null\n  max_red: null\n", encoding="utf-8")
        seen = []

        def run(config, **kwargs):
            seen.append(config)
            return {"status": "SUCCESS"}, {}

        monkeypatch.setattr(main, "run_single_experiment", run)
        monkeypatch.setattr(sys, "argv", ["main.py", "-c", str(path)])
        main.main()
        assert seen[0].timing_overrides == {"max_green": None, "max_red": None}


