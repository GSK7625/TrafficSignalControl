"""Tests for CLI main.py with Max Pressure controller and comparison mode."""

import argparse
from unittest.mock import MagicMock, patch
import pytest

from main import main, parse_args


def test_parse_args_controller_options():
    """Verify parse_args handles --controller max_pressure and default fixed_time."""
    # Default controller
    args_default = parse_args(["--scenario", "cologne1"])
    assert args_default.controller == "fixed_time"

    # Explicit fixed_time
    args_fixed = parse_args(["--scenario", "cologne1", "--controller", "fixed_time"])
    assert args_fixed.controller == "fixed_time"

    # Explicit max_pressure
    args_mp = parse_args(["--scenario", "cologne1", "--controller", "max_pressure"])
    assert args_mp.controller == "max_pressure"


def test_parse_args_compare_flag():
    """Verify parse_args handles --compare flag."""
    args_no_compare = parse_args(["--scenario", "cologne1"])
    assert not args_no_compare.compare

    args_compare = parse_args(["--compare"])
    assert args_compare.compare is True


def test_parse_args_invalid_controller():
    """Verify parse_args rejects invalid controller choices."""
    with pytest.raises(SystemExit):
        parse_args(["--scenario", "cologne1", "--controller", "invalid_controller"])


def test_main_cli_scenario_max_pressure(capsys):
    """Test main execution with --scenario cologne1 --controller max_pressure."""
    mock_res = {
        "scenario": "cologne1",
        "controller_type": "max_pressure",
        "average_waiting_time": 9.97,
        "average_travel_time": 43.86,
        "mean_queue_length": 4.93,
        "max_queue_length": 28,
        "average_time_loss": 21.12,
        "throughput": 1999,
        "total_steps": 3600,
        "tripinfo_path": "results/cologne1_max_pressure_tripinfo.xml",
    }

    with patch("main.evaluate_scenario", return_value=mock_res) as mock_eval, \
         patch("main.export_evaluation_json") as mock_json, \
         patch("main.plot_comparison_chart") as mock_plot:

        ret = main(["--scenario", "cologne1", "--controller", "max_pressure", "--max-steps", "100"])

        assert ret == 0
        mock_eval.assert_called_once_with(
            scenario_name="cologne1",
            max_steps=100,
            gui=False,
            output_dir="results",
            data_dir="data",
            controller_type="max_pressure",
        )
        mock_json.assert_called_once()
        mock_plot.assert_called_once()

        captured = capsys.readouterr().out
        assert "cologne1" in captured
        assert "max_pressure" in captured


def test_main_cli_all_max_pressure(capsys):
    """Test main execution with --all --controller max_pressure."""
    mock_results = {
        "cologne1": {
            "scenario": "cologne1",
            "controller_type": "max_pressure",
            "average_waiting_time": 10.0,
            "average_travel_time": 45.0,
            "mean_queue_length": 5.0,
            "max_queue_length": 30,
            "average_time_loss": 22.0,
            "throughput": 2000,
            "total_steps": 3600,
        },
        "ingolstadt1": {
            "scenario": "ingolstadt1",
            "controller_type": "max_pressure",
            "average_waiting_time": 12.0,
            "average_travel_time": 50.0,
            "mean_queue_length": 4.0,
            "max_queue_length": 25,
            "average_time_loss": 24.0,
            "throughput": 1700,
            "total_steps": 3600,
        },
    }

    with patch("main.evaluate_multiple", return_value=mock_results) as mock_eval:
        ret = main(["--all", "--controller", "max_pressure", "--max-steps", "200"])

        assert ret == 0
        mock_eval.assert_called_once_with(
            scenarios=["cologne1", "ingolstadt1"],
            max_steps=200,
            gui=False,
            output_dir="results",
            data_dir="data",
            controller_type="max_pressure",
        )

        captured = capsys.readouterr().out
        assert "cologne1" in captured
        assert "ingolstadt1" in captured
        assert "MAX_PRESSURE" in captured


def test_main_cli_compare_default_controllers(capsys):
    """Test main execution with --compare defaulting to ['fixed_time', 'max_pressure']."""
    mock_comp = {
        "cologne1": {
            "fixed_time": {"average_waiting_time": 70.0, "throughput": 1950, "total_steps": 3600},
            "max_pressure": {"average_waiting_time": 10.0, "throughput": 2000, "total_steps": 3600},
        },
        "ingolstadt1": {
            "fixed_time": {"average_waiting_time": 20.0, "throughput": 1690, "total_steps": 3600},
            "max_pressure": {"average_waiting_time": 12.0, "throughput": 1710, "total_steps": 3600},
        },
    }

    with patch("main.evaluate_comparison", return_value=mock_comp) as mock_comp_eval:
        ret = main(["--compare", "--max-steps", "150"])

        assert ret == 0
        mock_comp_eval.assert_called_once_with(
            scenarios=["cologne1", "ingolstadt1"],
            controllers=["fixed_time", "max_pressure"],
            max_steps=150,
            gui=False,
            output_dir="results",
            data_dir="data",
        )

        captured = capsys.readouterr().out
        assert "SO SÁNH" in captured
        assert "fixed_time, max_pressure" in captured


def test_main_cli_compare_single_scenario(capsys):
    """Test main execution with --compare and specific --scenario."""
    mock_comp = {
        "cologne1": {
            "fixed_time": {"average_waiting_time": 70.0, "throughput": 1950, "total_steps": 3600},
            "max_pressure": {"average_waiting_time": 10.0, "throughput": 2000, "total_steps": 3600},
        },
    }

    with patch("main.evaluate_comparison", return_value=mock_comp) as mock_comp_eval:
        ret = main(["--compare", "--scenario", "cologne1"])

        assert ret == 0
        mock_comp_eval.assert_called_once_with(
            scenarios=["cologne1"],
            controllers=["fixed_time", "max_pressure"],
            max_steps=3600,
            gui=False,
            output_dir="results",
            data_dir="data",
        )


def test_main_cli_scenario_error_handling(capsys):
    """Test main handles simulation exceptions gracefully by returning exit code 2."""
    with patch("main.evaluate_scenario", side_effect=RuntimeError("SUMO connection error")):
        ret = main(["--scenario", "cologne1", "--controller", "max_pressure"])
        assert ret == 2
        captured = capsys.readouterr().err
        assert "Lỗi trong quá trình mô phỏng" in captured


def test_main_cli_compare_custom_controllers(capsys):
    """Test main execution with --compare and custom --controllers flag."""
    mock_comp = {
        "cologne1": {
            "fixed_time": {"average_waiting_time": 70.0, "throughput": 1950, "total_steps": 3600},
            "max_pressure": {"average_waiting_time": 10.0, "throughput": 2000, "total_steps": 3600},
            "q_learning": {"average_waiting_time": 90.0, "throughput": 1100, "total_steps": 3600},
        },
    }

    with patch("main.evaluate_comparison", return_value=mock_comp) as mock_comp_eval:
        ret = main([
            "--compare",
            "--scenario", "cologne1",
            "--controllers", "fixed_time", "max_pressure", "q_learning",
        ])

        assert ret == 0
        mock_comp_eval.assert_called_once_with(
            scenarios=["cologne1"],
            controllers=["fixed_time", "max_pressure", "q_learning"],
            max_steps=3600,
            gui=False,
            output_dir="results",
            data_dir="data",
        )


def test_main_cli_compare_with_all_flag(capsys):
    """Test main execution with --compare --all targets all default scenarios."""
    mock_comp = {
        "cologne1": {"fixed_time": {"average_waiting_time": 70.0}},
        "ingolstadt1": {"fixed_time": {"average_waiting_time": 20.0}},
    }

    with patch("main.evaluate_comparison", return_value=mock_comp) as mock_comp_eval:
        ret = main(["--compare", "--all"])

        assert ret == 0
        mock_comp_eval.assert_called_once_with(
            scenarios=["cologne1", "ingolstadt1"],
            controllers=["fixed_time", "max_pressure"],
            max_steps=3600,
            gui=False,
            output_dir="results",
            data_dir="data",
        )


def test_main_cli_no_args_shows_help(capsys):
    """Test main execution with no args returns 1 and shows guidance."""
    ret = main([])
    assert ret == 1
    captured = capsys.readouterr().out
    assert "Lỗi cú pháp: Vui lòng chỉ định" in captured

