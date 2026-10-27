"""Unit and integration tests for Multi-Scenario Evaluation and Reporting Flow."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from src.evaluate import (
    evaluate_scenario,
    evaluate_multiple,
    export_evaluation_json,
    generate_comparison_markdown,
    plot_comparison_chart,
)
from main import parse_args, main


@pytest.fixture
def sample_evaluation_results():
    """Provides sample 5-metric evaluation results for cologne1 and ingolstadt1."""
    return {
        "cologne1": {
            "scenario": "cologne1",
            "average_waiting_time": 14.25,
            "average_travel_time": 68.50,
            "mean_queue_length": 2.35,
            "max_queue_length": 12,
            "average_time_loss": 21.80,
            "throughput": 420,
            "total_steps": 1200,
            "tripinfo_path": "results/cologne1_tripinfo.xml",
        },
        "ingolstadt1": {
            "scenario": "ingolstadt1",
            "average_waiting_time": 28.60,
            "average_travel_time": 105.10,
            "mean_queue_length": 5.80,
            "max_queue_length": 24,
            "average_time_loss": 39.40,
            "throughput": 310,
            "total_steps": 1500,
            "tripinfo_path": "results/ingolstadt1_tripinfo.xml",
        },
    }


def test_generate_comparison_markdown(sample_evaluation_results):
    """Test generating a markdown table comparing the 5 traffic metrics."""
    md = generate_comparison_markdown(sample_evaluation_results)

    assert isinstance(md, str)
    assert "| Metric |" in md or "| Chỉ số" in md or "Metric" in md
    assert "cologne1" in md
    assert "ingolstadt1" in md
    assert "Average Waiting Time" in md
    assert "Average Travel Time" in md
    assert "Mean Queue Length" in md
    assert "Average Time Loss" in md
    assert "Throughput" in md
    assert "14.25" in md
    assert "28.60" in md
    assert "420" in md
    assert "310" in md


def test_export_evaluation_json(sample_evaluation_results, tmp_path):
    """Test exporting evaluation results to JSON file."""
    output_path = tmp_path / "results" / "fixed_time_evaluation.json"
    result_file = export_evaluation_json(sample_evaluation_results, str(output_path))

    assert os.path.isfile(result_file)
    with open(result_file, "r", encoding="utf-8") as f:
        loaded = json.load(f)

    assert "cologne1" in loaded
    assert "ingolstadt1" in loaded
    assert loaded["cologne1"]["average_waiting_time"] == 14.25
    assert loaded["ingolstadt1"]["throughput"] == 310


def test_plot_comparison_chart(sample_evaluation_results, tmp_path):
    """Test generating comparative bar chart using matplotlib."""
    output_path = tmp_path / "results" / "fixed_time_metrics.png"
    result_img = plot_comparison_chart(sample_evaluation_results, str(output_path))

    assert os.path.isfile(result_img)
    assert os.path.getsize(result_img) > 1000  # Non-trivial image file


def test_evaluate_scenario_mocked(tmp_path):
    """Test evaluate_scenario runs simulation and extracts metrics."""
    scenario = "cologne1"
    output_dir = tmp_path / "results"

    mock_sim_result = MagicMock()
    mock_sim_result.total_steps = 100
    mock_sim_result.mean_queue_length = 2.5
    mock_sim_result.max_queue_length = 8
    mock_sim_result.queue_history = [1, 2, 3]

    mock_summary = {
        "average_waiting_time": 12.0,
        "average_travel_time": 55.0,
        "mean_queue_length": 2.5,
        "max_queue_length": 8,
        "average_time_loss": 18.0,
        "throughput": 150,
        "total_steps": 100,
        "queue_history": [1, 2, 3],
        "tripinfo_metrics": {},
    }

    with patch("src.evaluate.SimulationRunner") as mock_runner_cls, \
         patch("src.evaluate.MetricsCollector") as mock_collector_cls:

        mock_runner = mock_runner_cls.return_value
        mock_runner.run.return_value = mock_sim_result
        mock_runner.incoming_lanes = ["lane1", "lane2"]

        mock_collector = mock_collector_cls.return_value
        mock_collector.get_summary.return_value = mock_summary

        res = evaluate_scenario(
            scenario_name=scenario,
            max_steps=100,
            gui=False,
            output_dir=str(output_dir),
            data_dir="data",
        )

        assert res["scenario"] == scenario
        assert res["average_waiting_time"] == 12.0
        assert res["average_travel_time"] == 55.0
        assert res["mean_queue_length"] == 2.5
        assert res["average_time_loss"] == 18.0
        assert res["throughput"] == 150
        assert mock_runner.run.called


def test_evaluate_multiple_orchestration(tmp_path):
    """Test evaluate_multiple evaluates scenarios, exports json and plots chart."""
    output_dir = tmp_path / "results"

    sample_metric = {
        "average_waiting_time": 10.0,
        "average_travel_time": 50.0,
        "mean_queue_length": 1.5,
        "max_queue_length": 5,
        "average_time_loss": 15.0,
        "throughput": 100,
        "total_steps": 50,
        "tripinfo_path": "fake.xml",
    }

    with patch("src.evaluate.evaluate_scenario") as mock_eval_scen:
        mock_eval_scen.side_effect = lambda scenario_name, **kwargs: {
            **sample_metric,
            "scenario": scenario_name,
        }

        results = evaluate_multiple(
            scenarios=["cologne1", "ingolstadt1"],
            max_steps=50,
            output_dir=str(output_dir),
            data_dir="data",
        )

        assert "cologne1" in results
        assert "ingolstadt1" in results
        json_path = output_dir / "fixed_time_evaluation.json"
        png_path = output_dir / "fixed_time_metrics.png"
        assert os.path.isfile(json_path)
        assert os.path.isfile(png_path)


def test_parse_args():
    """Test CLI argument parsing for main.py."""
    args1 = parse_args(["--scenario", "cologne1"])
    assert args1.scenario == "cologne1"
    assert not args1.all

    args2 = parse_args(["--all"])
    assert args2.all
    assert args2.scenario is None

    args3 = parse_args(["--scenario", "ingolstadt1", "--gui", "--max-steps", "500"])
    assert args3.scenario == "ingolstadt1"
    assert args3.gui
    assert args3.max_steps == 500


def test_main_cli_scenario(capsys):
    """Test main execution with single --scenario flag."""
    sample_res = {
        "scenario": "cologne1",
        "average_waiting_time": 15.0,
        "average_travel_time": 60.0,
        "mean_queue_length": 2.0,
        "max_queue_length": 6,
        "average_time_loss": 20.0,
        "throughput": 200,
        "total_steps": 300,
        "tripinfo_path": "results/cologne1_tripinfo.xml",
    }
    with patch("main.evaluate_scenario", return_value=sample_res) as mock_eval:
        ret = main(["--scenario", "cologne1", "--max-steps", "300"])
        assert ret == 0
        captured = capsys.readouterr().out
        assert "cologne1" in captured
        assert "Average Waiting Time" in captured


def test_main_cli_all(capsys, sample_evaluation_results):
    """Test main execution with --all flag."""
    with patch("main.evaluate_multiple", return_value=sample_evaluation_results) as mock_eval:
        ret = main(["--all", "--max-steps", "300"])
        assert ret == 0
        captured = capsys.readouterr().out
        assert "cologne1" in captured
        assert "ingolstadt1" in captured
        assert "Average Waiting Time" in captured


def test_main_no_args(capsys):
    """Test main execution with no args returns error code and shows guidance."""
    ret = main([])
    assert ret != 0
    captured = capsys.readouterr().out
    assert "Vui lòng chỉ định" in captured or "error" in captured.lower() or "usage" in captured.lower()
