"""Unit and integration tests for Multi-Scenario Evaluation and Reporting Flow."""

import json
import os
import tempfile
from unittest.mock import MagicMock, patch
import pytest

from src.evaluate import (
    evaluate_scenario,
    evaluate_multiple,
    evaluate_comparison,
    export_evaluation_json,
    generate_comparison_markdown,
    generate_algorithm_comparison_markdown,
    plot_comparison_chart,
    plot_multi_controller_comparison,
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


@pytest.fixture
def sample_comparison_results():
    """Provides sample comparative results across scenarios and controllers."""
    return {
        "cologne1": {
            "fixed_time": {
                "scenario": "cologne1",
                "controller_type": "fixed_time",
                "average_waiting_time": 68.25,
                "average_travel_time": 145.50,
                "mean_queue_length": 8.35,
                "max_queue_length": 32,
                "average_time_loss": 85.80,
                "throughput": 1950,
                "total_steps": 3600,
                "tripinfo_path": "results/cologne1_fixed_time_tripinfo.xml",
            },
            "max_pressure": {
                "scenario": "cologne1",
                "controller_type": "max_pressure",
                "average_waiting_time": 24.10,
                "average_travel_time": 82.30,
                "mean_queue_length": 3.12,
                "max_queue_length": 14,
                "average_time_loss": 32.40,
                "throughput": 2010,
                "total_steps": 3600,
                "tripinfo_path": "results/cologne1_max_pressure_tripinfo.xml",
            },
        },
        "ingolstadt1": {
            "fixed_time": {
                "scenario": "ingolstadt1",
                "controller_type": "fixed_time",
                "average_waiting_time": 28.60,
                "average_travel_time": 105.10,
                "mean_queue_length": 5.80,
                "max_queue_length": 24,
                "average_time_loss": 39.40,
                "throughput": 1692,
                "total_steps": 3600,
                "tripinfo_path": "results/ingolstadt1_fixed_time_tripinfo.xml",
            },
            "max_pressure": {
                "scenario": "ingolstadt1",
                "controller_type": "max_pressure",
                "average_waiting_time": 18.20,
                "average_travel_time": 88.40,
                "mean_queue_length": 3.40,
                "max_queue_length": 15,
                "average_time_loss": 26.10,
                "throughput": 1708,
                "total_steps": 3600,
                "tripinfo_path": "results/ingolstadt1_max_pressure_tripinfo.xml",
            },
        },
    }


def test_evaluate_scenario_with_controller_type(tmp_path):
    """Test evaluate_scenario correctly passes controller_type to SimulationRunner."""
    scenario = "cologne1"
    output_dir = tmp_path / "results"

    mock_sim_result = MagicMock()
    mock_sim_result.total_steps = 100

    mock_summary = {
        "average_waiting_time": 15.0,
        "average_travel_time": 50.0,
        "mean_queue_length": 2.0,
        "max_queue_length": 6,
        "average_time_loss": 18.0,
        "throughput": 120,
        "total_steps": 100,
        "queue_history": [1, 2],
        "tripinfo_metrics": {},
    }

    with patch("src.evaluate.SimulationRunner") as mock_runner_cls, \
         patch("src.evaluate.MetricsCollector") as mock_collector_cls:

        mock_runner = mock_runner_cls.return_value
        mock_runner.run.return_value = mock_sim_result
        mock_runner.incoming_lanes = ["lane1"]

        mock_collector = mock_collector_cls.return_value
        mock_collector.get_summary.return_value = mock_summary

        res = evaluate_scenario(
            scenario_name=scenario,
            max_steps=100,
            gui=False,
            output_dir=str(output_dir),
            data_dir="data",
            controller_type="max_pressure",
        )

        assert res["scenario"] == scenario
        assert res["controller_type"] == "max_pressure"
        assert res["average_waiting_time"] == 15.0
        assert "max_pressure" in res["tripinfo_path"]
        # Verify SimulationRunner received controller_type="max_pressure"
        mock_runner_cls.assert_called_once()
        _, kwargs = mock_runner_cls.call_args
        assert kwargs.get("controller_type") == "max_pressure"


def test_evaluate_multiple_with_controller_type(tmp_path):
    """Test evaluate_multiple accepts controller_type and writes {controller}_evaluation.json."""
    output_dir = tmp_path / "results"
    sample_metric = {
        "scenario": "cologne1",
        "controller_type": "max_pressure",
        "average_waiting_time": 12.0,
        "average_travel_time": 45.0,
        "mean_queue_length": 1.8,
        "max_queue_length": 5,
        "average_time_loss": 14.0,
        "throughput": 150,
        "total_steps": 60,
        "tripinfo_path": "fake_mp.xml",
    }

    with patch("src.evaluate.evaluate_scenario") as mock_eval_scen:
        mock_eval_scen.return_value = sample_metric

        results = evaluate_multiple(
            scenarios=["cologne1"],
            max_steps=60,
            output_dir=str(output_dir),
            data_dir="data",
            controller_type="max_pressure",
        )

        assert "cologne1" in results
        assert results["cologne1"]["controller_type"] == "max_pressure"
        mock_eval_scen.assert_called_once_with(
            scenario_name="cologne1",
            max_steps=60,
            gui=False,
            output_dir=str(output_dir),
            data_dir="data",
            controller_type="max_pressure",
        )

        json_path = output_dir / "max_pressure_evaluation.json"
        assert os.path.isfile(json_path)


def test_plot_multi_controller_comparison(sample_comparison_results, tmp_path):
    """Test plot_multi_controller_comparison generates a valid PNG comparing algorithms."""
    chart_path = tmp_path / "results" / "algorithm_comparison.png"
    result_path = plot_multi_controller_comparison(
        sample_comparison_results,
        output_path=str(chart_path),
    )

    assert os.path.isfile(result_path)
    assert os.path.getsize(result_path) > 1000


def test_generate_algorithm_comparison_markdown(sample_comparison_results):
    """Test generating a comprehensive markdown report for algorithm comparison."""
    report_md = generate_algorithm_comparison_markdown(sample_comparison_results)

    assert isinstance(report_md, str)
    assert "Fixed-Time" in report_md or "fixed_time" in report_md
    assert "Max-Pressure" in report_md or "max_pressure" in report_md
    assert "cologne1" in report_md
    assert "ingolstadt1" in report_md
    assert "algorithm_comparison.png" in report_md
    assert "Average Waiting Time" in report_md
    assert "Throughput" in report_md
    # Check that improvement or comparison is analyzed
    assert "%" in report_md


def test_evaluate_comparison_orchestration(tmp_path):
    """Test evaluate_comparison runs multiple controllers and exports all comparison artifacts."""
    output_dir = tmp_path / "results"

    def mock_eval(scenario_name, controller_type="fixed_time", **kwargs):
        return {
            "scenario": scenario_name,
            "controller_type": controller_type,
            "average_waiting_time": 20.0 if controller_type == "fixed_time" else 10.0,
            "average_travel_time": 70.0 if controller_type == "fixed_time" else 50.0,
            "mean_queue_length": 3.0 if controller_type == "fixed_time" else 1.5,
            "max_queue_length": 10 if controller_type == "fixed_time" else 5,
            "average_time_loss": 30.0 if controller_type == "fixed_time" else 15.0,
            "throughput": 100 if controller_type == "fixed_time" else 110,
            "total_steps": 50,
            "tripinfo_path": f"{scenario_name}_{controller_type}_tripinfo.xml",
        }

    with patch("src.evaluate.evaluate_scenario", side_effect=mock_eval) as mock_scen:
        comp_results = evaluate_comparison(
            scenarios=["cologne1", "ingolstadt1"],
            controllers=["fixed_time", "max_pressure"],
            max_steps=50,
            gui=False,
            output_dir=str(output_dir),
            data_dir="data",
        )

        assert mock_scen.call_count == 4
        assert "cologne1" in comp_results
        assert "ingolstadt1" in comp_results
        assert "fixed_time" in comp_results["cologne1"]
        assert "max_pressure" in comp_results["cologne1"]

        # Check artifact files exist
        json_file = output_dir / "comparison_evaluation.json"
        chart_file = output_dir / "algorithm_comparison.png"
        report_file = output_dir / "COMPARISON_REPORT.md"

        assert os.path.isfile(json_file)
        assert os.path.isfile(chart_file)
        assert os.path.isfile(report_file)

        # Verify JSON content
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["cologne1"]["fixed_time"]["average_waiting_time"] == 20.0
        assert data["cologne1"]["max_pressure"]["average_waiting_time"] == 10.0



def test_evaluate_scenario_with_q_learning(tmp_path):
    """Test evaluate_scenario correctly handles controller_type='q_learning'."""
    scenario = "cologne1"
    output_dir = tmp_path / "results"

    mock_sim_result = MagicMock()
    mock_sim_result.total_steps = 100

    mock_summary = {
        "average_waiting_time": 11.0,
        "average_travel_time": 42.0,
        "mean_queue_length": 1.5,
        "max_queue_length": 5,
        "average_time_loss": 12.0,
        "throughput": 160,
        "total_steps": 100,
        "queue_history": [1, 2],
        "tripinfo_metrics": {},
    }

    with patch("src.evaluate.SimulationRunner") as mock_runner_cls, \
         patch("src.evaluate.MetricsCollector") as mock_collector_cls:

        mock_runner = mock_runner_cls.return_value
        mock_runner.run.return_value = mock_sim_result
        mock_runner.incoming_lanes = ["lane1"]

        mock_collector = mock_collector_cls.return_value
        mock_collector.get_summary.return_value = mock_summary

        res = evaluate_scenario(
            scenario_name=scenario,
            max_steps=100,
            gui=False,
            output_dir=str(output_dir),
            data_dir="data",
            controller_type="q_learning",
        )

        assert res["scenario"] == scenario
        assert res["controller_type"] == "q_learning"
        assert res["average_waiting_time"] == 11.0
        assert "q_learning" in res["tripinfo_path"]


def test_evaluate_comparison_3_controllers(tmp_path):
    """Test evaluate_comparison with 3 controllers (fixed_time, max_pressure, q_learning)."""
    output_dir = tmp_path / "results"

    def mock_eval(scenario_name, controller_type="fixed_time", **kwargs):
        return {
            "scenario": scenario_name,
            "controller_type": controller_type,
            "average_waiting_time": 20.0 if controller_type == "fixed_time" else (12.0 if controller_type == "max_pressure" else 8.0),
            "average_travel_time": 70.0,
            "mean_queue_length": 3.0,
            "max_queue_length": 10,
            "average_time_loss": 30.0,
            "throughput": 100,
            "total_steps": 50,
            "tripinfo_path": f"{scenario_name}_{controller_type}_tripinfo.xml",
        }

    with patch("src.evaluate.evaluate_scenario", side_effect=mock_eval):
        comp_results = evaluate_comparison(
            scenarios=["cologne1"],
            controllers=["fixed_time", "max_pressure", "q_learning"],
            max_steps=50,
            output_dir=str(output_dir),
        )
        assert "q_learning" in comp_results["cologne1"]
        assert comp_results["cologne1"]["q_learning"]["average_waiting_time"] == 8.0


def test_evaluate_comparison_defaults(tmp_path):
    """Test evaluate_comparison runs default scenarios and controllers when unconfigured."""
    output_dir = tmp_path / "results"

    def mock_eval(scenario_name, controller_type="fixed_time", **kwargs):
        return {
            "scenario": scenario_name,
            "controller_type": controller_type,
            "average_waiting_time": 25.0,
            "average_travel_time": 75.0,
            "mean_queue_length": 3.5,
            "max_queue_length": 10,
            "average_time_loss": 32.0,
            "throughput": 150,
            "total_steps": 50,
            "tripinfo_path": f"{scenario_name}_{controller_type}_tripinfo.xml",
        }

    with patch("src.evaluate.evaluate_scenario", side_effect=mock_eval) as mock_scen:
        comp_results = evaluate_comparison(
            output_dir=str(output_dir),
            data_dir="data",
            max_steps=50,
        )

        # Default scenarios: cologne1, ingolstadt1; default controllers: fixed_time, max_pressure
        assert mock_scen.call_count == 4
        assert "cologne1" in comp_results
        assert "ingolstadt1" in comp_results
        assert "fixed_time" in comp_results["cologne1"]
        assert "max_pressure" in comp_results["cologne1"]
        assert os.path.isfile(output_dir / "comparison_evaluation.json")
        assert os.path.isfile(output_dir / "algorithm_comparison.png")
        assert os.path.isfile(output_dir / "COMPARISON_REPORT.md")

