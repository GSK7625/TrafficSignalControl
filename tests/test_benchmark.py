"""Unit tests for multi-seed benchmark runner."""

import os
import pytest

from src.benchmark import calculate_mean_std, generate_benchmark_report_markdown, run_single_simulation, run_benchmark_matrix


def test_calculate_mean_std():
    """Test sample mean and standard deviation calculations."""
    assert calculate_mean_std([]) == (0.0, 0.0)
    assert calculate_mean_std([10.0]) == (10.0, 0.0)

    # [10, 20] -> mean = 15.0, var = ((10-15)^2 + (20-15)^2) / 1 = 50 -> std = sqrt(50) ~= 7.071
    mean, std = calculate_mean_std([10.0, 20.0])
    assert mean == 15.0
    assert pytest.approx(std, 0.001) == 7.0710678


def test_generate_benchmark_report_markdown():
    """Test markdown table generation formatting."""
    mock_data = {
        "scenario": "cologne1",
        "seeds": [42, 101],
        "algorithms": ["fixed_time", "max_pressure"],
        "max_steps": 100,
        "model_path": None,
        "aggregated_stats": {
            "fixed_time": {
                "average_waiting_time": {"mean": 25.0, "std": 2.0},
                "average_travel_time": {"mean": 60.0, "std": 3.0},
                "mean_queue_length": {"mean": 10.0, "std": 1.5},
                "average_time_loss": {"mean": 30.0, "std": 2.5},
                "throughput": {"mean": 100.0, "std": 5.0},
            },
            "max_pressure": {
                "average_waiting_time": {"mean": 15.0, "std": 1.0},
                "average_travel_time": {"mean": 45.0, "std": 2.0},
                "mean_queue_length": {"mean": 5.0, "std": 0.8},
                "average_time_loss": {"mean": 18.0, "std": 1.2},
                "throughput": {"mean": 110.0, "std": 4.0},
            },
        },
        "improvements_vs_fixed_time": {
            "max_pressure": {
                "average_waiting_time": 40.0,
                "average_travel_time": 25.0,
                "mean_queue_length": 50.0,
                "average_time_loss": 40.0,
                "throughput": 10.0,
            }
        },
    }

    report = generate_benchmark_report_markdown(mock_data)
    assert "# BÁO CÁO SO SÁNH HIỆU NĂNG" in report
    assert "cologne1" in report
    assert "Fixed-Time (Cố định)" in report
    assert "Max-Pressure (Áp suất cực đại)" in report
    assert "25.00 ± 2.00" in report
    assert "+40.00%" in report


def test_run_single_simulation_short():
    """Test single simulation run for 50 steps."""
    cfg_path = os.path.join("data", "cologne1", "cologne1.sumocfg")
    metrics = run_single_simulation(
        cfg_path=cfg_path,
        controller_type="fixed_time",
        seed=42,
        max_steps=50,
        use_libsumo=True,
    )
    assert "average_waiting_time" in metrics
    assert "average_travel_time" in metrics
    assert "mean_queue_length" in metrics
    assert "throughput" in metrics
    assert metrics["mean_queue_length"] >= 0.0


def test_run_benchmark_matrix_short(tmp_path):
    """Test short benchmark matrix across 2 seeds and 2 controllers."""
    out_dir = str(tmp_path / "bench_out")
    res = run_benchmark_matrix(
        scenario_name="cologne1",
        algorithms=["fixed_time", "max_pressure"],
        seeds=[42, 101],
        max_steps=50,
        use_libsumo=True,
        output_dir=out_dir,
        verbose=False,
    )

    assert "aggregated_stats" in res
    assert "fixed_time" in res["aggregated_stats"]
    assert "max_pressure" in res["aggregated_stats"]
    assert "improvements_vs_fixed_time" in res

    # Verify saved files
    assert os.path.isfile(os.path.join(out_dir, "benchmark_cologne1.json"))
    assert os.path.isfile(os.path.join(out_dir, "benchmark_cologne1.md"))
