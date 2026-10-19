"""Unit tests for benchmark report generation, plotting, and dashboard visualizations."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from src.evaluation.compare_algorithms import generate_comparison_report
from src.evaluation.experiment_runner import aggregate_benchmark_statistics
from src.evaluation.plot_comparison import (
    _draw_metric_bar,
    plot_benchmark_dashboard,
    plot_group_a_comparisons,
    plot_group_b_dynamics,
)


class TestGenerateComparisonReport:
    """Test suite for Markdown comparison report generation."""

    def test_generate_comparison_report_includes_safety_and_system_delay(self, tmp_path):
        mock_summary = {
            "cologne1": {
                "Fixed-Time": {
                    "tripinfo_mean_waiting_time": {"mean": 70.33, "std": 1.63},
                    "system_mean_delay": {"mean": 86.70, "std": 3.12},
                    "mean_queue_length": {"mean": 37.20, "std": 0.87},
                    "total_completed_vehicles": {"mean": 1953.0},
                    "completion_rate_pct": {"mean": 96.94},
                    "collision_count": {"mean": 0.0, "max": 0},
                    "teleport_count": {"mean": 0.0, "max": 0},
                    "safety_status": "SAFE",
                },
                "Max-Pressure": {
                    "tripinfo_mean_waiting_time": {"mean": 13.57, "std": 0.36},
                    "system_mean_delay": {"mean": 18.15, "std": 0.13},
                    "mean_queue_length": {"mean": 6.57, "std": 0.08},
                    "total_completed_vehicles": {"mean": 1997.0},
                    "completion_rate_pct": {"mean": 99.14},
                    "collision_count": {"mean": 1.33, "max": 4},
                    "teleport_count": {"mean": 1.33, "max": 4},
                    "safety_status": "INCIDENTS_DETECTED",
                },
            }
        }
        summary_path = tmp_path / "benchmark_summary.json"
        summary_path.write_text(json.dumps(mock_summary), encoding="utf-8")
        report_path = tmp_path / "benchmark_comparison.md"

        res = generate_comparison_report(
            eval_dir=str(tmp_path),
            phase6_summary=str(summary_path),
            output_report=str(report_path),
        )

        assert "cologne1" in res
        md_text = report_path.read_text(encoding="utf-8")
        assert "System Delay" in md_text
        assert "Va chạm" in md_text or "Collision" in md_text
        assert "INCIDENTS_DETECTED" in md_text or "Cảnh báo" in md_text or "4" in md_text

    def test_generate_comparison_report_handles_failed_and_total_collisions(self, tmp_path):
        mock_summary = {
            "cologne1": {
                "Fixed-Time": {
                    "tripinfo_mean_waiting_time": {"mean": 70.33, "std": 1.63},
                    "system_mean_delay": {"mean": 86.70, "std": 3.12},
                    "mean_queue_length": {"mean": 37.20, "std": 0.87},
                    "total_completed_vehicles": {"mean": 1953.0},
                    "completion_rate_pct": {"mean": 96.94},
                    "collision_count": {"mean": 0.0, "max": 0, "sum": 0},
                    "teleport_count": {"mean": 0.0, "max": 0, "sum": 0},
                    "safety_status": "SAFE",
                },
                "Max-Pressure": {
                    "tripinfo_mean_waiting_time": {"mean": 13.57, "std": 0.36},
                    "system_mean_delay": {"mean": 18.15, "std": 0.13},
                    "mean_queue_length": {"mean": 6.57, "std": 0.08},
                    "total_completed_vehicles": {"mean": 1997.0},
                    "completion_rate_pct": {"mean": 99.14},
                    "collision_count": {"mean": 1.67, "max": 4, "sum": 5},
                    "teleport_count": {"mean": 1.67, "max": 4, "sum": 5},
                    "safety_status": "INCIDENTS_DETECTED",
                },
                "Q-Learning": {
                    "status": "FAILED",
                    "is_valid": False,
                    "error_messages": ["Safety constraint mismatch (ckpt=1.0s vs cfg=0.0s)"],
                },
            }
        }
        summary_path = tmp_path / "benchmark_summary.json"
        summary_path.write_text(json.dumps(mock_summary), encoding="utf-8")
        report_path = tmp_path / "benchmark_comparison.md"

        generate_comparison_report(
            eval_dir=str(tmp_path),
            phase6_summary=str(summary_path),
            output_report=str(report_path),
        )

        md_text = report_path.read_text(encoding="utf-8")
        assert "FAILED" in md_text
        assert "Safety constraint mismatch" in md_text
        assert "5 / 5 (max 4)" in md_text
        assert "tổng cộng 5 va chạm" in md_text

    def test_report_handles_missing_metrics_and_zero_completion(self, tmp_path):
        records = [{
            "scenario_name": "cologne1", "algorithm": "Fixed-Time", "seed": 1,
            "status": "SUCCESS", "is_valid": True, "completion_rate_pct": 0.0,
            "collision_count": 0, "teleport_count": 0,
        }]
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(aggregate_benchmark_statistics(records)), encoding="utf-8")
        report = tmp_path / "report.md"
        generate_comparison_report(eval_dir=str(tmp_path), phase6_summary=str(path), output_report=str(report))
        text = report.read_text(encoding="utf-8")
        assert "N/A" in text
        assert "LOW_COMPLETION" in text


class TestPlottingAndVisualizations:
    """Test suite for matplotlib comparisons and dashboard generation."""

    def test_draw_metric_bar_helper(self):
        fig, ax = plt.subplots()
        bars = _draw_metric_bar(
            ax, ["A", "B"], [10.0, 20.0], [1.0, 2.0],
            ylabel="Wait Time", title="Comparison", fmt="%.1fs",
        )
        assert ax.get_ylabel() == "Wait Time"
        assert ax.get_title() == "Comparison"
        assert len(bars) == 2
        plt.close(fig)

    def test_no_baseline_does_not_generate_improvement_chart(self, tmp_path):
        path = tmp_path / "summary.json"
        path.write_text(
            json.dumps({"cologne3": {"Q-Learning": {"status": "SUCCESS", "is_valid": True,
                                                     "mean_queue_length": {"mean": 2.0, "std": 0}}}}),
            encoding="utf-8",
        )
        files = plot_group_a_comparisons(str(path), str(tmp_path))
        assert [Path(file).name for file in files] == ["cologne3_queue_length_comparison.png"]

    def test_phase_allocation_combines_seeds_before_plotting(self, tmp_path, monkeypatch):
        runs = [
            {"scenario_name": "cologne1", "algorithm": "Fixed-Time", "is_valid": True,
             "output_dir": None, "phase_durations_per_tls": {"tls": phases}}
            for phases in ({"0": 10, "1": 90}, {"0": 90, "1": 10})
        ]
        path = tmp_path / "runs.json"
        path.write_text(json.dumps(runs), encoding="utf-8")
        heights = []
        original_save = plt.Figure.savefig

        def capture(figure, *args, **kwargs):
            heights.append([bar.get_height() for bar in figure.axes[0].patches])
            return original_save(figure, *args, **kwargs)

        monkeypatch.setattr(plt.Figure, "savefig", capture)
        files = plot_group_b_dynamics(str(path), str(tmp_path))
        assert len(files) == 1
        assert heights == [[50.0, 50.0]]

    def test_comparison_plots_skip_failed_algorithms(self, tmp_path):
        metrics = {
            key: {"mean": 10.0, "std": 1.0} for key in (
                "tripinfo_mean_waiting_time", "mean_queue_length", "total_completed_vehicles",
                "tripinfo_total_time_loss", "completion_rate_pct",
            )
        }
        path = tmp_path / "summary.json"
        path.write_text(
            json.dumps({"cologne1": {
                "Fixed-Time": metrics, "Q-Learning": {"status": "FAILED", "is_valid": False},
            }}),
            encoding="utf-8",
        )
        plots = plot_group_a_comparisons(str(path), str(tmp_path / "plots"))
        assert len(plots) == 5
        assert {Path(p).name for p in plots} == {
            f"cologne1_{suffix}_comparison.png" for suffix in (
                "waiting_time", "queue_length", "throughput", "time_loss", "completion_rate",
            )
        }

    def test_dynamics_without_saved_run_details_does_not_crash(self, tmp_path):
        path = tmp_path / "runs.json"
        path.write_text(
            json.dumps([{
                "scenario_name": "cologne1", "algorithm": "Fixed-Time", "is_valid": True, "output_dir": None,
            }]),
            encoding="utf-8",
        )
        try:
            assert plot_group_b_dynamics(str(path), str(tmp_path / "plots")) == []
        finally:
            plt.close("all")

    def test_dashboard_handles_unavailable_waiting_metrics(self, tmp_path):
        path = tmp_path / "summary.json"
        path.write_text(
            json.dumps({"cologne1": {"Fixed-Time": {
                "tripinfo_mean_waiting_time": {"mean": None, "std": None, "available": False},
                "mean_queue_length": {"mean": 10.0, "std": 1.0},
                "total_completed_vehicles": {"mean": 0.0, "std": 0.0},
            }}}),
            encoding="utf-8",
        )
        try:
            figures = plot_benchmark_dashboard(str(path), str(tmp_path / "plots"))
            assert len(figures) == 1
            assert Path(figures[0]).is_file()
        finally:
            plt.close("all")
