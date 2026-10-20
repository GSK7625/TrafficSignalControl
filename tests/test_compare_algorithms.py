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
        assert "Trạng thái" not in md_text
        assert "Lưu ý về Survivorship Bias" not in md_text

    def test_generate_comparison_report_with_algo_aliases(self, tmp_path):
        mock_summary = {
            "cologne1": {
                "ft": {
                    "tripinfo_mean_waiting_time": {"mean": 70.0, "std": 1.0},
                    "system_mean_delay": {"mean": 80.0, "std": 1.0},
                    "mean_queue_length": {"mean": 30.0, "std": 1.0},
                    "total_completed_vehicles": {"mean": 1000.0},
                    "completion_rate_pct": {"mean": 95.0},
                    "status": "SUCCESS",
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
        assert "NOT_RUN" not in md_text
        assert "Fixed-Time" in md_text
        assert "70.00" in md_text

    def test_report_handles_missing_metrics_and_zero_completion(self, tmp_path):
        records = [{
            "scenario_name": "cologne1", "algorithm": "Fixed-Time", "seed": 1,
            "status": "SUCCESS", "is_valid": True, "completion_rate_pct": 0.0,
            "collision_count": 0, "teleport_count": 0,
        }]
        path = tmp_path / "summary.json"
        path.write_text(json.dumps(aggregate_benchmark_statistics(records)), encoding="utf-8")
        report = tmp_path / "report.md"
        res = generate_comparison_report(eval_dir=str(tmp_path), phase6_summary=str(path), output_report=str(report))
        text = report.read_text(encoding="utf-8")
        assert "N/A" in text
        assert "0.0%" in text
        assert "Trạng thái" not in text
        assert "LOW_COMPLETION" not in text
        assert "None" not in text
        assert res["cologne1"]["Fixed-Time"]["service_status"] == "LOW_COMPLETION"

    def test_comparisons_reject_different_watchdog_limits(self):
        shared = {
            "scenario_name": "cologne1", "seed": 42, "is_valid": True, "status": "SUCCESS",
            "total_steps": 10, "observation_window_begin": 0, "observation_window_end": 10,
            "observation_window_duration": 10,
        }
        records = [dict(shared, algorithm=algo, effective_config={"steps": 10, "step_length": 1, "max_red": red})
                   for algo, red in (("Fixed-Time", None), ("Max-Pressure", 90))]
        with pytest.raises(ValueError, match="conditions differ"):
            aggregate_benchmark_statistics(records)

    def test_chart_generation_reads_custom_summary(self, tmp_path):
        import matplotlib
        matplotlib.use("Agg")
        from src.evaluation.plot_comparison import generate_all_benchmark_charts
        path = tmp_path / "summary.json"
        path.write_text(json.dumps({"custom": {"Fixed-Time": {
            "mean_queue_length": {"mean": 2, "std": 0},
            "total_completed_vehicles": {"mean": 5, "std": 0},
        }}}), encoding="utf-8")
        files = generate_all_benchmark_charts(summary_file=str(path), output_dir=str(tmp_path / "figures"), dashboard_only=True)
        assert len(files) == 1
        assert files[0].endswith("custom_benchmark_dashboard.png")



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

    def test_phase_allocation_does_not_mix_different_cohorts(self, tmp_path, monkeypatch):
        runs = [
            {"scenario_name": "cologne1", "algorithm": "Fixed-Time", "seed": 42, "is_valid": True,
             "cohort_fingerprint": "cohort_old",
             "output_dir": None, "phase_durations_per_tls": {"tls": {"0": 100, "1": 0}}},
            {"scenario_name": "cologne1", "algorithm": "Fixed-Time", "seed": 42, "is_valid": True,
             "cohort_fingerprint": "cohort_new",
             "output_dir": None, "phase_durations_per_tls": {"tls": {"0": 0, "1": 100}}},
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
        assert heights == [[0.0, 100.0]]

    def test_phase_allocation_with_selected_records(self, tmp_path, monkeypatch):
        selected = [
            {"scenario_name": "cologne1", "algorithm": "Fixed-Time", "seed": 42, "is_valid": True,
             "output_dir": None, "phase_durations_per_tls": {"tls": {"0": 20, "1": 80}}},
        ]
        heights = []
        original_save = plt.Figure.savefig

        def capture(figure, *args, **kwargs):
            heights.append([bar.get_height() for bar in figure.axes[0].patches])
            return original_save(figure, *args, **kwargs)

        monkeypatch.setattr(plt.Figure, "savefig", capture)
        files = plot_group_b_dynamics(output_dir=str(tmp_path), selected_records=selected)
        assert len(files) == 1
        assert heights == [[20.0, 80.0]]

    def test_group_b_dynamics_csv_without_pandas(self, tmp_path):
        csv_file = tmp_path / "step_metrics.csv"
        csv_file.write_text(
            "step,sim_time,queue_length,running_vehicles,cumulative_completed_vehicles,pending_vehicles\n"
            "0,0.0,1.0,2.0,0.0,0.0\n"
            "1,1.0,2.0,3.0,1.0,0.0\n"
            "2,2.0,1.5,2.5,2.0,1.0\n",
            encoding="utf-8",
        )
        selected = [
            {
                "scenario_name": "cologne1",
                "algorithm": "Fixed-Time",
                "seed": 42,
                "is_valid": True,
                "output_csv": str(csv_file),
                "phase_durations_per_tls": {"tls": {"0": 30.0}},
            }
        ]
        files = plot_group_b_dynamics(output_dir=str(tmp_path / "figures"), selected_records=selected)
        # Should generate 4 trajectory charts + 1 phase allocation chart = 5
        assert len(files) == 5
        filenames = {Path(f).name for f in files}
        assert "cologne1_queue_length_over_time.png" in filenames
        assert "cologne1_running_vehicles_over_time.png" in filenames
        assert "cologne1_cumulative_completed_over_time.png" in filenames
        assert "cologne1_pending_vehicles_over_time.png" in filenames
        assert "cologne1_fixed_time_phase_allocation.png" in filenames

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
