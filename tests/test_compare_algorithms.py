import json
from src.evaluation.compare_algorithms import generate_comparison_report


def test_generate_comparison_report_includes_safety_and_system_delay(tmp_path):
    # Mock benchmark_summary.json
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
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(mock_summary, f)

    report_path = tmp_path / "benchmark_comparison.md"
    eval_dir = str(tmp_path)

    res = generate_comparison_report(
        eval_dir=eval_dir,
        phase6_summary=str(summary_path),
        output_report=str(report_path),
    )

    assert "cologne1" in res
    with open(report_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Must contain System Delay and collision/teleport info
    assert "System Delay" in md_text
    assert "Va chạm" in md_text or "Collision" in md_text
    assert "INCIDENTS_DETECTED" in md_text or "Cảnh báo" in md_text or "4" in md_text


def test_generate_comparison_report_handles_failed_and_total_collisions(tmp_path):
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
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(mock_summary, f)

    report_path = tmp_path / "benchmark_comparison.md"
    eval_dir = str(tmp_path)

    generate_comparison_report(
        eval_dir=eval_dir,
        phase6_summary=str(summary_path),
        output_report=str(report_path),
    )

    with open(report_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    # Must transparently display FAILED status for Q-Learning
    assert "FAILED" in md_text
    assert "Safety constraint mismatch" in md_text
    # Must display total 5 collisions with max 4
    assert "5 / 5 (max 4)" in md_text
    assert "tổng cộng 5 va chạm" in md_text


def test_draw_metric_bar_helper():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.evaluation.plot_comparison import _draw_metric_bar

    fig, ax = plt.subplots()
    bars = _draw_metric_bar(ax, ["A", "B"], [10.0, 20.0], [1.0, 2.0], ylabel="Wait Time", title="Comparison", fmt="%.1fs")
    assert ax.get_ylabel() == "Wait Time"
    assert ax.get_title() == "Comparison"
    assert len(bars) == 2
    plt.close(fig)
