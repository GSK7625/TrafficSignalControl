import pytest
from src.evaluation.experiment_runner import get_sumo_version, compute_improvement


def test_get_sumo_version():
    ver = get_sumo_version()
    assert ver != "unknown"
    assert ver != "2", f"SUMO version should not be '2', got '{ver}'"
    assert "." in ver, f"SUMO version should contain a dot, got '{ver}'"


def test_compute_improvement():
    # Lower is better (waiting time, queue, collisions)
    assert compute_improvement(20.0, 10.0, higher_is_better=False) == 50.0
    assert compute_improvement(10.0, 20.0, higher_is_better=False) == -100.0
    assert compute_improvement(0.0, 0.0, higher_is_better=False) == 0.0
    # Baseline 0 collisions, algorithm 4 collisions -> must be negative (degradation)
    deg = compute_improvement(0.0, 4.0, higher_is_better=False)
    assert deg is None  # Relative percent has no nonzero denominator.

    # Higher is better (throughput)
    assert compute_improvement(100.0, 150.0, higher_is_better=True) == 50.0
    assert compute_improvement(150.0, 100.0, higher_is_better=True) == pytest.approx(-33.333, abs=0.01)
    assert compute_improvement(0.0, 10.0, higher_is_better=True) is None
